"""
distillation_utils.py

Utilities for knowledge distillation from teacher (LAPA) hidden states to student (OpenVLA) models.

This module provides:
1. SequenceAggregationMLP: Converts variable-length sequences to fixed representations
2. FrameAlignmentStrategy: Aligns teacher and student hidden states across different frame rates
3. SimilarityMatrixDistillationLoss: KL divergence loss on similarity matrices
"""

from enum import Enum
from typing import Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregationMethod(Enum):
    """Methods for converting sequence dimension to single representation."""
    LAST = "last"  # Use only the last token in sequence
    MEAN = "mean"  # Average all tokens in sequence


class FrameAlignmentMode(Enum):
    """Methods for aligning teacher and student frames."""
    SUPERVISED_ONLY = "supervised_only"  # Only use frames with teacher supervision
    INTERPOLATED = "interpolated"  # Interpolate teacher states for all frames


class SequenceAggregationMLP(nn.Module):
    """
    Aggregates [batch, seq_len, hidden_dim] → [batch, hidden_dim] using either:
    - Last token: takes final hidden state
    - Mean: averages all tokens

    Used for TEACHER only (pure aggregation, no MLP).
    """

    def __init__(
        self,
        hidden_dim: int,
        aggregation_method: AggregationMethod = AggregationMethod.LAST,
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states (e.g., 4096)
            aggregation_method: How to aggregate sequence dimension (last or mean)
        """
        super().__init__()
        self.aggregation_method = aggregation_method
        self.hidden_dim = hidden_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            aggregated: [batch, hidden_dim]
        """
        if self.aggregation_method == AggregationMethod.LAST:
            # Take only the last token
            aggregated = hidden_states[:, -1, :]  # [batch, hidden_dim]
        elif self.aggregation_method == AggregationMethod.MEAN:
            # Average all tokens
            aggregated = hidden_states.mean(dim=1)  # [batch, hidden_dim]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        return aggregated


class StudentSequenceProjectionMLP(nn.Module):
    """
    Projects student hidden states through a bottleneck MLP.
    Converts [batch, seq_len, input_dim] → [batch, input_dim]

    Architecture:
        Input [batch, seq_len, 4096]
            ↓
        Aggregate (last or mean) → [batch, 4096]
            ↓
        Linear(4096 → 2048)
            ↓
        ReLU
            ↓
        Linear(2048 → 4096)
            ↓
        Output [batch, 4096]

    Used for STUDENT only (aggregation + bottleneck MLP).
    """

    def __init__(
        self,
        input_dim: int = 4096,
        bottleneck_dim: int = 2048,
        aggregation_method: AggregationMethod = AggregationMethod.LAST,
    ):
        """
        Args:
            input_dim: Input dimension from model (e.g., 4096 for OpenVLA)
            bottleneck_dim: Intermediate bottleneck dimension (e.g., 2048 for 50% compression)
            aggregation_method: How to aggregate sequence dimension (last or mean)
        """
        super().__init__()
        self.aggregation_method = aggregation_method
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # MLP: input_dim → bottleneck_dim → input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, input_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, input_dim]

        Returns:
            projected: [batch, input_dim]
        """
        # First: aggregate sequence dimension
        if self.aggregation_method == AggregationMethod.LAST:
            aggregated = hidden_states[:, -1, :]  # [batch, input_dim]
        elif self.aggregation_method == AggregationMethod.MEAN:
            aggregated = hidden_states.mean(dim=1)  # [batch, input_dim]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Convert to float32 for MLP computation (handle BFloat16/Float16 inputs)
        original_dtype = aggregated.dtype
        aggregated = aggregated.float()

        # Second: project through bottleneck MLP
        projected = self.mlp(aggregated)  # [batch, input_dim]

        # Convert back to original dtype
        projected = projected.to(original_dtype)

        return projected


class FrameAlignmentStrategy:
    """
    Handles alignment of teacher hidden states (every 12 frames) with student frames (every frame).

    **IMPORTANT**: Teacher hidden states are ALREADY AGGREGATED at this point (each is [hidden_dim] not [seq_len, hidden_dim]).
    Aggregation happens in TeacherHiddenStateLoader.load_episode() before alignment.

    Teacher frames: 0, 12, 24, 36, ... (each is a single aggregated vector [4096])
    Student frames: 0, 1, 2, 3, ..., 11, 12, ..., 23, 24, ...

    Two strategies:
    1. supervised_only: Only supervise frames 0, 12, 24, ... (with teacher supervision)
    2. interpolated: Interpolate between adjacent teacher frames for all student frames
       (e.g., frame 6 gets 0.5 * aggregated_state[0] + 0.5 * aggregated_state[12])
    """

    def __init__(self, mode: FrameAlignmentMode = FrameAlignmentMode.SUPERVISED_ONLY, teacher_stride: int = 12):
        """
        Args:
            mode: Alignment strategy
            teacher_stride: How many student frames between teacher samples (default: 12)
        """
        self.mode = mode
        self.teacher_stride = teacher_stride

    def align(
        self,
        batch_frame_indices: np.ndarray,  # [batch_size] - frame index for each sample
        teacher_hidden_states: Dict[int, torch.Tensor],  # {frame_idx: [hidden_dim]} (already aggregated)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Align teacher states to student batch frames.

        Args:
            batch_frame_indices: [batch_size] frame indices for each sample in the batch
            teacher_hidden_states: Dict mapping frame indices to aggregated teacher hidden states [hidden_dim]

        Returns:
            aligned_teacher_states: [batch_size, hidden_dim] aligned aggregated teacher states
            valid_mask: [batch_size] boolean mask indicating which samples have valid supervision
            interpolation_weights: [batch_size, 2] weights for interpolation (for debugging/analysis)
        """
        batch_size = len(batch_frame_indices)
        device = list(teacher_hidden_states.values())[0].device if teacher_hidden_states else torch.device("cpu")

        aligned_states = []
        valid_mask = []
        interp_weights = []

        for frame_idx in batch_frame_indices:
            frame_idx = int(frame_idx)

            if self.mode == FrameAlignmentMode.SUPERVISED_ONLY:
                # Only supervise frames that have teacher states (multiples of teacher_stride)
                if frame_idx % self.teacher_stride == 0 and frame_idx in teacher_hidden_states:
                    aligned_states.append(teacher_hidden_states[frame_idx])
                    valid_mask.append(True)
                    interp_weights.append([1.0, 0.0])
                else:
                    # Placeholder - will be masked out anyway
                    aligned_states.append(torch.zeros_like(list(teacher_hidden_states.values())[0]))
                    valid_mask.append(False)
                    interp_weights.append([0.0, 0.0])

            elif self.mode == FrameAlignmentMode.INTERPOLATED:
                # Interpolate between nearest teacher frames
                lower_frame = (frame_idx // self.teacher_stride) * self.teacher_stride
                upper_frame = lower_frame + self.teacher_stride

                has_lower = lower_frame in teacher_hidden_states
                has_upper = upper_frame in teacher_hidden_states

                if has_lower and has_upper:
                    # Both frames exist, interpolate
                    ratio = (frame_idx - lower_frame) / self.teacher_stride
                    state_lower = teacher_hidden_states[lower_frame]
                    state_upper = teacher_hidden_states[upper_frame]
                    interpolated = (1 - ratio) * state_lower + ratio * state_upper
                    aligned_states.append(interpolated)
                    valid_mask.append(True)
                    interp_weights.append([1 - ratio, ratio])

                elif has_lower:
                    # Only lower frame exists, use it
                    aligned_states.append(teacher_hidden_states[lower_frame])
                    valid_mask.append(True)
                    interp_weights.append([1.0, 0.0])

                elif has_upper:
                    # Only upper frame exists, use it
                    aligned_states.append(teacher_hidden_states[upper_frame])
                    valid_mask.append(True)
                    interp_weights.append([0.0, 1.0])

                else:
                    # No teacher states nearby - this is an error condition
                    available_frames = sorted(teacher_hidden_states.keys())
                    raise ValueError(
                        f"Frame {frame_idx} has no nearby teacher states in interpolated mode. "
                        f"Teacher stride is {self.teacher_stride}, but no teacher states found at "
                        f"frames {lower_frame} or {upper_frame}. "
                        f"Available teacher frames: {available_frames[:10]}{'...' if len(available_frames) > 10 else ''}. "
                        f"Check that your dataset frame indices match the teacher HDF5 file."
                    )

        # Stack all states
        if aligned_states:
            # Stack aggregated states: [batch_size, hidden_dim]
            stacked_states = torch.stack(aligned_states, dim=0)
        else:
            stacked_states = torch.zeros((batch_size, 4096), device=device)

        valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=device)
        interp_weights = torch.tensor(interp_weights, dtype=torch.float32, device=device)

        return stacked_states, valid_mask, interp_weights


class SimilarityMatrixDistillationLoss(nn.Module):
    """
    KL divergence loss between student and teacher similarity matrices.

    Key idea:
    - Student: H_s @ H_s.T = [batch, batch] similarity matrix (how similar are representations)
    - Teacher: H_t @ H_t.T = [batch, batch] similarity matrix
    - Loss: KL(softmax(teacher_sim), softmax(student_sim))

    This makes the loss invariant to absolute magnitudes and focuses on relative relationships.
    """

    def __init__(
        self,
        student_hidden_dim: int = 4096,
        teacher_hidden_dim: int = 4096,
        temperature: float = 1.0,
        normalize: bool = True,
        projection_dim: Optional[int] = None,
    ):
        """
        Args:
            student_hidden_dim: Dimension of student hidden states
            teacher_hidden_dim: Dimension of teacher hidden states
            temperature: Temperature for softmax (higher = softer)
            normalize: Whether to L2 normalize before computing similarity
            projection_dim: If specified, project to this dimension before similarity computation
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

        # Optional projection layer to match dimensions
        self.projection = None
        if projection_dim is not None:
            self.projection = nn.Linear(teacher_hidden_dim, projection_dim)

    def forward(
        self,
        student_hidden_states: torch.Tensor,  # [batch, hidden_dim]
        teacher_hidden_states: torch.Tensor,  # [batch, hidden_dim]
        valid_mask: Optional[torch.Tensor] = None,  # [batch] boolean mask
    ) -> torch.Tensor:
        """
        Args:
            student_hidden_states: [batch, hidden_dim]
            teacher_hidden_states: [batch, hidden_dim]
            valid_mask: [batch] boolean mask indicating valid supervision samples

        Returns:
            loss: scalar KL divergence loss
        """
        batch_size = student_hidden_states.size(0)

        # Convert to float32 for loss computation (handle BFloat16/Float16 inputs)
        student_hidden_states = student_hidden_states.float()
        teacher_hidden_states = teacher_hidden_states.float()

        # Optional projection
        if self.projection is not None:
            student_hidden_states = self.projection(student_hidden_states)
            teacher_hidden_states = self.projection(teacher_hidden_states)

        # Optional normalization
        if self.normalize:
            student_hidden_states = F.normalize(student_hidden_states, p=2, dim=1)
            teacher_hidden_states = F.normalize(teacher_hidden_states, p=2, dim=1)

        # Compute similarity matrices: [batch, batch]
        # sim[i,j] = cosine similarity between sample i and sample j
        student_sim = torch.mm(student_hidden_states, student_hidden_states.t())  # [batch, batch]
        teacher_sim = torch.mm(teacher_hidden_states, teacher_hidden_states.t())  # [batch, batch]

        # Mask out diagonal (self-similarity is always 1.0, not informative)
        # Focus on inter-sample relationships instead
        mask = torch.eye(batch_size, device=student_sim.device, dtype=torch.bool)
        student_sim = student_sim.masked_fill(mask, float('-inf'))
        teacher_sim = teacher_sim.masked_fill(mask, float('-inf'))

        # Apply temperature scaling
        student_sim = student_sim / self.temperature
        teacher_sim = teacher_sim / self.temperature

        # Convert to probability distributions
        student_prob = F.softmax(student_sim, dim=1)  # [batch, batch]
        teacher_prob = F.softmax(teacher_sim, dim=1)  # [batch, batch]

        # KL divergence: KL(P || Q) = sum(P * log(P/Q))
        kl_loss = F.kl_div(
            torch.log(student_prob + 1e-8),  # log Q (student)
            teacher_prob,  # P (teacher)
            reduction="batchmean",
        )

        # Apply valid mask if provided
        if valid_mask is not None:
            # Zero out loss for invalid samples
            # Note: This is approximate since KL is computed on the full batch
            # For more precise masking, we could compute KL per-sample
            n_valid = valid_mask.sum().float()
            if n_valid > 0:
                # Reweight: upscale loss from valid samples to maintain expected magnitude
                kl_loss = kl_loss * (batch_size / max(n_valid, 1))
            else:
                kl_loss = torch.tensor(0.0, device=student_hidden_states.device)

        return kl_loss


class TeacherHiddenStateLoader:
    """
    Loads teacher hidden states from HDF5 file and caches them for efficient access.
    """

    def __init__(self, h5_file_path: str):
        """
        Args:
            h5_file_path: Path to lapa_hidden_states.h5 file
        """
        self.h5_file_path = h5_file_path
        self.cache = {}
        self._h5_file = None

    def load_episode(self, episode_idx: int, aggregation_method: AggregationMethod = AggregationMethod.LAST) -> Dict[int, torch.Tensor]:
        """
        Load all teacher hidden states for a specific episode and aggregate each sequence.

        Args:
            episode_idx: Episode index to load
            aggregation_method: How to aggregate sequences (LAST token or MEAN of all tokens)

        Returns:
            Dict mapping frame indices to aggregated hidden states [4096] as torch tensors on CPU
        """
        cache_key = f"episode_{episode_idx}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            import h5py
        except ImportError:
            print("❌ h5py not installed. Install with: pip install h5py")
            return {}

        episode_states = {}

        try:
            with h5py.File(self.h5_file_path, "r") as f:
                episode_indices = f["episode_indices"][:]
                matching_indices = np.where(episode_indices == episode_idx)[0]

                for sample_idx in matching_indices:
                    hidden_state_flat = f["hidden_states"][sample_idx]
                    seq_len = f["seq_lengths"][sample_idx]
                    frame_idx = int(f["frame_indices"][sample_idx])

                    # Reshape to sequence form
                    hidden_state_2d = hidden_state_flat.reshape(seq_len, 4096)
                    hidden_state_tensor = torch.from_numpy(hidden_state_2d).float()

                    # Aggregate sequence to single representation
                    if aggregation_method == AggregationMethod.LAST:
                        aggregated = hidden_state_tensor[-1, :]  # Take last token [4096]
                    elif aggregation_method == AggregationMethod.MEAN:
                        aggregated = hidden_state_tensor.mean(dim=0)  # Average all tokens [4096]
                    else:
                        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

                    episode_states[frame_idx] = aggregated

        except Exception as e:
            print(f"Error loading teacher hidden states from {self.h5_file_path}: {e}")
            return {}

        # Cache the result
        self.cache[cache_key] = episode_states
        return episode_states

    def get_batch_teacher_states(
        self, episode_indices: np.ndarray, frame_indices: np.ndarray,
        aggregation_method: AggregationMethod = AggregationMethod.LAST
    ) -> Tuple[Dict[int, torch.Tensor], np.ndarray]:
        """
        Get teacher hidden states for a batch of samples.

        Args:
            episode_indices: [batch_size] episode indices
            frame_indices: [batch_size] frame indices
            aggregation_method: How to aggregate sequences (LAST or MEAN)

        Returns:
            teacher_states: Dict mapping frame indices to aggregated hidden states [4096]
            batch_frame_indices: [batch_size] frame indices for the batch
        """
        # For now, assume all samples in batch are from same episode
        # In practice, you may want to handle mixed episodes
        episode_idx = int(episode_indices[0])
        teacher_states = self.load_episode(episode_idx, aggregation_method)

        return teacher_states, frame_indices
