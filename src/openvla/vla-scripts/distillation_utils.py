"""
distillation_utils.py

Utilities for knowledge distillation from teacher (LAPA) hidden states to student (OpenVLA) models.

This module provides:
1. AggregationMethod: Enum for sequence aggregation methods (LAST or MEAN)
2. StudentSequenceProjectionMLP: Projects student hidden states through bottleneck MLP
3. SimilarityMatrixDistillationLoss: KL divergence loss on similarity matrices

Note: Teacher states are precomputed and aggregated offline using precompute_teacher_dataset.py
"""

from enum import Enum
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregationMethod(Enum):
    """Methods for converting sequence dimension to single representation."""
    LAST = "last"  # Use only the last token in sequence
    MEAN = "mean"  # Average all tokens in sequence


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
        temperature_student: float = 1.0,
        temperature_teacher: float = 1.0,
        normalize: bool = True,
        mask_diagonal: bool = True,
        projection_dim: Optional[int] = None,
        use_layer_norm: bool = False,
        apply_softmax: bool = True,
    ):
        """
        Args:
            student_hidden_dim: Dimension of student hidden states
            teacher_hidden_dim: Dimension of teacher hidden states
            temperature_student: Temperature for student softmax (higher = softer, only used if apply_softmax=True)
            temperature_teacher: Temperature for teacher softmax (higher = softer, only used if apply_softmax=True)
            normalize: Whether to L2 normalize before computing similarity
            mask_diagonal: Whether to mask diagonal with -inf (softmax will give 0)
            projection_dim: If specified, project to this dimension before similarity computation
            use_layer_norm: Whether to apply LayerNorm per sample (useful when normalize=False)
            apply_softmax: Whether to apply softmax before KL divergence (default: True)
        """
        super().__init__()
        self.temperature_student = temperature_student
        self.temperature_teacher = temperature_teacher
        self.normalize = normalize
        self.mask_diagonal = mask_diagonal
        self.use_layer_norm = use_layer_norm
        self.apply_softmax = apply_softmax

        # Optional projection layer to match dimensions
        self.projection = None
        if projection_dim is not None:
            self.projection = nn.Linear(teacher_hidden_dim, projection_dim)

        # Optional LayerNorm for per-sample stabilization (alternative to L2 normalization)
        self.student_ln = None
        self.teacher_ln = None
        if use_layer_norm:
            self.student_ln = nn.LayerNorm(student_hidden_dim)
            self.teacher_ln = nn.LayerNorm(teacher_hidden_dim)

    def forward(
        self,
        student_hidden_states: torch.Tensor,  # [batch, hidden_dim]
        teacher_hidden_states: torch.Tensor,  # [batch, hidden_dim]
    ) -> torch.Tensor:
        """
        Args:
            student_hidden_states: [batch, hidden_dim]
            teacher_hidden_states: [batch, hidden_dim]

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

        # Optional LayerNorm for per-sample stabilization (independent of L2 norm)
        if self.use_layer_norm:
            student_hidden_states = self.student_ln(student_hidden_states)
            teacher_hidden_states = self.teacher_ln(teacher_hidden_states)

        # Compute similarity matrices: [batch, batch]
        # sim[i,j] = cosine similarity between sample i and sample j
        student_sim = torch.mm(student_hidden_states, student_hidden_states.t())  # [batch, batch]
        teacher_sim = torch.mm(teacher_hidden_states, teacher_hidden_states.t())  # [batch, batch]

        # Optionally mask out diagonal (self-similarity is always 1.0, not informative)
        # Focus on inter-sample relationships instead
        if self.mask_diagonal:
            mask = torch.eye(batch_size, device=student_sim.device, dtype=torch.bool)
            student_sim = student_sim.masked_fill(mask, float('-inf'))
            teacher_sim = teacher_sim.masked_fill(mask, float('-inf'))

        # Apply temperature scaling (only affects loss if softmax is applied)
        if self.apply_softmax:
            # Scale with separate temperatures for student and teacher
            student_sim_scaled = student_sim / self.temperature_student
            teacher_sim_scaled = teacher_sim / self.temperature_teacher

            # Convert to probability distributions
            student_prob = F.softmax(student_sim_scaled, dim=1)  # [batch, batch]
            teacher_prob = F.softmax(teacher_sim_scaled, dim=1)  # [batch, batch]

            # KL divergence: KL(P || Q) = sum(P * log(P/Q))
            kl_loss = F.kl_div(
                torch.log(student_prob + 1e-8),  # log Q (student)
                teacher_prob,  # P (teacher)
                reduction="batchmean",
            )
        else:
            # Direct MSE loss on similarity matrices (no softmax, no temperature)
            kl_loss = F.mse_loss(student_sim, teacher_sim)

        return kl_loss
