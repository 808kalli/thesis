"""
distill.py

Distillation script for OpenVLAP <-> LAPA latent alignment.
Uses combined loss strategy:
    1. Embedding distillation: Direct alignment of latent action representations  
    2. Contrastive distillation: Preserves similarity structure across batch

Key approach:
    - Teacher (LAPA): 4 latent action token IDs from 8-word vocab
    - Student (OpenVLAP): Uses OpenVLA-7B weights + custom distillation projection (7D -> 4D)  
    - Loss combines embedding alignment + contrastive similarity
    - LoRA fine-tuning for memory efficiency

Run with:
    - [Single GPU]: python distill.py
    - [Override Config Values]: python distill.py --batch_size 32 --learning_rate 1e-4 ...
    - [Resume Training]: python distill.py --resume --resume_from_checkpoint <PATH/TO/CHECKPOINT/DIR>

Usage:
    torchrun --standalone --nnodes=1 --nproc-per-node=1 vla-scripts/distill.py \
        --batch_size 24 \
        --learning_rate 1e-3 \
        --teacher_latent_dir /root/thesis/lapa_latents \
        --num_epochs 10 --save_every_n_epochs 1 \
        --contrastive_loss_type "kl_divergence" \
        --wandb_entity "eliaskallioras-national-technical-university-of-athens"

"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler

import wandb
import torch.nn as nn

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# SEQUENCE ACTION ENCODER WITH CAUSAL ATTENTION
# ============================================================

class SequenceActionEncoder(nn.Module):
    """
    Encodes a sequence of student actions to a 4D latent using causal self-attention.

    Input: [B, seq_len, 7] - sequence of student actions
    Output: [B, 4] - encoded latent
    """
    def __init__(self, action_dim: int = 7, hidden_dim: int = 64, latent_dim: int = 4, num_heads: int = 4):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        # Project each action: 7 → hidden_dim
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Causal self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0
        )

        # Final projection to latent: hidden_dim → latent_dim
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),  # Keep latents in [-1, 1] to match teacher scale
        )

    def forward(self, action_seq):
        """
        Args:
            action_seq: [B, seq_len, 7] - sequence of actions

        Returns:
            latent: [B, 4] - encoded latent
        """
        batch_size, seq_len, _ = action_seq.shape

        # Project actions to hidden dimension
        x = self.action_proj(action_seq)  # [B, seq_len, hidden_dim]

        # Create causal mask (lower triangular, can only attend to current and past)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=action_seq.device) * float('-inf'), diagonal=1)

        # Apply causal self-attention
        attn_out, _ = self.attention(
            x, x, x,
            attn_mask=causal_mask,
            need_weights=False
        )  # [B, seq_len, hidden_dim]

        # Pool: take last token (it has seen the full sequence)
        last_hidden = attn_out[:, -1, :]  # [B, hidden_dim]

        # Project to latent
        latent = self.latent_proj(last_hidden)  # [B, latent_dim]

        return latent


# ============================================================
# AUTOENCODER FOR ACTION SPACE COMPRESSION
# ============================================================

class ActionAutoencoder(nn.Module):
    """
    Autoencoder that compresses sequences of 7D student actions to 4D latent space.

    Encoder: Uses SequenceActionEncoder with causal attention on action sequences
    Decoder: 4D (latent) → 128D (hidden) → 64D (hidden) → 7D (reconstruction with Tanh)
    """
    def __init__(self, action_dim: int = 7, hidden_dim: int = 64, latent_dim: int = 4):
        super().__init__()

        # Encoder: compress sequence of 7D actions → 4D latent using causal attention
        self.encoder = SequenceActionEncoder(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_heads=4
        )

        # Decoder: reconstruct 4D → 7D (deeper and wider)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),  # Expand to 128
            nn.ReLU(),
            nn.Linear(128, 64),  # Further expand
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # Constrain output to [-1, 1] to match input action bounds
        )

    def encode(self, x):
        """
        Encode sequence of actions to 4D latent.

        Args:
            x: [B, seq_len, 7] sequence of student actions

        Returns:
            [B, 4] encoded latent
        """
        return self.encoder(x)

    def decode(self, z):
        """Decode 4D latent to 7D reconstruction."""
        return self.decoder(z)

    def forward(self, x):
        """Full autoencoder pass: encode then decode."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


@dataclass
class DistillConfig:
    # fmt: off

    # Directory Paths
    teacher_latent_dir: Path = Path("/root/thesis/lapa_latents")    # Path to directory with student_inference_results.npy

    # Autoencoder Training Parameters
    batch_size: int = 64                                            # Training batch size
    num_epochs: int = 10                                            # Number of epochs (full passes through dataset)
    save_every_n_epochs: int = 1                                    # Save checkpoint every N epochs
    learning_rate: float = 1e-3                                     # Autoencoder learning rate
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run

    # Autoencoder Architecture Parameters
    ae_action_dim: int = 7                                          # Student action dimension
    ae_hidden_dim: int = 64                                         # Encoder hidden dimension
    ae_latent_dim: int = 4                                          # Latent dimension (matches teacher)

    # Loss Parameters
    distill_loss_type: str = "mse"                                  # "mse" or "cosine" for latent alignment
    distill_loss_weight: float = 0.5                                # Weight for embedding distillation loss
    contrastive_loss_weight: float = 1.0                            # Weight for contrastive loss
    contrastive_loss_type: str = "kl_divergence"                    # "similarity_structure", "kl_divergence", or "contrastive"
    reconstruction_weight: float = 0.5                              # Weight for reconstruction loss

    # Scheduler Parameters
    lr_scheduler_type: str = "cosine"                                # Learning rate scheduler type
    warmup_ratio: float = 0.03                                      # Warmup ratio
    weight_decay: float = 1e-4                                       # Weight decay
    max_grad_norm: float = 1.0                                       # Max gradient norm for clipping

    # Tracking Parameters
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    wandb_project: str = "openvla-distillation"                     # Name of W&B project to log to
    wandb_entity: str = "eliaskallioras"                            # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # Other Parameters
    seed: int = 7                                                   # Random seed

    # fmt: on


# ============================================================
# DISTILLATION LOSS FUNCTIONS
# ============================================================

def compute_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix for a batch of embeddings.

    Args:
        embeddings: [batch, hidden_dim] or [batch, seq_len, hidden_dim]

    Returns:
        similarities: [batch, batch] matrix of cosine similarities
    """
    # If sequence dimension exists, pool it
    if embeddings.ndim == 3:
        embeddings = embeddings.mean(dim=1)  # [batch, hidden_dim]

    # Normalize for cosine similarity
    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Compute pairwise similarities: [batch, hidden_dim] @ [hidden_dim, batch] -> [batch, batch]
    similarities = embeddings_norm @ embeddings_norm.T

    return similarities


def embedding_distill_loss(z_s: torch.Tensor, z_t: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    """
    Direct embedding alignment loss between student and teacher latent actions.

    Args:
        z_s: student latent actions [batch, student_dim] - e.g., [batch, 7] continuous values
        z_t: teacher latent actions [batch, teacher_dim] - e.g., [batch, 4] discrete tokens
    """
    if loss_type == "mse":
        return torch.mean((z_s - z_t) ** 2)
    elif loss_type == "cosine":
        z_s = torch.nn.functional.normalize(z_s, dim=-1)
        z_t = torch.nn.functional.normalize(z_t, dim=-1)
        return 1.0 - torch.mean((z_s * z_t).sum(dim=-1))
    else:
        raise ValueError(f"Unknown distillation loss: {loss_type}")


def similarity_structure_matching_loss(z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """
    Similarity structure matching loss - MSE between student and teacher similarity matrices.

    Forces student to learn the same action similarity relationships as teacher:
    if teacher says action A and B are similar, student should too.

    Computes cosine similarity matrices for both sides and measures MSE between them
    (excluding diagonal since self-similarity is always 1.0).

    Args:
        z_s: student latent actions [batch, student_dim]
        z_t: teacher latent actions [batch, teacher_dim]

    Returns:
        MSE between off-diagonal elements of student and teacher similarity matrices
    """
    # Compute cosine similarity matrices
    student_sim = compute_similarity_matrix(z_s)  # [batch, batch]
    teacher_sim = compute_similarity_matrix(z_t)  # [batch, batch]

    # Mask diagonal (self-similarity is always 1.0, not informative)
    batch_size = z_s.shape[0]
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_s.device)

    # MSE between off-diagonal similarities
    return torch.mean((student_sim[mask] - teacher_sim[mask]) ** 2)


def similarity_kl_divergence_loss(z_s: torch.Tensor, z_t: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    KL divergence loss between student and teacher similarity matrices.

    Computes cosine similarity matrices for both sides, normalizes them to probability
    distributions (via softmax), and measures KL divergence between them.

    This encourages the student to learn the same relative similarity structure as the teacher,
    with stronger penalty for getting the relative rankings wrong.

    Args:
        z_s: student latent actions [batch, student_dim]
        z_t: teacher latent actions [batch, teacher_dim]
        temperature: temperature for softmax scaling of similarity matrices (default 1.0)

    Returns:
        KL divergence between student and teacher similarity distributions
    """
    # Compute cosine similarity matrices
    student_sim = compute_similarity_matrix(z_s)  # [batch, batch]
    teacher_sim = compute_similarity_matrix(z_t)  # [batch, batch]

    # Convert to probability distributions using softmax with temperature
    # Flatten to 1D for softmax (treats each row as independent distribution)
    batch_size = z_s.shape[0]

    # Softmax over each row (for each sample, distribution over other samples)
    student_probs = torch.nn.functional.softmax(student_sim / temperature, dim=1)
    teacher_probs = torch.nn.functional.softmax(teacher_sim / temperature, dim=1)

    # KL divergence: sum over rows
    # KL(P||Q) = sum(P * (log(P) - log(Q)))
    kl_div = torch.nn.functional.kl_div(
        torch.log(student_probs + 1e-8),  # log of student (predicted)
        teacher_probs,                     # target teacher distribution
        reduction='batchmean'
    )

    return kl_div


def contrastive_loss(z_s: torch.Tensor, z_t: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Contrastive loss (InfoNCE) - maximize alignment between student[i] and teacher[i].

    For each student sample, computes:
    - Positive similarity: cos(student[i], teacher[i])
    - Negative similarities: cos(student[i], teacher[j]) for all j != i
    - Loss: -log(exp(pos/T) / sum(exp(all/T)))

    This encourages the positive pair to have high similarity while negative pairs have low similarity.

    Args:
        z_s: student latent actions [batch, student_dim]
        z_t: teacher latent actions [batch, teacher_dim]
        temperature: temperature for scaling similarities (default 0.07)

    Returns:
        Contrastive loss
    """
    # Normalize for cosine similarity
    z_s_norm = torch.nn.functional.normalize(z_s, p=2, dim=1)  # [batch, dim]
    z_t_norm = torch.nn.functional.normalize(z_t, p=2, dim=1)  # [batch, dim]

    batch_size = z_s.shape[0]
    device = z_s.device

    # Cross-model similarity: [batch, batch]
    # cross_sim[i,j] = similarity between student[i] and teacher[j]
    cross_sim = z_s_norm @ z_t_norm.T / temperature  # [batch, batch]

    # Labels: for each student[i], the positive is teacher[i] (diagonal)
    labels = torch.arange(batch_size, device=device)

    # Loss: each student should match its corresponding teacher
    return torch.nn.functional.cross_entropy(cross_sim, labels)


def combined_distill_loss(
    z_s: torch.Tensor,
    z_t: torch.Tensor,
    embedding_weight: float = 1.0,
    contrastive_weight: float = 0.5,
    loss_type: str = "mse",
    contrastive_type: str = "structure"
) -> Tuple[torch.Tensor, dict]:
    """
    Combined loss: embedding alignment + contrastive similarity.

    Args:
        z_s: student latent actions
        z_t: teacher latent actions
        embedding_weight: weight for embedding distillation loss
        contrastive_weight: weight for contrastive loss
        loss_type: type of embedding loss ("mse" or "cosine")
        contrastive_type: type of contrastive loss
            - "similarity_structure": MSE between similarity matrices (student-student vs teacher-teacher)
            - "kl_divergence": KL divergence between similarity matrices (student-student vs teacher-teacher)
            - "contrastive": contrastive loss (student[i] vs teacher[i])

    Returns:
        total_loss: weighted combination
        loss_dict: individual loss components for logging
    """
    # Direct embedding alignment
    embed_loss = embedding_distill_loss(z_s, z_t, loss_type)

    # Contrastive loss (only if batch > 1)
    if z_s.shape[0] > 1:
        if contrastive_type == "similarity_structure":
            contrast_loss = similarity_structure_matching_loss(z_s, z_t)
        elif contrastive_type == "kl_divergence":
            contrast_loss = similarity_kl_divergence_loss(z_s, z_t)
        elif contrastive_type == "contrastive":
            contrast_loss = contrastive_loss(z_s, z_t)
        else:
            raise ValueError(f"Unknown contrastive_type: {contrastive_type}. Use 'similarity_structure', 'kl_divergence', or 'contrastive'")
    else:
        contrast_loss = torch.tensor(0.0, device=z_s.device)

    # Combine
    total_loss = embedding_weight * embed_loss + contrastive_weight * contrast_loss

    loss_dict = {
        "embed_loss": embed_loss.item(),
        "contrast_loss": contrast_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, loss_dict


def autoencoder_distill_loss(
    student_4d: torch.Tensor,
    teacher_4d: torch.Tensor,
    student_7d_sequence: torch.Tensor,
    recon_7d: torch.Tensor,
    embedding_weight: float = 0.5,
    contrastive_weight: float = 1.0,
    recon_weight: float = 0.5,
    loss_type: str = "mse",
    contrastive_type: str = "kl_divergence",
) -> Tuple[torch.Tensor, dict]:
    """
    Combined autoencoder loss: latent alignment + reconstruction.

    Args:
        student_4d: [B, 4] encoded student latent (from sequence encoder)
        teacher_4d: [B, 4] teacher latent actions
        student_7d_sequence: [B, seq_len, 7] original student action sequence
        recon_7d: [B, 7] reconstructed actions from autoencoder decoder
        embedding_weight: weight for embedding distillation loss (latent alignment)
        contrastive_weight: weight for contrastive loss (latent alignment)
        recon_weight: weight for reconstruction loss
        loss_type: type of embedding loss ("mse" or "cosine")
        contrastive_type: type of contrastive loss

    Returns:
        total_loss: weighted combination of latent alignment + reconstruction
        loss_dict: individual loss components for logging
    """
    # Latent alignment loss (MSE + contrastive between student 4D encoded and teacher 4D)
    latent_align_loss, latent_loss_dict = combined_distill_loss(
        z_s=student_4d,  # [B, 4] encoded latent from sequence
        z_t=teacher_4d,  # [B, 4] teacher latent
        embedding_weight=embedding_weight,
        contrastive_weight=contrastive_weight,
        loss_type=loss_type,
        contrastive_type=contrastive_type,
    )

    # Reconstruction loss: MSE between original sequence and reconstructed 7D actions
    # Note: recon_7d is [B, 7] single action, student_7d_sequence is [B, seq_len, 7]
    # We compare against the LAST action in the sequence (or mean of sequence)
    student_sequence_mean = student_7d_sequence.mean(dim=1)  # [B, 7] - average action in sequence
    recon_loss = torch.nn.functional.mse_loss(recon_7d, student_sequence_mean)

    # Combine losses
    total_loss = latent_align_loss + recon_weight * recon_loss

    loss_dict = {
        "latent_align_loss": latent_align_loss.item(),
        "embed_loss": latent_loss_dict["embed_loss"],
        "contrast_loss": latent_loss_dict["contrast_loss"],
        "recon_loss": recon_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, loss_dict


@draccus.wrap()
def distill(cfg: DistillConfig) -> None:
    print("\n" + "="*70)
    print("\033[91m" + " "*15 + "Do or do not; there is no try." + "\033[0m")
    print("="*70 + "\n")

    print(f"Training Autoencoder for Action Space Distillation")
    print(f"Using pre-computed student action sequences from distill_inference.py")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Distillation assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Configure Unique Experiment ID & Log Directory
    exp_id = f"ae-distill-b{cfg.batch_size}-lr{cfg.learning_rate}"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"

    # Start =>> Build Directories
    run_dir = cfg.run_root_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb_id = f"ae-distill+{exp_id}"
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=wandb_id,
            id=None,
            resume=None,
        )

    # Load pre-computed student action sequences and teacher latents from distill_inference.py
    print(f"\nLoading pre-computed student action sequences from {cfg.teacher_latent_dir}")
    results_file = Path(cfg.teacher_latent_dir) / "student_inference_results.npy"

    if not results_file.exists():
        raise FileNotFoundError(
            f"Student inference results not found at {results_file}. "
            f"Please run distill_inference.py first to generate student action sequences."
        )

    results = np.load(results_file, allow_pickle=True).item()
    student_actions_list = results["student_actions"]
    teacher_latents_list = results["teacher_latents"]
    video_ids_list = results["video_ids"]

    print(f"Loaded {len(student_actions_list)} pre-computed samples")
    print(f"  Example student action seq shape: {student_actions_list[0].shape}")
    print(f"  Example teacher latent shape: {teacher_latents_list[0].shape}")

    # Create simple dataset from pre-computed data
    class PrecomputedDataset(torch.utils.data.Dataset):
        def __init__(self, student_actions, teacher_latents):
            # student_actions: list of [seq_len, 7] numpy arrays
            # teacher_latents: list of [4] numpy arrays
            self.student_actions = [torch.tensor(a, dtype=torch.float32) for a in student_actions]
            self.teacher_latents = [torch.tensor(t, dtype=torch.float32) for t in teacher_latents]

        def __len__(self):
            return len(self.student_actions)

        def __getitem__(self, idx):
            student_seq = self.student_actions[idx]  # [seq_len, 7]
            teacher_latent = self.teacher_latents[idx]  # [4]

            # Ensure shapes are correct
            assert student_seq.ndim == 2 and student_seq.shape[1] == 7, f"Expected [seq_len, 7], got {student_seq.shape}"
            assert teacher_latent.ndim == 1 and teacher_latent.shape[0] == 4, f"Expected [4], got {teacher_latent.shape}"

            return {
                "action_sequence": student_seq,  # [seq_len, 7]
                "teacher_latent": teacher_latent,  # [4]
            }

    dataset = PrecomputedDataset(student_actions_list, teacher_latents_list)

    # Calculate total steps based on epochs and dataset size
    steps_per_epoch = len(dataset) // cfg.batch_size
    total_steps = cfg.num_epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_ratio * total_steps)

    def ae_collator(batch):
        """Collator that handles variable-length action sequences for autoencoder."""
        # Handle action sequences (pad to same length)
        action_sequences = [item["action_sequence"] for item in batch]
        max_seq_len = max(seq.shape[0] for seq in action_sequences)

        padded_actions = []
        for seq in action_sequences:
            if seq.shape[0] < max_seq_len:
                # Pad with zeros to max_seq_len
                pad_amount = max_seq_len - seq.shape[0]
                padded_seq = torch.cat([seq, torch.zeros(pad_amount, 7)])
                padded_actions.append(padded_seq)
            else:
                padded_actions.append(seq)

        return {
            "action_sequence": torch.stack(padded_actions),  # [B, max_seq_len, 7]
            "teacher_latent": torch.stack([item["teacher_latent"] for item in batch]),  # [B, 4]
        }

    # Use the custom collator
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=ae_collator,
        num_workers=0,
    )

    print(f"Loaded {len(dataset)} pre-computed samples for autoencoder training with batch_size={cfg.batch_size}")

    # Deque to store recent train metrics
    recent_losses = deque(maxlen=1)
    recent_embed_losses = deque(maxlen=1)
    recent_contrast_losses = deque(maxlen=1)

    # Initialize autoencoder for action space compression
    print(f"\nInitializing autoencoder...")
    autoencoder = ActionAutoencoder(action_dim=7, hidden_dim=64, latent_dim=4).to(device_id)
    print(f"Autoencoder initialized and moved to device {device_id}")

    # Create optimizer for autoencoder only
    ae_optimizer = AdamW(autoencoder.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    ae_scheduler = get_scheduler(
        cfg.lr_scheduler_type,
        optimizer=ae_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Start training!
    print(f"\nStarting autoencoder training for {cfg.num_epochs} epochs...")
    autoencoder.train()
    ae_optimizer.zero_grad()
    global_step = 0

    with tqdm.tqdm(total=cfg.num_epochs, desc="Overall Progress", leave=True, position=1) as epoch_progress:
        for epoch in range(cfg.num_epochs):
            with tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{cfg.num_epochs}", leave=False, position=0) as batch_progress:
                for batch_idx, batch in enumerate(dataloader):

                    with torch.autocast("cuda", dtype=torch.bfloat16):

                        # Get action sequence from batch [B, seq_len, 7]
                        student_action_seq = batch["action_sequence"].to(device_id)

                        # Autoencoder: encode sequence to 4D latent and reconstruct to 7D
                        student_4d_encoded = autoencoder.encode(student_action_seq)  # [B, 4]
                        student_7d_recon = autoencoder.decode(student_4d_encoded)  # [B, 7]

                        # Get teacher latents
                        teacher_hidden = batch["teacher_latent"].to(device_id)  # [B, 4]
                        teacher_hidden = (teacher_hidden / 7.0) * 2.0 - 1.0  # normalize to [-1, 1]

                        # ========================================
                        # COMPUTE COMBINED LOSS
                        # ========================================

                        loss, loss_dict = autoencoder_distill_loss(
                            student_4d=student_4d_encoded,  # [B, 4] - encoded latent from sequence
                            teacher_4d=teacher_hidden,  # [B, 4] - teacher latent tokens
                            student_7d_sequence=student_action_seq,  # [B, seq_len, 7] - action sequence
                            recon_7d=student_7d_recon,  # [B, 7] - reconstructed 7D action
                            embedding_weight=cfg.distill_loss_weight,
                            contrastive_weight=cfg.contrastive_loss_weight,
                            recon_weight=0.5,  # Weight for reconstruction loss
                            loss_type=cfg.distill_loss_type,
                            contrastive_type=cfg.contrastive_loss_type
                        )

                    # Backward pass
                    loss.backward()

                    # Optimizer Step (autoencoder only)
                    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), cfg.max_grad_norm)

                    ae_optimizer.step()
                    ae_scheduler.step()
                    ae_optimizer.zero_grad()
                    batch_progress.update()

                    # Store train metrics
                    recent_losses.append(loss.item())
                    recent_embed_losses.append(loss_dict["embed_loss"])
                    recent_contrast_losses.append(loss_dict["contrast_loss"])

                    global_step += 1

                    # Compute train metrics
                    smoothened_loss = recent_losses[0]
                    smoothened_embed_loss = recent_embed_losses[0]
                    smoothened_contrast_loss = recent_contrast_losses[0]

                    # Push Metrics to W&B (every 10 gradient steps)
                    if distributed_state.is_main_process and global_step % 10 == 0:
                        wandb.log(
                            {
                                "total_loss": smoothened_loss,
                                "embed_loss": smoothened_embed_loss,
                                "contrast_loss": smoothened_contrast_loss,
                                "learning_rate": ae_scheduler.get_last_lr()[0],
                            },
                            step=global_step,
                        )

            # Save Autoencoder Checkpoint after each epoch
            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                if distributed_state.is_main_process:
                    if cfg.save_latest_checkpoint_only:
                        checkpoint_path = run_dir / "autoencoder_latest.pt"
                    else:
                        checkpoint_path = run_dir / f"autoencoder_epoch{epoch+1}.pt"

                    torch.save({
                        "epoch": epoch,
                        "global_step": global_step,
                        "autoencoder_state_dict": autoencoder.state_dict(),
                        "optimizer_state_dict": ae_optimizer.state_dict(),
                        "scheduler_state_dict": ae_scheduler.state_dict(),
                    }, checkpoint_path)
                    print(f"  Saved autoencoder checkpoint to {checkpoint_path}")

            epoch_progress.update()

    print("\nAutoencoder Training Complete ✅")
    if distributed_state.is_main_process:
        wandb.finish()
        print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    distill()