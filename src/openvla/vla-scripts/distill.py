"""
distill.py

Distillation script for OpenVLAP <-> LAPA latent alignment.
Uses combined loss strategy:
    1. Embedding distillation: Direct alignment of latent action representations
    2. Contrastive distillation: Preserves similarity structure across batch

Key approach:
    - Teacher (LAPA): 4 latent action token IDs from 8-word vocab
    - Student (OpenVLA): 7 continuous action values (bin centers from 256-word vocab)
    - MLP on student side: Projects 7D student actions to 4D teacher latent space
    - Loss combines embedding alignment + contrastive similarity

Note: We distill latent action representations, not hidden states.
      This makes more sense given the different action spaces.

Run:
    torchrun --standalone --nproc-per-node=8 distill.py
"""

import os
import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.models import load, load_openvlap
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics
from prismatic.util import set_global_seed
from experiments.robot.openvla_utils import get_vlap_action


# ============================================================
# INITIAL SETUP
# ============================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
overwatch = initialize_overwatch(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class DistillConfig:
    """Configuration for OpenVLAP distillation."""

    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(
            VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id
        )
    )

    teacher_latent_dir: Path = Path("/home/elias/Thesis/teacher_latents")
    pretrained_checkpoint: Optional[Path] = None
    run_root_dir: Path = Path("runs")
    run_id: Optional[str] = None

    seed: int = 7
    save_interval: int = 2000
    distill_loss_type: str = "mse"
    distill_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.5
    contrastive_loss_type: str = "similarity_structure"    # "similarity_structure": MSE of similarity matrices (student-student vs teacher-teacher)
                                                          # "kl_divergence": KL divergence of similarity matrices (student-student vs teacher-teacher)
                                                          # "contrastive": contrastive loss (student[i] vs teacher[i])
    align_dim: int = 4
    align_hidden_dim: int = 64     #hidden projection dimentioni to grasp complex relationships between the 2 latent action spaces (7 → 64 → 4)

    # --- new additions ---
    grad_accumulation_steps: int = 4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03

    trackers: Tuple[str, ...] = ("jsonl", "wandb")
    wandb_project: str = "openvla-distillation"
    wandb_entity: str = "eliaskallioras"

    hf_token: Union[str, Path] = Path(".hf_token")

    def __post_init__(self):
        """Expose optimizer params and training constants."""
        self.epochs = self.vla.epochs
        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.enable_mixed_precision_training = self.vla.enable_mixed_precision_training


# ============================================================
# DATASET
# ============================================================

class VLADistillDataset(Dataset):
    """
    Loads .npy files containing individual frame data (rgb, prompt, teacher_latent, video_id, frame_number).

    Each .npy file corresponds to a single frame from a video.
    Filename format: {video_id}_{frame_number}.npy
    where frame_number is the actual video frame (0, 30, 60, ...)

    teacher_latent should be the LAPA latent action tokens:
        - Shape: [4] containing token IDs from 0-7 (8-word vocabulary)
        - These are the discrete latent action codes produced by LAPA

    video_id and frame_number are included to allow stitching frames back together if needed.
    """

    def __init__(self, npy_dir: Path):
        self.paths = sorted(Path(npy_dir).glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No .npy files found in {npy_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx], allow_pickle=True).item()
        rgb = torch.tensor(data["rgb"]).permute(2, 0, 1).float() / 255.0
        prompt = data["prompt"]

        # Teacher latent: LAPA's 4 token IDs (discrete)
        teacher_latent = torch.tensor(data["teacher_latent"]).float()

        # Video ID and frame number for potential stitching back together
        video_id = data.get("video_id", self.paths[idx].stem.rsplit("_", 1)[0])
        frame_number = data.get("frame_number", int(self.paths[idx].stem.rsplit("_", 1)[1]))

        return {
            "image": rgb,
            "prompt": prompt,
            "teacher_latent": teacher_latent,
            "video_id": video_id,
            "frame_number": frame_number,
        }


# ============================================================
# DISTILLATION LOSS
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


# ============================================================
# MAIN DISTILLATION FUNCTION
# ============================================================

@draccus.wrap()
def distill(cfg: DistillConfig) -> None:
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()
    set_global_seed(cfg.seed)

    cfg.run_id = cfg.run_id or f"openvla-distill-x{cfg.seed}"
    run_dir = cfg.run_root_dir / cfg.run_id
    os.makedirs(run_dir / "checkpoints", exist_ok=True)

    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            json.dump(yaml.safe_load(f_yaml), f_json, indent=2)

    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]

    # ------------------------------------------------------------
    # Load student model with distillation projection
    # ------------------------------------------------------------
    # Load with distillation projection: 7D student actions -> 4D teacher latent
    vlm = (
        load_openvlap(
            cfg.pretrained_checkpoint,
            hf_token=hf_token,
            load_for_training=True,
            distill_projection_dim=cfg.align_dim,  # LAPA's latent action dim (4)
            distill_projection_hidden_dim=cfg.align_hidden_dim,
        )
        if cfg.pretrained_checkpoint
        else load(cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True)
    )
    vlm.train()
    overwatch.info("Loaded OpenVLAP student with distillation projection (7 -> 4) ✅")

    # ------------------------------------------------------------
    # Dataset & Dataloader
    # ------------------------------------------------------------
    dataset = VLADistillDataset(cfg.teacher_latent_dir)
    # Batch size > 1 required for contrastive loss to work
    # Adjust based on GPU memory (16 is reasonable for 40GB GPU)
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    overwatch.info(
        f"Loaded {len(dataset)} distillation samples with batch_size={batch_size}\n"
        f"Expected teacher latent shape: [4] (LAPA's 4 latent action tokens)\n"
        f"Expected student action shape: [7] (OpenVLA's 7 continuous actions)"
    )

    # ------------------------------------------------------------
    # Optimizer / Scheduler / Metrics
    # ------------------------------------------------------------
    # Optimizer now includes distill_projection parameters (already part of vlm.parameters())
    trainable_params = list(vlm.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    total_steps = (cfg.epochs * len(dataloader)) // max(1, cfg.grad_accumulation_steps)
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_scheduler(
        cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
    )

    processor = vlm.vision_backbone.image_transform
    base_vla_name = cfg.vla.base_vlm
    global_step = 0

    # ------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------
    for epoch in range(cfg.epochs):
        with tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not overwatch.is_rank_zero()) as progress:
            optimizer.zero_grad()
            for step, batch in enumerate(dataloader):
                # Prepare batch data
                images = batch["image"].to(device_id)  # [batch, C, H, W]
                prompts = batch["prompt"]  # List of strings
                teacher_hidden = batch["teacher_latent"].to(device_id)  # [batch, 4]
                teacher_hidden = (teacher_hidden / 7.0) * 2.0 - 1.0  # normalize to [-1, 1]

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.enable_mixed_precision_training):
                    # ========================================
                    # STUDENT FORWARD PASS - Get Projected Latent Actions
                    # ========================================
                    # get_vla_action -> predict_action already applies distill_projection
                    # Returns [4] projected actions (7D -> 4D via integrated MLP)

                    student_latent_list = []
                    for i in range(len(prompts)):
                        rgb = images[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        obs = {"full_image": rgb}

                        # get_vla_action calls predict_action which now returns projected 4D tensor
                        projected_action = get_vlap_action(
                            vlm,
                            processor,
                            base_vla_name,
                            obs,
                            prompts[i],
                            center_crop=False,
                        )
                        student_latent_list.append(projected_action if isinstance(projected_action, torch.Tensor) else torch.tensor(projected_action).to(device_id).float())

                    student_latent_projected = torch.stack(student_latent_list)  # [batch, 4]

                    # ========================================
                    # APPLY NON-LEARNABLE LAYER NORMALIZATION TO TEACHER LATENT
                    # ========================================
                    teacher_hidden = torch.nn.functional.layer_norm(
                        teacher_hidden,
                        normalized_shape=(teacher_hidden.shape[-1],),
                        weight=None,  # No learnable affine parameters
                        bias=None,    # No learnable affine parameters
                        eps=1e-6
                    )

                    # ========================================
                    # COMPUTE COMBINED LOSS
                    # ========================================
                    loss, loss_dict = combined_distill_loss(
                        z_s=student_latent_projected,  # [batch, 4] - projected student actions
                        z_t=teacher_hidden,             # [batch, 4] - teacher latent tokens
                        embedding_weight=cfg.distill_loss_weight,
                        contrastive_weight=cfg.contrastive_loss_weight,
                        loss_type=cfg.distill_loss_type,
                        contrastive_type=cfg.contrastive_loss_type
                    )

                # --- gradient accumulation ---
                loss = loss / max(1, cfg.grad_accumulation_steps)
                loss.backward()

                did_step = False
                if ((step + 1) % max(1, cfg.grad_accumulation_steps)) == 0:
                    # Clip gradients (includes distill_projection params since they're in vlm.parameters())
                    torch.nn.utils.clip_grad_norm_(vlm.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    did_step = True

                    # Save checkpoint every save_interval steps
                    if overwatch.is_rank_zero() and (global_step % cfg.save_interval == 0):
                        ckpt_path = run_dir / "checkpoints" / f"step_{global_step}.pt"
                        torch.save(
                            {
                                "model": vlm.state_dict(),  # Includes distill_projection
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "epoch": epoch,
                                "step": global_step,
                            },
                            ckpt_path,
                        )
                        overwatch.info(f"Saved checkpoint at {ckpt_path}")

                # --- metrics ---
                if did_step:
                    metrics.commit(
                        loss=loss_dict["total_loss"],
                        embed_loss=loss_dict["embed_loss"],
                        contrast_loss=loss_dict["contrast_loss"],
                        lr=scheduler.get_last_lr()[0]
                    )
                else:
                    metrics.commit(
                        loss=loss_dict["total_loss"],
                        embed_loss=loss_dict["embed_loss"],
                        contrast_loss=loss_dict["contrast_loss"]
                    )

                if overwatch.is_rank_zero():
                    postfix = {
                        "total": f"{loss_dict['total_loss']:.4f}",
                        "embed": f"{loss_dict['embed_loss']:.4f}",
                        "contr": f"{loss_dict['contrast_loss']:.4f}"
                    }
                    if did_step:
                        postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
                    progress.set_postfix(postfix)

        # Save checkpoint per epoch
        if overwatch.is_rank_zero():
            ckpt_path = run_dir / "checkpoints" / f"epoch_{epoch+1}.pt"
            torch.save(
                {
                    "model": vlm.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )
            overwatch.info(f"Saved checkpoint at {ckpt_path}")

    # ------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------
    metrics.finalize()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    overwatch.info("Distillation Complete ✅")


if __name__ == "__main__":
    distill()
