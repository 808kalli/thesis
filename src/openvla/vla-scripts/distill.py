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
"""

import os
import json
import yaml
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
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor, get_scheduler
from PIL import Image

import wandb
import torch.nn as nn
from prismatic.extern.hf.configuration_prismatic import OpenVLAPConfig, OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAPForActionPrediction, OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from transformers import DataCollatorWithPadding

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def add_distillation_layers(vla_model, action_dim: int = 7, hidden_dim: int = 64, projection_dim: int = 4):
    """
    Adds distillation projection and normalization layers to a standard OpenVLA model.
    These layers are randomly initialized and trainable.
    Args:
        vla_model: Loaded OpenVLAForActionPrediction model
        action_dim: Dimension of action space (default: 7)
        hidden_dim: Hidden dimension for projection MLP (default: 64)
        projection_dim: Final projection dimension matching teacher latent dim (default: 4)
    Returns:
        Modified model with distill_projection and distill_norm layers added
    """
    vla_model.action_dim = action_dim
    
    # Add distillation projection layers (randomly initialized)
    vla_model.distill_projection = nn.Sequential(
        nn.Linear(action_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, projection_dim),
        nn.Tanh(),
    )
    
    # Add distillation normalization layer (randomly initialized)
    vla_model.distill_norm = nn.LayerNorm(
        projection_dim,
        elementwise_affine=True,  # learnable γ, β
        eps=1e-6
    )
    
    def get_projected_actions_from_batch(self, input_ids, attention_mask, pixel_values):
        """Get projected actions from tokenized batch"""
        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=None,
        )
        
        action_logits = output.logits[:, self.vision_backbone.featurizer.patch_embed.num_patches : -1]
        action_preds = action_logits.argmax(dim=2)
        
        normalized_actions_batch = []
        for i in range(action_preds.shape[0]):
            action_token_ids = action_preds[i].cpu().numpy()
            discretized_actions = self.vocab_size - action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
            normalized_actions = self.bin_centers[discretized_actions]
            normalized_actions_batch.append(normalized_actions)
        
        normalized_actions_tensor = torch.from_numpy(np.stack(normalized_actions_batch)).to(self.device).float()
        projected_actions = self.distill_projection(normalized_actions_tensor)
        projected_actions = self.distill_norm(projected_actions)
        
        return projected_actions
    
    # Bind the method to the model instance
    vla_model.get_projected_actions_from_batch = get_projected_actions_from_batch.__get__(vla_model)
    
    return vla_model


@dataclass
class DistillConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    teacher_latent_dir: Path = Path("/root/thesis/lapa_latents")    # Path to LAPA latent directory
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Distillation Parameters
    batch_size: int = 64                                            # Distillation batch size
    max_steps: int = 50_000                                         # Max number of distillation steps
    save_steps: int = 2000                                          # Interval for checkpoint saving
    learning_rate: float = 1e-4                                     # Distillation learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run

    # Distillation Loss Parameters
    distill_loss_type: str = "mse"                                  # "mse" or "cosine"
    distill_loss_weight: float = 0.5                                # Weight for embedding distillation loss
    contrastive_loss_weight: float = 1.0                            # Weight for contrastive loss
    contrastive_loss_type: str = "kl_divergence"                    # "similarity_structure", "kl_divergence", or "contrastive"
    align_dim: int = 4                                               # LAPA's latent action dimension
    align_hidden_dim: int = 64                                       # Hidden projection dimension (7 → 64 → 4)

    # LoRA Arguments
    use_lora: bool = True                                            # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                        # Dropout applied to LoRA weights
    use_quantization: bool = False                                   # Whether to 4-bit quantize VLA for LoRA fine-tuning

    # Scheduler Parameters
    lr_scheduler_type: str = "cosine"                                # Learning rate scheduler type
    warmup_ratio: float = 0.03                                      # Warmup ratio
    weight_decay: float = 1e-4                                       # Weight decay
    max_grad_norm: float = 1.0                                       # Max gradient norm for clipping

    # Checkpoint Resumption
    resume: bool = False                                             # Whether to resume from checkpoint
    resume_from_checkpoint: Optional[Path] = None                    # Specific checkpoint path to resume from

    # Tracking Parameters
    wandb_project: str = "openvla-distillation"                     # Name of W&B project to log to
    wandb_entity: str = "eliaskallioras"                            # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # Other Parameters
    seed: int = 7                                                   # Random seed
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # fmt: on


class VLADistillDataset(Dataset):
    """
    Loads .npy files and tokenizes prompts + processes images for distillation training.
    Returns input_ids, pixel_values, and teacher_latent ready for VLA model.
    """
    
    def __init__(self, npy_dir: Path, processor):
        self.paths = sorted(Path(npy_dir).glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No .npy files found in {npy_dir}")
        
        self.processor = processor
        
        # Import prompt builder (same as finetune.py)
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        self.prompt_builder_fn = PurePromptBuilder

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx], allow_pickle=True).item()
        
        # Load raw data
        rgb = data["rgb"]  # Keep as numpy array for PIL conversion
        prompt = data["prompt"]
        teacher_latent = torch.tensor(data["teacher_latent"]).float()
        
        # Convert image to PIL (like RLDSBatchTransform does)
        rgb_pil = Image.fromarray(rgb.astype(np.uint8))
        
        # Build prompt (same format as RLDSBatchTransform, but no action tokens)
        prompt_builder = self.prompt_builder_fn("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {prompt.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        
        # Tokenize prompt (like RLDSBatchTransform does)
        input_ids = self.processor.tokenizer(prompt_text, add_special_tokens=True).input_ids
        input_ids = torch.tensor(input_ids)
        
        # Process image (like RLDSBatchTransform does)
        pixel_values = self.processor.image_processor.apply_transform(rgb_pil)
        
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values, 
            "teacher_latent": teacher_latent,
            "video_id": data.get("video_id", self.paths[idx].stem.rsplit("_", 1)[0]),
            "frame_number": data.get("frame_number", int(self.paths[idx].stem.rsplit("_", 1)[1])),
        }


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


def save_checkpoint(
    vla,
    optimizer,
    scheduler,
    gradient_step_idx,
    batch_idx,
    checkpoint_dir,
    processor,
    cfg,
    distributed_state,
    adapter_dir=None,
):
    """Save complete training state for resumption."""
    if distributed_state.is_main_process:
        print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

        # Save processor
        processor.save_pretrained(checkpoint_dir)

        # If LoRA, save adapter weights to temporary directory
        save_dir = adapter_dir if cfg.use_lora else checkpoint_dir
        vla.module.save_pretrained(save_dir)

        # Save training state
        training_state = {
            'gradient_step_idx': gradient_step_idx,
            'batch_idx': batch_idx,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': [state.cpu() for state in torch.cuda.get_rng_state_all()],  # Move to CPU
        }
        torch.save(training_state, checkpoint_dir / "training_state.pt")

    # Wait for main process to finish saving
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dist.barrier()

    # Merge LoRA weights into model backbone if using LoRA
    if cfg.use_lora:
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()
        
        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

    # Block on main process checkpointing
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        dist.barrier()


def load_checkpoint(checkpoint_dir, optimizer, scheduler, device_id, distributed_state):
    """Load complete training state for resumption."""
    training_state_path = checkpoint_dir / "training_state.pt"
    
    if not training_state_path.exists():
        if distributed_state.is_main_process:
            print(f"No training state found at {training_state_path}")
        return None
    
    if distributed_state.is_main_process:
        print(f"Loading training state from {training_state_path}")
    
    training_state = torch.load(training_state_path, map_location='cpu')  # Load to CPU first
    
    # Restore optimizer state
    optimizer.load_state_dict(training_state['optimizer_state_dict'])
    
    # Restore scheduler state
    scheduler.load_state_dict(training_state['scheduler_state_dict'])
    
    # Restore RNG states
    torch.set_rng_state(training_state['rng_state'])
    
    # Restore CUDA RNG states - ensure they're ByteTensors
    cuda_rng_states = training_state['cuda_rng_state']
    if isinstance(cuda_rng_states, list):
        # Convert to ByteTensor if needed and move to appropriate device
        cuda_rng_states = [state.to(torch.uint8) if state.dtype != torch.uint8 else state 
                          for state in cuda_rng_states]
        torch.cuda.set_rng_state_all(cuda_rng_states)
    
    if distributed_state.is_main_process:
        print(f"Resumed from gradient step {training_state['gradient_step_idx']}, batch {training_state['batch_idx']}")
    
    return training_state


@draccus.wrap()
def distill(cfg: DistillConfig) -> None:
    print("\n" + "="*70)
    print("\033[91m" + " "*15 + "Do or do not; there is no try." + "\033[0m")
    print("="*70 + "\n")

    print(f"Distilling OpenVLAP Model using LAPA teacher latents")
    print(f"Base model: {cfg.vla_path}")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Distillation assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Configure Unique Experiment ID & Log Directory
    exp_id = f"openvlap-distill-b{cfg.batch_size}-lr{cfg.learning_rate}"
    if cfg.use_lora:
        exp_id += f"-lora-r{cfg.lora_rank}"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb_id = f"distill+{exp_id}"
        wandb.init(
            entity=cfg.wandb_entity, 
            project=cfg.wandb_project, 
            name=wandb_id,
            id=None,
            resume=None,
        )

    # Determine checkpoint directory for resumption
    checkpoint_dir = cfg.resume_from_checkpoint if cfg.resume_from_checkpoint is not None else run_dir

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvlap", OpenVLAPConfig)
    AutoImageProcessor.register(OpenVLAPConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAPConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAPConfig, OpenVLAPForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    # If resuming, load from checkpoint; otherwise load from original model
    model_path = checkpoint_dir if cfg.resume else cfg.vla_path
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Add distillation layers with random initialization
    vla = add_distillation_layers(
        vla,
        action_dim=7,
        hidden_dim=cfg.align_hidden_dim,
        projection_dim=cfg.align_dim,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # Move new layers to device (if not using quantization)
    if not cfg.use_quantization:
        vla.distill_projection = vla.distill_projection.to(device_id)
        vla.distill_norm = vla.distill_norm.to(device_id)

    # 🎯 FREEZING CONTROL POINT - finetune.py style with LoRA
    print(f"   use_lora: {cfg.use_lora}")

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Only LLM layers
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    # Create Learning Rate Scheduler
    total_steps = cfg.max_steps
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_scheduler(
        cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Load Distillation Dataset
    dataset = VLADistillDataset(cfg.teacher_latent_dir, processor)

    def distill_collator(batch):
        # Pad input_ids
        input_ids = [item["input_ids"] for item in batch]
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        attention_masks = []
        
        for ids in input_ids:
            pad_len = max_len - len(ids)
            padded_ids = torch.cat([ids, torch.full((pad_len,), processor.tokenizer.pad_token_id)])
            attention_mask = torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)])
            
            padded_input_ids.append(padded_ids)
            attention_masks.append(attention_mask)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks),
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "teacher_latent": torch.stack([item["teacher_latent"] for item in batch]),
        }

    # Use the custom collator
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=distill_collator,
        num_workers=0,
    )

    print(f"Loaded {len(dataset)} distillation samples with batch_size={cfg.batch_size}")

    # Deque to store recent train metrics
    recent_losses = deque(maxlen=1)
    recent_embed_losses = deque(maxlen=1)
    recent_contrast_losses = deque(maxlen=1)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                
                student_latent_projected = vla.module.get_projected_actions_from_batch(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(device_id),
                )

                teacher_hidden = batch["teacher_latent"].to(device_id)  # [batch, 4]
                teacher_hidden = (teacher_hidden / 7.0) * 2.0 - 1.0  # normalize to [-1, 1]
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

            # Backward pass
            loss.backward()

            # Optimizer Step
            torch.nn.utils.clip_grad_norm_(vla.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.update()

            # Store train metrics
            recent_losses.append(loss.item())
            recent_embed_losses.append(loss_dict["embed_loss"])
            recent_contrast_losses.append(loss_dict["contrast_loss"])

            # Gradient step index (1:1 with batch index now)
            gradient_step_idx = batch_idx + 1

            # Compute train metrics
            smoothened_loss = recent_losses[0]
            smoothened_embed_loss = recent_embed_losses[0]
            smoothened_contrast_loss = recent_contrast_losses[0]

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "total_loss": smoothened_loss,
                        "embed_loss": smoothened_embed_loss,
                        "contrast_loss": smoothened_contrast_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                    },
                    step=gradient_step_idx,
                )

            # Save Model Checkpoint
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if cfg.save_latest_checkpoint_only:
                    # Overwrite latest checkpoint
                    save_checkpoint(
                        vla, optimizer, scheduler, gradient_step_idx, batch_idx,
                        run_dir, processor, cfg,
                        distributed_state, adapter_dir if cfg.use_lora else None
                    )
                else:
                    # Save checkpoint in new directory
                    checkpoint_dir_step = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                    os.makedirs(checkpoint_dir_step, exist_ok=True)
                    
                    adapter_dir_step = Path(str(adapter_dir) + f"--{gradient_step_idx}_chkpt") if cfg.use_lora else None
                    if adapter_dir_step:
                        os.makedirs(adapter_dir_step, exist_ok=True)
                    
                    save_checkpoint(
                        vla, optimizer, scheduler, gradient_step_idx, batch_idx,
                        checkpoint_dir_step, processor, cfg,
                        distributed_state, adapter_dir_step
                    )

            # Stop training when max_steps is reached
            if gradient_step_idx >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

    print("Distillation Complete ✅")
    if distributed_state.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    distill()