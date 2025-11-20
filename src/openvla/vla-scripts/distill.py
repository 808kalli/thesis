"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
    - [Resume Training]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                            --resume \
                            --resume_from_checkpoint <PATH/TO/CHECKPOINT/DIR>
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from distillation_utils import (
    AggregationMethod,
    StudentSequenceProjectionMLP,
    SimilarityMatrixDistillationLoss,
)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Distillation Parameters
    use_distillation: bool = True                                  # Whether to use knowledge distillation from teacher
    teacher_dataset_h5_path: Optional[Path] = None                  # Path to precomputed teacher_dataset_interpolated.h5 file
    distill_weight: float = 0.1                                     # Weight of distillation loss in total loss
    aggregation_method: str = "mean"                                # How to aggregate sequence: "last" or "mean"
    frame_alignment_mode: str = "interpolated"                   # Frame alignment: "supervised_only" or "interpolated"
    distill_temperature: float = 1.0                                # Temperature for similarity matrix softmax
    distill_normalize: bool = True                                  # Whether to L2 normalize before similarity computation
    distill_projection_dim: Optional[int] = None                    # Optional projection dimension for hidden states

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "eliaskallioras"                            # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on


def save_checkpoint(
    vla,
    optimizer,
    gradient_step_idx,
    batch_idx,
    checkpoint_dir,
    processor,
    vla_dataset,
    cfg,
    distributed_state,
    adapter_dir=None,
):
    """Save complete training state for resumption."""
    if distributed_state.is_main_process:
        print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

        # Save processor
        processor.save_pretrained(checkpoint_dir)

        # Save dataset statistics
        save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

        # If LoRA, save adapter weights to temporary directory
        save_dir = adapter_dir if cfg.use_lora else checkpoint_dir
        vla.module.save_pretrained(save_dir)

        # Save training state
        training_state = {
            'gradient_step_idx': gradient_step_idx,
            'batch_idx': batch_idx,
            'optimizer_state_dict': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': [state.cpu() for state in torch.cuda.get_rng_state_all()],  # Move to CPU
        }
        torch.save(training_state, checkpoint_dir / "training_state.pt")

    # Wait for main process to finish saving
    dist.barrier()

    # Merge LoRA weights into model backbone if using LoRA
    if cfg.use_lora:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()
        
        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

    # Block on main process checkpointing
    dist.barrier()


def load_checkpoint(checkpoint_dir, optimizer, device_id, distributed_state):
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
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    # exp_id = (
    #     f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
    #     f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
    #     f"+lr-{cfg.learning_rate}"
    # )
    # if cfg.use_lora:
    #     exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    # if cfg.use_quantization:
    #     exp_id += "+q-4bit"
    # if cfg.run_id_note is not None:
    #     exp_id += f"--{cfg.run_id_note}"
    # if cfg.image_aug:
    #     exp_id += "--image_aug"

    exp_id = "spatial-to-object-finetuning"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        # Resume W&B run if resuming training
        wandb_id = f"ft+{exp_id}"
        wandb.init(
            entity=cfg.wandb_entity, 
            project=cfg.wandb_project, 
            name=wandb_id,
            id=None,
            resume=None,
        )

    # Determine checkpoint directory for resumption
    checkpoint_dir = run_dir

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    # If resuming, load from checkpoint; otherwise load from original model
    model_path = cfg.vla_path
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Initialize Distillation Components
    if cfg.teacher_dataset_h5_path is None:
        raise ValueError("use_distillation=True but teacher_dataset_h5_path is not specified!")

    if distributed_state.is_main_process:
        print(f"Initializing knowledge distillation...")
        print(f"  - Teacher Dataset H5: {cfg.teacher_dataset_h5_path}")
        print(f"  - Distill Weight: {cfg.distill_weight}")

    # Load precomputed teacher dataset (already aggregated and aligned)
    import h5py
    teacher_dataset_file = h5py.File(str(cfg.teacher_dataset_h5_path), "r")

    # Note: No aggregation needed for teacher - already in precomputed dataset
    aggregation_enum = AggregationMethod(cfg.aggregation_method)

    # Student: aggregation + bottleneck MLP (4096 → 2048 → 4096)
    # Teacher states are ALREADY aggregated and aligned in precomputed dataset
    sequence_aggregation_student = StudentSequenceProjectionMLP(
        input_dim=4096,
        bottleneck_dim=2048,
        aggregation_method=aggregation_enum,
    ).to(device_id)

    # Initialize distillation loss
    distillation_loss_fn = SimilarityMatrixDistillationLoss(
        student_hidden_dim=4096,
        teacher_hidden_dim=4096,
        temperature=cfg.distill_temperature,
        normalize=cfg.distill_normalize,
        projection_dim=cfg.distill_projection_dim,
    ).to(device_id)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    # IMPORTANT: Must be after distillation module initialization so their parameters are included
    trainable_params = [param for param in vla.parameters() if param.requires_grad]

    # Add distillation module parameters to optimizer
    trainable_params.extend(sequence_aggregation_student.parameters())
    trainable_params.extend(distillation_loss_fn.parameters())

    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Initialize training state tracking
    start_gradient_step = 0
    start_batch_idx = 0

    # Load Fine-tuning Dataset
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_distill_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Create directory for hidden state logging (first 100 batches)
    hidden_state_dir = Path(run_dir) / "hidden_states_logs"
    hidden_state_dir.mkdir(parents=True, exist_ok=True)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, initial=start_gradient_step, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Skip batches if resuming from checkpoint
            if batch_idx < start_batch_idx:
                continue

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                    output_hidden_states=cfg.use_distillation,  # Request hidden states if distilling
                )
                action_loss = output.loss

            # Compute distillation loss
            # Extract student hidden states from model output
            # Hidden states are [batch_size, seq_len, 4096]
            student_hidden_states_full = output.hidden_states[-1]  # Get last layer

            # Get batch metadata for frame/episode indices
            batch_size = student_hidden_states_full.shape[0]
            episode_indices = batch.get("episode_indices", np.zeros(batch_size, dtype=np.int32))
            frame_indices = batch.get("frame_indices", np.zeros(batch_size, dtype=np.int32))

            if not isinstance(episode_indices, np.ndarray):
                episode_indices = np.array(episode_indices)
            if not isinstance(frame_indices, np.ndarray):
                frame_indices = np.array(frame_indices)

            # Load precomputed teacher hidden states for this batch
            # Build lookup table on first batch: (episode_idx, frame_idx) -> h5_index
            # This avoids searching through the entire h5 file for each sample
            if batch_idx == 0:
                teacher_ep_indices = teacher_dataset_file["episode_indices"][:]
                teacher_fr_indices = teacher_dataset_file["frame_indices"][:]
                teacher_lookup = {}
                for idx in range(len(teacher_ep_indices)):
                    key = (int(teacher_ep_indices[idx]), int(teacher_fr_indices[idx]))
                    teacher_lookup[key] = idx

            # Look up teacher states for each sample in batch using the lookup dict
            teacher_hidden_aggregated_list = []
            valid_mask_list = []

            for ep_idx, fr_idx in zip(episode_indices, frame_indices):
                ep_idx = int(ep_idx)
                fr_idx = int(fr_idx)
                key = (ep_idx, fr_idx)

                # Direct lookup in precomputed dataset
                idx = teacher_lookup[key]
                teacher_state = torch.from_numpy(
                    np.array(teacher_dataset_file["teacher_states"][idx], dtype=np.float32)
                )
                has_supervision = bool(teacher_dataset_file["has_supervision"][idx])

                teacher_hidden_aggregated_list.append(teacher_state)
                valid_mask_list.append(has_supervision)

            teacher_hidden_aggregated = torch.stack(teacher_hidden_aggregated_list).to(device_id)
            valid_mask = torch.tensor(valid_mask_list, dtype=torch.bool).to(device_id)

            # Aggregate student sequences to single representations
            student_hidden_aggregated = sequence_aggregation_student(student_hidden_states_full)

            # Log aggregated hidden states for first 100 batches (AFTER aggregation)
            if batch_idx < 100:
                log_file = hidden_state_dir / f"batch_{batch_idx:04d}.npz"
                np.savez_compressed(
                    log_file,
                    student_hidden=student_hidden_aggregated.detach().float().cpu().numpy(),
                    teacher_hidden=teacher_hidden_aggregated.detach().float().cpu().numpy(),
                    batch_idx=batch_idx,
                    aggregation_method=cfg.aggregation_method,
                    batch_size=batch_size,
                )

            # Compute KL divergence loss on similarity matrices
            with torch.autocast("cuda", dtype=torch.float32):  # Use full precision for loss
                distill_loss = distillation_loss_fn(
                    student_hidden_aggregated,
                    teacher_hidden_aggregated,
                    valid_mask=valid_mask,
                )

            # Combine losses
            total_loss = action_loss + cfg.distill_weight * distill_loss

            # Normalize loss to account for gradient accumulation
            normalized_loss = total_loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(action_loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())
            recent_distill_losses.append(distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss)

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            # Prepare logging dict
            log_dict = {
                "train_loss": smoothened_loss,
                "action_accuracy": smoothened_action_accuracy,
                "l1_loss": smoothened_l1_loss,
            }

            # Add distillation metrics
            if len(recent_distill_losses) > 0:
                smoothened_distill_loss = sum(recent_distill_losses) / len(recent_distill_losses)
                log_dict["distill_loss"] = smoothened_distill_loss
                log_dict["total_loss"] = smoothened_loss + cfg.distill_weight * smoothened_distill_loss

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(log_dict, step=gradient_step_idx)

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            # Save Model Checkpoint
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if cfg.save_latest_checkpoint_only:
                    # Overwrite latest checkpoint
                    save_checkpoint(
                        vla, optimizer, gradient_step_idx, batch_idx,
                        run_dir, processor, vla_dataset, cfg,
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
                        vla, optimizer, gradient_step_idx, batch_idx,
                        checkpoint_dir_step, processor, vla_dataset, cfg,
                        distributed_state, adapter_dir_step
                    )

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()