"""
Extract Hidden States from VLA Models (Vanilla or Distilled)

This script performs a single inference pass through the dataset with a frozen VLA model
and saves hidden states + global indices for later analysis.

Usage:
    # Extract from vanilla model
    python vla-scripts/extract_hidden_states.py \
        --model_family openvla \
        --pretrained_checkpoint openvla/openvla-7b \
        --output_dir runs/hidden_states_vanilla

    # Extract from distilled model
    python vla-scripts/extract_hidden_states.py \
        --model_family openvla \
        --pretrained_checkpoint openvla/openvla-7b \
        --load_from_checkpoint ../../runs/distill_run/checkpoints \
        --output_dir runs/hidden_states_distilled
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import numpy as np
import torch
import tqdm
from accelerate import PartialState
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor

# Append current directory to import prismatic
sys.path.append("../..")
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


@dataclass
class ExtractConfig:
    """Configuration for hidden state extraction."""

    # Model Parameters
    model_family: str = "openvla"
    pretrained_checkpoint: Path = Path("openvla/openvla-7b")
    load_from_checkpoint: Optional[Path] = None  # Path to distilled checkpoint directory (merged model)

    # Data Parameters
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    dataset_name: str = "droid_wipe"
    image_aug: bool = False

    # Inference Parameters
    batch_size: int = 16
    shuffle_buffer_size: int = 100_000
    num_workers: int = 0
    max_batches: int = 5000  # Maximum number of batches to extract (set to 0 for full dataset)

    # Output Parameters
    output_dir: Path = Path("runs/hidden_states_extraction")

    # System Parameters
    seed: int = 7


def extract_hidden_states(cfg: ExtractConfig) -> None:
    """Extract hidden states from VLA model with frozen backbone."""

    # Initialize Distributed State
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if distributed_state.is_main_process:
        print("=" * 80)
        print("Hidden State Extraction Configuration")
        print("=" * 80)
        print(f"Model Family: {cfg.model_family}")
        print(f"Pretrained Checkpoint: {cfg.pretrained_checkpoint}")
        print(f"Load from Checkpoint: {cfg.load_from_checkpoint}")
        print(f"Dataset: {cfg.dataset_name}")
        print(f"Batch Size: {cfg.batch_size}")
        print(f"Output Directory: {cfg.output_dir}")
        print("=" * 80)

    # Determine which checkpoint to load
    model_path = cfg.load_from_checkpoint if cfg.load_from_checkpoint is not None else cfg.pretrained_checkpoint

    # Load VLA model
    if distributed_state.is_main_process:
        model_type = "distilled" if cfg.load_from_checkpoint is not None else "vanilla"
        print(f"\nLoading {model_type} VLA model from {model_path}...")

    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if distributed_state.is_main_process:
        print("✓ Model loaded")

    vla = vla.to(device_id)

    # Freeze all parameters (no gradient computation needed)
    for param in vla.parameters():
        param.requires_grad = False

    vla.eval()  # Set to evaluation mode

    if distributed_state.is_main_process:
        print("✓ Model loaded and frozen")

    # Load processor (from the same checkpoint as the model)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )

    # Create Action Tokenizer (same as training)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Create prompt builder
    if cfg.model_family == "openvla":
        prompt_builder_fn = PurePromptBuilder
    else:
        prompt_builder_fn = VicunaV15ChatPromptBuilder

    # Load dataset
    if distributed_state.is_main_process:
        print(f"\nLoading dataset: {cfg.dataset_name}...")

    # Create batch transform (same as training)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=prompt_builder_fn,
    )

    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=1,  # No shuffling - sequential pass for complete extraction
        image_aug=cfg.image_aug,
    )

    # Save dataset statistics
    save_dataset_statistics(vla_dataset.dataset_statistics, output_dir)

    # Create dataloader with padding collator
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right"
    )

    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=cfg.num_workers,
    )

    if distributed_state.is_main_process:
        print(f"✓ Dataset loaded: {len(vla_dataset)} samples")
        total_batches = len(dataloader)
        batches_to_extract = cfg.max_batches if cfg.max_batches > 0 else total_batches
        print(f"✓ Total batches available: {total_batches}")
        print(f"✓ Batches to extract: {batches_to_extract}")

    # Extract hidden states
    if distributed_state.is_main_process:
        print("\n" + "=" * 80)
        print("Starting Hidden State Extraction")
        print("=" * 80)

    with torch.no_grad():  # No gradients needed
        for batch_idx, batch in enumerate(tqdm.tqdm(
            dataloader,
            desc="Extracting hidden states",
            disable=not distributed_state.is_main_process,
            total=cfg.max_batches if cfg.max_batches > 0 else len(dataloader)
        )):
            # Check if we've reached max_batches
            if cfg.max_batches > 0 and batch_idx >= cfg.max_batches:
                if distributed_state.is_main_process:
                    print(f"\nReached max_batches limit ({cfg.max_batches}). Stopping extraction.")
                break

            # Move batch to device
            input_ids = batch["input_ids"].to(device_id)
            attention_mask = batch["attention_mask"].to(device_id)
            pixel_values = batch["pixel_values"].to(torch.bfloat16).to(device_id)
            labels = batch["labels"]

            # Forward pass with output_hidden_states=True
            output = vla(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                output_hidden_states=True,
            )

            # Extract last layer hidden states
            # Shape: [batch_size, seq_len, hidden_dim]
            student_hidden_states_full = output.hidden_states[-1]

            # Aggregate using mean (same as training)
            # Shape: [batch_size, hidden_dim]
            student_hidden_aggregated = student_hidden_states_full.mean(dim=1)

            # Get global indices for this batch
            global_indices = batch["global_indices"].cpu().numpy()
            batch_size = student_hidden_aggregated.shape[0]

            # Save aggregated hidden states for this batch
            log_file = output_dir / f"batch_{batch_idx:04d}.npz"
            np.savez_compressed(
                log_file,
                student_hidden=student_hidden_aggregated.detach().float().cpu().numpy(),
                global_indices=global_indices,
                batch_idx=batch_idx,
                batch_size=batch_size,
            )

            # Print progress every 100 batches
            if distributed_state.is_main_process and (batch_idx + 1) % 100 == 0:
                print(f"Saved batch {batch_idx + 1}/{cfg.max_batches if cfg.max_batches > 0 else len(dataloader)}")

    if distributed_state.is_main_process:
        print("\n" + "=" * 80)
        print("✓ Hidden State Extraction Complete")
        print("=" * 80)
        actual_batches = min(batch_idx + 1, cfg.max_batches if cfg.max_batches > 0 else batch_idx + 1)
        print(f"Total batches saved: {actual_batches}")
        print(f"Output directory: {output_dir}")
        print("=" * 80)


@draccus.wrap()
def main(cfg: ExtractConfig) -> None:
    extract_hidden_states(cfg)


if __name__ == "__main__":
    main()
