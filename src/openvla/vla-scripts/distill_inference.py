"""
distill_inference.py

Runs distillation inference on SomethingSomethingV2 dataset.
Follows the pattern from run_libero_eval.py but for custom data.

Extracts student action sequences from frozen OpenVLA backbone for each video,
paired with teacher latents loaded from .npy files.

Usage:

python vla-scripts/distill_inference.py \
    --pretrained_checkpoint openvla/openvla-7b \
    --data_npy_dir /home/elias/Thesis/lapa_latents \
    --video_dir /home/elias/Thesis/raw_datasets/sthv2/20bn-something-something-v2 \
    --teacher_latent_dir /home/elias/Thesis \
    --load_in_8bit True \
    --center_crop True \
    --num_samples 16000

Arguments:
    --pretrained_checkpoint: HuggingFace model ID (default: openvla/openvla-7b)
    --data_npy_dir: Directory with .npy files containing video_id, prompt, teacher_latent
    --video_dir: Directory with .webm video files
    --teacher_latent_dir: Directory where output student_inference_results.npy will be saved
    --num_samples: Optional, process only first N samples
    --load_in_4bit: Optional, use 4-bit quantization
    --center_crop: Optional, center crop images (use if model trained with augmentations)

Outputs:
    {teacher_latent_dir}/student_inference_results.npy containing:
    - student_actions: list of [seq_len, 7] action sequences
    - teacher_latents: list of [4] teacher latent vectors
    - video_ids: list of video ID strings
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import torch
import tqdm
from PIL import Image

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append(str(Path(__file__).parent.parent))
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)

# Video loading
import cv2


# ============================================================
# VIDEO LOADING UTILITIES
# ============================================================

def load_video_frames(video_path: Path, frame_indices: list) -> list:
    """Load specific frames from a video file."""
    frames = []
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        cap.release()
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return []

    return frames


def extract_frame_indices(video_path: Path, num_frames: int = 30, frame_offset: int = 2) -> list:
    """Extract frame indices at specified intervals."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return []

    target_count = num_frames // frame_offset
    frame_indices = []
    current_frame = 0

    while len(frame_indices) < target_count and current_frame < total_frames:
        frame_indices.append(current_frame)
        current_frame += frame_offset

    return frame_indices


# ============================================================
# CONFIG
# ============================================================

@dataclass
class DistillInferenceConfig:
    """Configuration for distillation inference."""
    # fmt: off

    # Model parameters
    pretrained_checkpoint: Union[str, Path] = "openvla-7b"  # Path or HF ID of model
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = False  # Center crop images if model trained with augmentations
    model_family: str = "openvla"  # Model family
    unnorm_key: str = "bridge_orig"  # Action denormalization key

    # Data parameters
    teacher_latent_dir: Union[str, Path] = ""  # Dir with teacher .npy files
    data_npy_dir: Union[str, Path] = ""  # Dir with training data .npy files
    video_dir: Union[str, Path] = ""  # Dir with video files

    # Inference parameters
    num_frames: int = 30
    frame_offset: int = 2
    num_samples: Optional[int] = None  # If set, only process first N samples

    # Logging
    run_id_note: Optional[str] = None
    use_wandb: bool = False
    wandb_project: str = "openvla-distillation"
    wandb_entity: str = "eliaskallioras"

    # Other
    seed: int = 7

    # fmt: on


# ============================================================
# INFERENCE
# ============================================================

@draccus.wrap()
def distill_inference(cfg: DistillInferenceConfig) -> None:
    """Run inference on our data following libero_eval pattern."""

    print("\n" + "="*70)
    print("OpenVLA Distillation Inference")
    print("="*70 + "\n")

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor using eval script utilities
    print(f"Loading model from {cfg.pretrained_checkpoint}")
    vla = get_model(cfg)
    if not cfg.load_in_8bit:
        vla = vla.to(device)
    vla = vla.eval()

    print(f"Loading processor from {cfg.pretrained_checkpoint}")
    processor = get_processor(cfg)

    # Get image resize size
    resize_size = get_image_resize_size(cfg)

    # Load dataset
    npy_files = sorted(Path(cfg.data_npy_dir).glob("*.npy"), key=lambda x: int(x.stem.rsplit("_", 1)[0]))
    if cfg.num_samples is not None:
        npy_files = npy_files[:cfg.num_samples]
    print(f"Loaded {len(npy_files)} samples")

    # Initialize metrics
    total_samples = 0
    student_actions_list = []
    teacher_latents_list = []
    video_ids_list = []

    # Process samples
    print("\nProcessing samples...")
    with torch.no_grad():
        for npy_file in tqdm.tqdm(npy_files, desc="Processing"):
            try:
                # Load NPY file
                data = np.load(npy_file, allow_pickle=True).item()
                video_id = data.get("video_id", npy_file.stem.rsplit("_", 1)[0])
                prompt = data["prompt"]
                teacher_latent = np.array(data["teacher_latent"], dtype=np.float32)  # [4]

                # Load video and extract frames at intervals
                video_path = Path(cfg.video_dir) / f"{video_id}.webm"
                if not video_path.exists():
                    print(f"  ⚠️ Video not found: {video_path}")
                    continue

                frame_indices = extract_frame_indices(video_path, cfg.num_frames, cfg.frame_offset)
                if not frame_indices:
                    print(f"  ⚠️ Could not extract frames from {video_path}")
                    continue

                frames = load_video_frames(video_path, frame_indices)
                if not frames:
                    print(f"  ⚠️ Could not load frames from {video_path}")
                    continue

                student_actions = []
                for frame_pil in frames:
                    # Resize frame to match model input size
                    frame_resized = frame_pil.resize((resize_size, resize_size))

                    obs = {"full_image": np.array(frame_resized)}

                    action = get_action(cfg, vla, obs, prompt.lower(), processor=processor)  # [7]

                    student_actions.append(action.reshape(1, -1))  # Reshape to [1, 7] for stacking

                # Stack actions [seq_len, 7]
                student_action_seq = np.concatenate(student_actions, axis=0)  # [seq_len, 7]

                # Store results
                student_actions_list.append(student_action_seq)
                teacher_latents_list.append(teacher_latent)  # [4]
                video_ids_list.append(video_id)
                total_samples += 1

            except Exception as e:
                print(f"  ❌ Error processing {npy_file}: {e}")
                continue

    print(f"\n✅ Successfully processed {total_samples} samples")

    # Log results
    if student_actions_list:
        print(f"\nStudent action sequences:")
        print(f"  - Shape examples: {[s.shape for s in student_actions_list[:3]]}")
        print(f"  - Expected: (seq_len, 7)")
        print(f"\nTeacher latents:")
        print(f"  - Shape examples: {[t.shape for t in teacher_latents_list[:3]]}")
        print(f"  - Expected: (4,)")

    # Save results for autoencoder training
    results_file = Path(cfg.teacher_latent_dir) / "student_inference_results.npy"
    results = {
        "student_actions": student_actions_list,
        "teacher_latents": teacher_latents_list,
        "video_ids": video_ids_list,
    }
    np.save(results_file, results, allow_pickle=True)
    print(f"\nSaved results to {results_file}")

    # Log to W&B
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=f"inference-{cfg.run_id_note or 'default'}",
        )
        wandb.log({
            "total_samples": total_samples,
            "avg_seq_len": np.mean([s.shape[0] for s in student_actions_list]),
        })
        wandb.finish()


if __name__ == "__main__":
    distill_inference()
