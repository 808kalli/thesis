"""
Usage:

python -m process_sthv2_openvla --videos_dir /workspace/thesis/raw_datasets/sthv2/20bn-something-something-v2 \
      --labels_json /workspace/thesis/raw_datasets/sthv2/train.json \
      --pretrained_checkpoint <PATH_TO_OPENVLA_CHECKPOINT_OR_HF_MODEL_ID> \
      --output_dir /workspace/thesis/openvla_predictions

Example with HuggingFace model:
python -m process_sthv2_openvla --videos_dir /path/to/videos \
      --labels_json /path/to/labels.json \
      --pretrained_checkpoint "openvla/openvla-7b" \
      --output_dir /path/to/output
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm
import cv2
import sys

import torch

# Append parent directories to path so we can import from experiments.robot
script_dir = Path(__file__).parent  # vla-scripts/
src_openvla_dir = script_dir.parent  # src/openvla/
sys.path.insert(0, str(src_openvla_dir))  # Add src/openvla/ to path
from experiments.robot.openvla_utils import get_vla, get_processor, get_vla_action
from prismatic.util import set_global_seed


# ---------------------------------------------------------------------------- #
#                          Configuration & Model Loading                       #
# ---------------------------------------------------------------------------- #

@dataclass
class OpenVLAConfig:
    """Minimal config for OpenVLA model loading."""
    pretrained_checkpoint: str
    load_in_8bit: bool = False
    load_in_4bit: bool = False


class OpenVLAInference:
    def __init__(
        self,
        pretrained_checkpoint: str,
        image_size: int = 256,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        """
        Initialize OpenVLA model for inference.

        Args:
            pretrained_checkpoint: Path or HuggingFace model ID (e.g., "openvla/openvla-7b")
            image_size: Image size for preprocessing
            load_in_8bit: Load model in 8-bit quantization
            load_in_4bit: Load model in 4-bit quantization
        """
        print(f"Loading OpenVLA model: {pretrained_checkpoint}")

        # Create config
        cfg = OpenVLAConfig(
            pretrained_checkpoint=pretrained_checkpoint,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )

        # Load model using the standard get_vla function
        self.model = get_vla(cfg)
        self.model.eval()

        # Load processor
        self.processor = get_processor(cfg)
        self.image_size = image_size
        self.pretrained_checkpoint = pretrained_checkpoint

    def inference(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> np.ndarray:
        """
        Run inference on a single frame.

        Args:
            image: RGB image as numpy array (H, W, 3) with values in [0, 255]
            task_description: Task instruction/prompt

        Returns:
            Predicted action as numpy array [7] (continuous action values)
        """
        assert image.dtype == np.uint8

        # Prepare observation dict
        obs = {"full_image": image}

        with torch.no_grad():
            # Get predicted action from OpenVLA
            # This returns a [7] tensor of continuous action values
            predicted_action = get_vla_action(
                self.model,
                self.processor,
                self.pretrained_checkpoint,
                obs,
                task_description,
                unnorm_key="bridge_orig",  # Use Bridge dataset normalization stats
                center_crop=False,
            )

        # Convert to numpy if needed
        if isinstance(predicted_action, torch.Tensor):
            predicted_action = predicted_action.cpu().numpy()

        return predicted_action.astype(np.float32)


# ---------------------------------------------------------------------------- #
#                          Dataset Processing Logic                            #
# ---------------------------------------------------------------------------- #

def load_video_frames_at_intervals(video_path: Path, frame_interval: int = 30):
    """Load frames at 30-frame intervals (0, 30, 60, ...).

    If video has fewer than 30 frames, only return frame 0.
    """
    cap = cv2.VideoCapture(str(video_path))
    all_frames = []
    frame_idx = 0

    # Load all frames from video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1

    cap.release()

    # If fewer than 30 frames, return only frame 0
    if len(all_frames) < frame_interval:
        return [all_frames[0]] if all_frames else []

    # Otherwise, return frames at intervals: 0, 30, 60, ...
    frames_at_intervals = []
    idx = 0
    while idx < len(all_frames):
        frames_at_intervals.append(all_frames[idx])
        idx += frame_interval

    return frames_at_intervals


def process_sthv2_dataset(
    videos_dir: Path,
    labels_json: Path,
    output_dir: Path,
    pretrained_checkpoint: str,
    num_frames: int = 30,
    image_size: int = 256,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
):
    """Main processing loop to generate OpenVLA predictions on Something-Something V2."""
    # ---------------- Load dataset ----------------
    with open(labels_json, 'r') as f:
        train_data = json.load(f)
    id_to_label = {item['id']: item['label'] for item in train_data}
    valid_ids = set(id_to_label.keys())
    print(f"Loaded {len(valid_ids)} valid video IDs from {labels_json}")

    # ---------------- Setup output ----------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Initialize OpenVLA ----------------
    print("Loading OpenVLA model...")
    set_global_seed(7)

    openvla_model = OpenVLAInference(
        pretrained_checkpoint=pretrained_checkpoint,
        image_size=image_size,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )
    print("✅ OpenVLA model loaded successfully")

    # ---------------- Collect videos ----------------
    all_video_files = list(videos_dir.glob("*.webm"))
    video_files = [v for v in all_video_files if v.stem in valid_ids]

    print(f"Found {len(video_files)} / {len(all_video_files)} valid videos.")
    print(f"Num frames per video: {num_frames}")
    print(f"Output dir: {output_dir}")

    # ---------------- Process videos ----------------
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_id = video_path.stem
        instruction = id_to_label[video_id]
        try:
            frames = load_video_frames_at_intervals(video_path, frame_interval=30)
            if not frames:
                print(f"⚠️  No frames extracted from {video_id}, skipping.")
                continue

            # Process each frame individually and save to separate .npy files
            for interval_idx, frame in enumerate(frames):
                # Actual video frame number (0, 30, 60, ...)
                actual_frame_num = interval_idx * 30

                rgb_frame_resized = cv2.resize(frame, (image_size, image_size))

                # Run inference
                student_action = openvla_model.inference(
                    image=rgb_frame_resized,
                    task_description=instruction
                )

                # Save individual frame + action to .npy
                # Filename format: {video_id}_{actual_frame_number}.npy
                output_path = output_dir / f"{video_id}_{actual_frame_num}.npy"
                np.save(
                    output_path,
                    {
                        'rgb': rgb_frame_resized,
                        'prompt': instruction,
                        'student_action': student_action,  # [7] continuous action values
                        'video_id': video_id,
                        'frame_number': actual_frame_num,
                    },
                    allow_pickle=True
                )

        except Exception as e:
            print(f"❌ Error processing {video_id}: {e}")
            continue


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate OpenVLA predictions from Something-Something V2 videos"
    )

    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Directory containing .webm videos')
    parser.add_argument('--labels_json', type=str, required=True,
                        help='Path to train.json with labels and IDs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for .npy files')
    parser.add_argument('--pretrained_checkpoint', type=str, required=True,
                        help='Path or HuggingFace model ID for OpenVLA (e.g., openvla/openvla-7b)')
    parser.add_argument('--load_in_8bit', action='store_true',
                        help='Load model in 8-bit quantization')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Load model in 4-bit quantization')
    parser.add_argument('--num_frames', type=int, default=30,
                        help='Number of frames to extract from start of video')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for OpenVLA model')

    args = parser.parse_args()

    videos_dir = Path(args.videos_dir).expanduser()
    labels_json = Path(args.labels_json).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos dir not found: {videos_dir}")
    if not labels_json.exists():
        raise FileNotFoundError(f"Labels JSON not found: {labels_json}")

    process_sthv2_dataset(
        videos_dir=videos_dir,
        labels_json=labels_json,
        output_dir=output_dir,
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_frames=args.num_frames,
        image_size=args.image_size,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    print(f"\n✅ Processing complete! Files saved to {output_dir}")
