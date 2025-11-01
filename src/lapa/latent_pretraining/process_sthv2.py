'''
Usage:

python process_sthv2.py \
    --videos_dir ~/Thesis/raw_datasets/sthv2/20bn-something-something-v2 \
    --labels_json ~/Thesis/raw_datasets/sthv2/train.json \
    --lapa_checkpoint /path/to/lapa/checkpoint \
    --output_dir ~/Thesis/teacher_latents \
'''

import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from latent_pretraining.sampler_latent_pretrain import DeltaSampler
from latent_pretraining.delta_llama import VideoLLaMAConfig
from tux import JaxDistributedConfig, set_random_seed


class FLAGSClass:
    def __init__(self, flag_dict):
        for key, value in flag_dict.items():
            setattr(self, key, value)


class LAPAInference:
    def __init__(
        self,
        image_size: int = 256,
        **kwargs,
    ) -> None:
        flags = FLAGSClass(kwargs)

        self.model = DeltaSampler(FLAGS=flags)
        self.image_size = image_size
        self.tokens_per_delta = kwargs['tokens_per_delta']
        self.task_description = None

    def inference(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs):
        assert image.dtype == np.uint8
        image = Image.fromarray(image)
        prompts = [{'image': [image], 'question': task_description}]

        latent_output = self.model(prompts)
        latent_action = latent_output[0]

        return latent_action


def load_video_frames(video_path: Path, frame_offset: int = 30):
    """
    Load video frames with specified offset between frames.

    Args:
        video_path: Path to video file (.webm)
        frame_offset: Distance between frames to sample

    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_offset == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_idx += 1

    cap.release()
    return frames


def process_sthv2_dataset(
    videos_dir: Path,
    labels_json: Path,
    output_dir: Path,
    lapa_checkpoint: Path,
    frame_offset: int = 30,
    image_size: int = 256,
):
    """
    Process Something-Something V2 dataset and generate LAPA teacher latents.

    Args:
        videos_dir: Directory containing .webm videos (named 1.webm, 2.webm, etc.)
        labels_json: Path to train.json containing labels and IDs
        output_dir: Output directory for .npy files
        lapa_checkpoint: Path to LAPA checkpoint
        frame_offset: Frame sampling offset
        image_size: Image size for LAPA
    """
    # Load train.json to get valid video IDs and labels
    with open(labels_json, 'r') as f:
        train_data = json.load(f)

    # Create id -> label mapping and extract valid IDs
    id_to_label = {item['id']: item['label'] for item in train_data}
    valid_ids = set(id_to_label.keys())

    print(f"Loaded {len(valid_ids)} valid video IDs from {labels_json}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LAPA model
    print("Loading LAPA model...")
    lapa_config = {
        'image_size': image_size,
        'checkpoint_path': str(lapa_checkpoint),
        'tokens_per_delta': 4,  # LAPA uses 4 latent action tokens
    }
    lapa_model = LAPAInference(**lapa_config)
    print("LAPA model loaded ✅")

    # Get all video files and filter by valid IDs
    all_video_files = list(videos_dir.glob("*.webm"))
    video_files = [v for v in all_video_files if v.stem in valid_ids]

    print(f"Found {len(video_files)} / {len(all_video_files)} videos matching train.json IDs")
    print(f"Frame offset: {frame_offset}")
    print(f"Output directory: {output_dir}")

    # Process each video
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_id = video_path.stem  # e.g., "78687"
        instruction = id_to_label[video_id]

        # Load frames with offset
        try:
            frames = load_video_frames(video_path, frame_offset)

            if len(frames) == 0:
                print(f"Warning: No frames extracted from {video_id}, skipping")
                continue

            # Use the first frame as RGB observation (or middle frame)
            # For distillation, we use a single frame similar to OpenVLA's single-image input
            middle_idx = len(frames) // 2
            rgb_frame = frames[middle_idx]

            # Resize to expected image size
            rgb_frame_resized = cv2.resize(rgb_frame, (image_size, image_size))

            # Generate teacher latent using LAPA
            teacher_latent = lapa_model.inference(
                image=rgb_frame_resized,
                task_description=instruction
            )

            # Save to .npy with format: (rgb, prompt, teacher_latent, id)
            output_path = output_dir / f"{video_id}.npy"
            np.save(
                output_path,
                {
                    'rgb': rgb_frame_resized,  # [H, W, C] uint8
                    'prompt': instruction,      # str
                    'teacher_latent': teacher_latent,  # [4] latent action tokens
                    'id': video_id,            # str
                },
                allow_pickle=True
            )

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Something-Something V2 dataset and generate LAPA teacher latents"
    )

    # Dataset paths
    parser.add_argument('--videos_dir', type=str,
                        default='~/Thesis/raw_datasets/sthv2/20bn-something-something-v2',
                        help='Directory containing .webm videos')
    parser.add_argument('--labels_json', type=str,
                        default='~/Thesis/raw_datasets/sthv2/train.json',
                        help='Path to train.json with labels and IDs')
    parser.add_argument('--output_dir', type=str,
                        default='~/Thesis/teacher_latents',
                        help='Output directory for .npy files with teacher latents')
    parser.add_argument('--lapa_checkpoint', type=str, required=True,
                        help='Path to LAPA checkpoint')
    parser.add_argument('--frame_offset', type=int, default=30,
                        help='Frame sampling offset (distance between frames)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for LAPA model')

    args = parser.parse_args()

    # Expand paths
    videos_dir = Path(args.videos_dir).expanduser()
    labels_json = Path(args.labels_json).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    lapa_checkpoint = Path(args.lapa_checkpoint).expanduser()

    # Validate paths
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    if not labels_json.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_json}")
    if not lapa_checkpoint.exists():
        raise FileNotFoundError(f"LAPA checkpoint not found: {lapa_checkpoint}")

    # Process dataset
    process_sthv2_dataset(
        videos_dir=videos_dir,
        labels_json=labels_json,
        output_dir=output_dir,
        lapa_checkpoint=lapa_checkpoint,
        frame_offset=args.frame_offset,
        image_size=args.image_size,
    )

    print(f"\nProcessing complete! Files saved to {output_dir}")
