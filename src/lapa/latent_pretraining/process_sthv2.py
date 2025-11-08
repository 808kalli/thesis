"""
Usage:

python -m latent_pretraining.process_sthv2 --videos_dir /workspace/thesis/raw_datasets/sthv2/20bn-something-something-v2 \
      --labels_json /workspace/thesis/raw_datasets/sthv2/train.json \
      --lapa_checkpoint /workspace/thesis/src/lapa/lapa_checkpoints/params_sthv2 \
      --output_dir /workspace/thesis/lapa_latents
"""

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


# ---------------------------------------------------------------------------- #
#                               Helper Classes                                 #
# ---------------------------------------------------------------------------- #

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
    lapa_checkpoint: Path,
    num_frames: int = 30,
    image_size: int = 256,
):
    """Main processing loop."""
    # ---------------- Load dataset ----------------
    with open(labels_json, 'r') as f:
        train_data = json.load(f)
    id_to_label = {item['id']: item['label'] for item in train_data}
    valid_ids = set(id_to_label.keys())
    print(f"Loaded {len(valid_ids)} valid video IDs from {labels_json}")

    # ---------------- Setup output ----------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Initialize LAPA ----------------
    print("Loading LAPA model...")

    tokenizer = VideoLLaMAConfig.get_tokenizer_config()
    llama = VideoLLaMAConfig.get_default_config()
    tokenizer.vocab_file = "lapa_checkpoints/tokenizer.model"

    # Distributed and random seed init
    jax_dist_cfg = JaxDistributedConfig.get_default_config()
    JaxDistributedConfig.initialize(jax_dist_cfg)
    set_random_seed(1234)

    lapa_config = {
        'image_size': image_size,
        'tokens_per_delta': 4,
        'vqgan_checkpoint': 'lapa_checkpoints/vqgan',
        'vocab_file': 'lapa_checkpoints/tokenizer.model',
        'multi_image': 1,
        'jax_distributed': jax_dist_cfg,
        'seed': 1234,
        'mesh_dim': "1,-1,1,1",
        'dtype': "bf16",
        'load_llama_config': "7b",
        'update_llama_config': "dict(delta_vocab_size=8,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)",
        'load_checkpoint': f"params::{lapa_checkpoint}",
        'codebook_size': 8,
        'tokenizer': tokenizer,
        'llama': llama,
    }

    lapa_model = LAPAInference(**lapa_config)
    print("✅ LAPA model loaded successfully")

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
                teacher_latent = lapa_model.inference(
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
                        'teacher_latent': teacher_latent,
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
        description="Generate LAPA teacher latents from Something-Something V2 videos"
    )

    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Directory containing .webm videos')
    parser.add_argument('--labels_json', type=str, required=True,
                        help='Path to train.json with labels and IDs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for .npy files')
    parser.add_argument('--lapa_checkpoint', type=str, required=True,
                        help='Path to LAPA checkpoint')
    parser.add_argument('--num_frames', type=int, default=30,
                        help='Number of frames to extract from start of video')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for LAPA model')

    args = parser.parse_args()

    videos_dir = Path(args.videos_dir).expanduser()
    labels_json = Path(args.labels_json).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    lapa_checkpoint = Path(args.lapa_checkpoint).expanduser()

    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos dir not found: {videos_dir}")
    if not labels_json.exists():
        raise FileNotFoundError(f"Labels JSON not found: {labels_json}")
    if not lapa_checkpoint.exists():
        raise FileNotFoundError(f"LAPA checkpoint not found: {lapa_checkpoint}")

    process_sthv2_dataset(
        videos_dir=videos_dir,
        labels_json=labels_json,
        output_dir=output_dir,
        lapa_checkpoint=lapa_checkpoint,
        num_frames=args.num_frames,
        image_size=args.image_size,
    )

    print(f"\n✅ Processing complete! Files saved to {output_dir}")
