"""
Create 10 episode .npy files from LIBERO Object dataset (one per task).

This script:
1. Loads episodes [0, 1, 3, 4, 6, 7, 9, 12, 14, 16] from libero_object_noops
2. Extracts all frames from each episode
3. Creates .npy files in rlds_dataset_builder/openvla_libero_object/data/train_10eps

Each .npy file contains:
  - action: (num_frames, 7)
  - observation: dict with image, wrist_image, state
  - language_instruction: str
  - episode_metadata: dict with episode_index, task_index, num_steps
  - global_indices: (num_frames,) - original global indices in dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tqdm
import subprocess
from PIL import Image
from io import BytesIO
import tempfile

# Paths
DATASET_DIR = Path("/home/elias/Thesis/raw_datasets/libero_object_noops")
OUTPUT_DIR = Path("/home/elias/Thesis/scripts/rlds_dataset_builder/openvla_libero_object/data/train_10eps")

# Episode list (one per task, from find_libero_object_episodes.py)
EPISODE_LIST = [0, 1, 3, 4, 6, 7, 9, 12, 14, 16]

# LIBERO-Object task descriptions
LIBERO_OBJECT_TASKS = [
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_ketchup_and_place_it_in_the_basket",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    "pick_up_the_butter_and_place_it_in_the_basket",
    "pick_up_the_milk_and_place_it_in_the_basket",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
]


def extract_all_frames_from_video(video_path):
    """Extract ALL frames from video using ffmpeg."""
    temp_dir = tempfile.mkdtemp(prefix="libero_frames_")

    try:
        frame_pattern = temp_dir + "/frame_%04d.png"
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-f', 'image2',
            '-pix_fmt', 'rgb24',
            frame_pattern,
            '-loglevel', 'error'
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=30)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:500]}")

        # Load frames
        frames = []
        frame_files = sorted(Path(temp_dir).glob("frame_*.png"))

        for frame_file in frame_files:
            img = Image.open(frame_file).convert('RGB')
            # Resize to 256x256 to match libero_spatial format
            img = img.resize((256, 256), Image.Resampling.BILINEAR)
            frames.append(np.array(img, dtype=np.uint8))

        return frames

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_episode_data_from_dataset(episode_idx, dataset_dir):
    """Load all data for an episode from the lerobot dataset."""
    dataset_dir = Path(dataset_dir)
    data_dir = dataset_dir / 'data'
    videos_dir = dataset_dir / 'videos'

    # Find video files
    image_video = None
    wrist_video = None

    for chunk_dir in sorted(videos_dir.glob('chunk-*')):
        img_dir = chunk_dir / 'observation.images.image'
        wrist_dir = chunk_dir / 'observation.images.wrist_image'

        potential_img = img_dir / f'episode_{episode_idx:06d}.mp4'
        potential_wrist = wrist_dir / f'episode_{episode_idx:06d}.mp4'

        if potential_img.exists():
            image_video = potential_img
        if potential_wrist.exists():
            wrist_video = potential_wrist

    if not image_video or not wrist_video:
        raise FileNotFoundError(f"Videos not found for episode {episode_idx}")

    # Extract frames from videos
    print(f"  Extracting frames from videos...")
    image_frames = extract_all_frames_from_video(image_video)
    wrist_frames = extract_all_frames_from_video(wrist_video)

    num_frames = len(image_frames)

    # Load metadata and actions from parquet files
    parquet_files = sorted(data_dir.glob('**/*.parquet'))

    actions = []
    states = []
    global_indices = []
    task_index = None

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        episode_data = df[df['episode_index'] == episode_idx]

        if not episode_data.empty:
            # Sort by frame_index
            episode_data = episode_data.sort_values('frame_index')

            actions.extend(episode_data['action'].tolist())
            states.extend(episode_data['observation.state'].tolist())
            global_indices.extend(episode_data['index'].tolist())

            if task_index is None:
                task_index = int(episode_data['task_index'].iloc[0])

    if task_index is None:
        raise ValueError(f"No data found for episode {episode_idx}")

    # Convert to numpy arrays
    actions = np.array(actions, dtype=np.float32)
    states = np.array(states, dtype=np.float32)
    global_indices = np.array(global_indices, dtype=np.int32)
    image_frames = np.array(image_frames, dtype=np.uint8)
    wrist_frames = np.array(wrist_frames, dtype=np.uint8)

    # Get language instruction
    language_instruction = LIBERO_OBJECT_TASKS[task_index]

    return {
        'action': actions,
        'observation': {
            'image': image_frames,
            'wrist_image': wrist_frames,
            'state': states,
        },
        'language_instruction': language_instruction,
        'episode_metadata': {
            'episode_index': episode_idx,
            'task_index': task_index,
            'num_steps': num_frames,
        },
        'global_indices': global_indices,
    }


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Creating 10 episode .npy files from LIBERO Object dataset")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Episodes: {EPISODE_LIST}\n")

    for idx, episode_idx in enumerate(tqdm.tqdm(EPISODE_LIST, desc="Processing episodes")):
        print(f"\nEpisode {episode_idx} (task {idx}):")

        try:
            # Load episode data
            episode_data = load_episode_data_from_dataset(episode_idx, DATASET_DIR)

            # Save to .npy file
            output_file = OUTPUT_DIR / f"episode_{idx}.npy"
            np.save(output_file, episode_data, allow_pickle=True)

            print(f"  ✓ Saved {output_file.name}")
            print(f"    Task: {episode_data['language_instruction']}")
            print(f"    Frames: {episode_data['episode_metadata']['num_steps']}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"✅ Successfully created {len(EPISODE_LIST)} episode files in {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
