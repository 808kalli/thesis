"""
Create supervised episode .npy files from the teacher_dataset_supervised.h5.

This script:
1. Loads global indices from teacher_dataset_supervised.h5
2. Maps global indices to (episode_idx, frame_idx) using dataset parquets
3. Groups supervised frames by episode
4. For each episode with supervised frames, loads the actual data from libero_spatial_noops dataset
5. Creates a new .npy file with only supervised frames, maintaining global indices

Output: /train_supervised/ folder with 432 .npy files (one per supervised episode)
Each .npy contains:
  - action: (num_supervised_frames, 7)
  - observation: dict with image, wrist_image, state, joint_state (all with num_supervised_frames)
  - language_instruction: str
  - episode_metadata: dict with episode_index, task_index, num_steps
  - global_indices: (num_supervised_frames,) - original global indices in dataset
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import tqdm
import argparse

# Paths
SUPERVISED_H5 = Path("/home/elias/Thesis/teacher_dataset_supervised.h5")
LAPA_H5 = Path("/home/elias/Thesis/lapa_hidden_states.h5")
DATASET_DIR = Path("/home/elias/Thesis/raw_datasets/libero_spatial_noops")
OUTPUT_DIR = Path("/home/elias/Thesis/scripts/rlds_dataset_builder/openvla_libero_spatial/data/train_supervised")


def build_global_to_episode_frame_map():
    """Build mapping from global_index -> (episode_idx, frame_idx)."""
    print("Building global index to episode/frame mapping...")

    data_dir = DATASET_DIR / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))

    global_to_ep_frame = {}
    for pf in tqdm.tqdm(parquet_files, desc="Building mapping"):
        df = pd.read_parquet(pf, columns=["index", "episode_index", "frame_index"])
        for _, row in df.iterrows():
            global_idx = int(row["index"])
            episode_idx = int(row["episode_index"])
            frame_idx = int(row["frame_index"])
            global_to_ep_frame[global_idx] = (episode_idx, frame_idx)

    print(f"✓ Built mapping for {len(global_to_ep_frame)} global indices")
    return global_to_ep_frame


def load_supervised_indices():
    """Load global indices from teacher_dataset_supervised.h5."""
    print(f"Loading supervised global indices from {SUPERVISED_H5}...")

    with h5py.File(SUPERVISED_H5, "r") as f:
        global_indices = f["global_indices"][:]

    print(f"✓ Loaded {len(global_indices)} supervised global indices")
    return global_indices


def group_supervised_frames(global_indices, global_to_ep_frame):
    """Group supervised frames by episode."""
    print("Grouping supervised frames by episode...")

    episodes_supervised = defaultdict(list)

    for global_idx in global_indices:
        if global_idx in global_to_ep_frame:
            episode_idx, frame_idx = global_to_ep_frame[global_idx]
            episodes_supervised[episode_idx].append({
                'global_idx': global_idx,
                'frame_idx': frame_idx,
            })

    # Sort frames within each episode by frame_idx
    for episode_idx in episodes_supervised:
        episodes_supervised[episode_idx].sort(key=lambda x: x['frame_idx'])

    print(f"✓ Grouped into {len(episodes_supervised)} episodes with supervised frames")
    return episodes_supervised


def load_dataset_parquet(dataset_dir):
    """Load dataset from parquet files."""
    data_dir = dataset_dir / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    from datasets import load_dataset
    # Convert Path objects to strings
    data_files = [str(f) for f in parquet_files]
    dataset = load_dataset("parquet", data_files=data_files, split="train")
    return dataset


def load_episode_from_existing_npy(episode_idx, existing_episodes_dir):
    """Load full episode data from existing .npy file."""
    episode_file = existing_episodes_dir / f"episode_{episode_idx}.npy"

    if not episode_file.exists():
        return None

    try:
        episode_data = np.load(episode_file, allow_pickle=True).item()
        return episode_data
    except Exception as e:
        print(f"  ✗ Failed to load existing episode {episode_idx}: {e}")
        return None


def extract_frame_from_video(video_path, frame_idx):
    """Extract a single frame from video at specified index using imageio."""
    try:
        import imageio

        # Read all frames from video (imageio handles AV1 codec better than OpenCV)
        try:
            reader = imageio.get_reader(str(video_path), 'ffmpeg')

            # Seek to desired frame
            if frame_idx < len(reader):
                frame = reader.get_data(frame_idx)
                reader.close()

                # Frame is already RGB from imageio
                # Resize to 256x256
                from PIL import Image
                pil_img = Image.fromarray(frame)
                pil_img = pil_img.resize((256, 256), Image.LANCZOS)
                return np.array(pil_img, dtype=np.uint8)
            else:
                reader.close()
                return None
        except Exception as e:
            return None

    except Exception as e:
        return None


def extract_frames_from_lapa(episode_idx, frame_indices, dataset_dir):
    """
    Extract image and wrist_image frames from video files for the given frame indices.

    Video files are organized in dataset_dir/videos/chunk-000/{image_type}/episode_{episode_idx:06d}.mp4
    """
    dataset_dir = Path(dataset_dir)
    video_base = dataset_dir / "videos" / "chunk-000"

    images_dict = {}
    wrist_images_dict = {}

    for frame_idx in frame_indices:
        # Load from video files
        image_video = video_base / "observation.images.image" / f"episode_{episode_idx:06d}.mp4"
        wrist_video = video_base / "observation.images.wrist_image" / f"episode_{episode_idx:06d}.mp4"

        # Extract frames
        if image_video.exists():
            img = extract_frame_from_video(image_video, frame_idx)
            if img is not None:
                images_dict[frame_idx] = img
            else:
                images_dict[frame_idx] = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            images_dict[frame_idx] = np.zeros((256, 256, 3), dtype=np.uint8)

        if wrist_video.exists():
            wrist_img = extract_frame_from_video(wrist_video, frame_idx)
            if wrist_img is not None:
                wrist_images_dict[frame_idx] = wrist_img
            else:
                wrist_images_dict[frame_idx] = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            wrist_images_dict[frame_idx] = np.zeros((256, 256, 3), dtype=np.uint8)

    return images_dict, wrist_images_dict


def create_supervised_episode_npy(episode_idx, supervised_frames, dataset, dataset_dir, output_dir):
    """
    Create supervised episode .npy with REAL data.

    Fetches:
    - action, state from libero_spatial_noops parquet dataset (indexed by global_idx)
    - images from video files indexed by (episode_idx, frame_idx)

    Args:
        episode_idx: Episode index
        supervised_frames: List of dicts with 'frame_idx' and 'global_idx'
        dataset: Loaded libero_spatial_noops dataset (indexed by global_idx)
        dataset_dir: Path to libero_spatial_noops directory (for video extraction)
        output_dir: Directory to save supervised episode .npy

    Returns:
        True if successful, False otherwise
    """
    try:
        # Sort frames by frame_idx to maintain order within episode
        supervised_frames_sorted = sorted(supervised_frames, key=lambda x: x['frame_idx'])
        global_indices_list = np.array([f['global_idx'] for f in supervised_frames_sorted], dtype=np.int32)
        frame_indices = [f['frame_idx'] for f in supervised_frames_sorted]

        # Extract images from video files
        images_dict, wrist_images_dict = extract_frames_from_lapa(episode_idx, frame_indices, dataset_dir)

        # Fetch REAL data from parquet dataset using global indices
        actions_list = []
        images_list = []
        wrist_images_list = []
        states_list = []
        joint_states_list = []
        task_index = None
        language_instruction = None

        for frame_info in supervised_frames_sorted:
            global_idx = int(frame_info['global_idx'])
            frame_idx = int(frame_info['frame_idx'])

            # Fetch action and state from parquet dataset
            sample = dataset[global_idx]
            actions_list.append(sample['action'])
            states_list.append(sample['observation.state'])

            # Get image from LAPA-indexed frames
            images_list.append(images_dict.get(frame_idx, np.zeros((256, 256, 3), dtype=np.uint8)))
            wrist_images_list.append(wrist_images_dict.get(frame_idx, np.zeros((256, 256, 3), dtype=np.uint8)))

            # Get task index and language instruction (same for all frames in episode)
            if task_index is None:
                task_index = int(sample.get('task_index', 0))
                language_instruction = sample.get('language_instruction', f'Task for episode {episode_idx}')

        # Convert lists to numpy arrays
        actions = np.array(actions_list, dtype=np.float32)
        images = np.array(images_list, dtype=np.uint8)
        wrist_images = np.array(wrist_images_list, dtype=np.uint8)
        states = np.array(states_list, dtype=np.float32)

        num_frames = len(supervised_frames_sorted)

        # Create episode dictionary with REAL data
        supervised_episode = {
            'action': actions,  # Shape: (num_frames, 7)
            'observation': {
                'image': images,  # Shape: (num_frames, 256, 256, 3)
                'wrist_image': wrist_images,  # Shape: (num_frames, 256, 256, 3)
                'state': states,  # Shape: (num_frames, 8)
                'joint_state': np.zeros((num_frames, 7), dtype=np.float32),  # Placeholder for now
            },
            'language_instruction': language_instruction,
            'episode_metadata': {
                'episode_index': int(episode_idx),
                'task_index': task_index,
                'num_steps': num_frames,
            },
            'global_indices': global_indices_list,
        }

        # Save supervised episode
        output_file = output_dir / f"episode_{episode_idx}.npy"
        np.save(output_file, supervised_episode, allow_pickle=True)

        return True

    except Exception as e:
        print(f"  ✗ Failed to create episode {episode_idx}: {e}")
        return False


def main():
    """Create supervised episode .npy files."""

    # Check paths
    if not SUPERVISED_H5.exists():
        raise FileNotFoundError(f"Supervised H5 not found: {SUPERVISED_H5}")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Load dataset once
    print("Loading libero_spatial_noops dataset...")
    dataset = load_dataset_parquet(DATASET_DIR)
    print(f"✓ Loaded dataset with {len(dataset)} samples\n")

    # Load and map indices
    global_indices = load_supervised_indices()
    global_to_ep_frame = build_global_to_episode_frame_map()
    print()

    # Group by episode
    episodes_supervised = group_supervised_frames(global_indices, global_to_ep_frame)
    print()

    # Create supervised episode files
    print("Creating supervised episode .npy files...")
    successful = 0

    for episode_idx in tqdm.tqdm(sorted(episodes_supervised.keys()), desc="Episodes"):
        supervised_frames = episodes_supervised[episode_idx]

        if create_supervised_episode_npy(episode_idx, supervised_frames, dataset, DATASET_DIR, OUTPUT_DIR):
            successful += 1

    print(f"\n✓ Created {successful} supervised episode files in {OUTPUT_DIR}")
    print(f"  Total episodes with supervised frames: {len(episodes_supervised)}")

    # Verify output
    output_files = sorted(OUTPUT_DIR.glob("episode_*.npy"))
    print(f"  Actual files created: {len(output_files)}")

    if len(output_files) > 0:
        # Check a sample
        sample_file = output_files[0]
        sample_ep = np.load(sample_file, allow_pickle=True).item()
        print(f"\nSample episode ({sample_file.name}):")
        print(f"  action shape: {sample_ep['action'].shape}")
        print(f"  image shape: {sample_ep['observation']['image'].shape}")
        print(f"  state shape: {sample_ep['observation']['state'].shape}")
        print(f"  joint_state shape: {sample_ep['observation']['joint_state'].shape}")
        print(f"  global_indices shape: {sample_ep['global_indices'].shape}")
        print(f"  global_indices (first 10): {sample_ep['global_indices'][:10]}")
        print(f"  language_instruction: {sample_ep['language_instruction']}")
        print(f"  episode_metadata: {sample_ep['episode_metadata']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create supervised episode .npy files from teacher_dataset_supervised.h5"
    )
    args = parser.parse_args()

    main()
