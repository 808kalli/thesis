"""
Precompute teacher dataset with aggregated and interpolated hidden states.

MEMORY EFFICIENT: Streams data and only keeps one episode in RAM at a time.

This script:
1. Loads aggregated teacher hidden states from lapa_hidden_states.h5 (teacher frames at 0, 12, 24, ...)
2. Loads the libero_spatial_noops dataset
3. For each episode:
   - Aggregate teacher frame sequences [seq_len, 4096] → [4096]
   - For frames WITHOUT teacher supervision (not multiples of 12):
     - Interpolate between adjacent supervised frames
4. Saves a new HDF5 with [batch, 4096] dense teacher states for ALL frames in each episode

Output: teacher_dataset_interpolated.h5
  - teacher_states: [total_samples, 4096]
  - global_indices: [total_samples] (index into dataset, 0..52969)
  - has_supervision: [total_samples] boolean (True if frame_idx % 12 == 0)

Usage:
  python precompute_teacher_dataset.py
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import tqdm
import argparse
from typing import Dict, Tuple

# Paths
TEACHER_H5 = Path("/home/elias/Thesis/lapa_hidden_states.h5")
DATASET_DIR = Path("/home/elias/Thesis/raw_datasets/libero_spatial_noops")
OUTPUT_FILE = Path("/home/elias/Thesis/teacher_dataset_interpolated.h5")
OUTPUT_FILE_SUPERVISED = Path("/home/elias/Thesis/teacher_dataset_supervised.h5")

TEACHER_STRIDE = 12  # Frames are sampled every 12 (0, 12, 24, ...)
HIDDEN_DIM = 4096
AGGREGATION_METHOD = "mean"  # Options: "last" or "mean"


def aggregate_sequence(seq: np.ndarray, method: str = "last") -> np.ndarray:
    """Aggregate [seq_len, 4096] → [4096]"""
    if method == "last":
        return seq[-1, :].astype(np.float32)  # Last token
    elif method == "mean":
        return seq.mean(axis=0).astype(np.float32)  # Mean of all tokens
    else:
        raise ValueError(f"Unknown aggregation: {method}")


def get_actual_episode_lengths() -> Dict[int, int]:
    """
    Get actual number of frames per episode from dataset parquet files.

    This is needed because teacher frames might not extend to the end of an episode.
    e.g., if last teacher frame is 240, but episode has 250 frames, we need to know that.
    """
    print(f"Scanning dataset for actual episode lengths...")

    data_dir = DATASET_DIR / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))

    episode_lengths = {}

    for pf in tqdm.tqdm(parquet_files, desc="Scanning parquets"):
        df = pd.read_parquet(pf, columns=["episode_index"])
        for episode_idx in df["episode_index"].unique():
            count = len(df[df["episode_index"] == episode_idx])
            if episode_idx not in episode_lengths:
                episode_lengths[episode_idx] = count

    print(f"✓ Found {len(episode_lengths)} episodes")
    return episode_lengths


def build_global_index_map() -> Dict[Tuple[int, int], int]:
    """
    Build mapping from (episode_idx, frame_idx) to global sequential index.

    The dataset has a global 'index' column that's sequential 0..52969.
    This allows us to look up teacher states by the global sample index
    instead of episode/frame pairs.
    """
    print(f"Building global index map...")

    data_dir = DATASET_DIR / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))

    index_map = {}

    for pf in tqdm.tqdm(parquet_files, desc="Building index map"):
        df = pd.read_parquet(pf, columns=["index", "episode_index", "frame_index"])
        for _, row in df.iterrows():
            key = (int(row["episode_index"]), int(row["frame_index"]))
            global_idx = int(row["index"])
            index_map[key] = global_idx

    print(f"✓ Built index map with {len(index_map)} entries")
    return index_map


def interpolate_teacher_state(
    teacher_states: Dict[int, np.ndarray],
    frame_idx: int,
    teacher_stride: int = 12
) -> Tuple[np.ndarray, bool]:
    """Get teacher state for a frame, interpolating if necessary."""
    if frame_idx in teacher_states:
        return teacher_states[frame_idx], True

    lower_frame = (frame_idx // teacher_stride) * teacher_stride
    upper_frame = lower_frame + teacher_stride

    has_lower = lower_frame in teacher_states
    has_upper = upper_frame in teacher_states

    if has_lower and has_upper:
        ratio = (frame_idx - lower_frame) / teacher_stride
        state_lower = teacher_states[lower_frame]
        state_upper = teacher_states[upper_frame]
        interpolated = ((1 - ratio) * state_lower + ratio * state_upper).astype(np.float32)
        return interpolated, False
    elif has_lower:
        return teacher_states[lower_frame], False
    elif has_upper:
        return teacher_states[upper_frame], False
    else:
        return np.zeros(HIDDEN_DIM, dtype=np.float32), False


def main():
    """Precompute teacher dataset - MEMORY EFFICIENT VERSION."""

    if not TEACHER_H5.exists():
        raise FileNotFoundError(f"Teacher H5 not found: {TEACHER_H5}")

    print(f"Opening teacher H5: {TEACHER_H5}")
    print(f"Creating output file: {OUTPUT_FILE}\n")

    # Get actual episode lengths from dataset
    episode_lengths = get_actual_episode_lengths()

    # Build mapping from (episode, frame) to global sequential index
    global_index_map = build_global_index_map()

    total_samples = 0
    supervised_count = 0
    interpolated_count = 0

    with h5py.File(TEACHER_H5, "r") as teacher_f, \
         h5py.File(OUTPUT_FILE, "w") as out_f:

        # Create output datasets
        out_f.create_dataset("teacher_states", (0, HIDDEN_DIM), maxshape=(None, HIDDEN_DIM),
                            dtype=np.float32, chunks=(1000, HIDDEN_DIM), compression="gzip")
        out_f.create_dataset("global_indices", (0,), maxshape=(None,),
                            dtype=np.int32, chunks=10000, compression="gzip")
        out_f.create_dataset("has_supervision", (0,), maxshape=(None,),
                            dtype=bool, chunks=10000, compression="gzip")

        # Read teacher metadata
        teacher_episodes = teacher_f["episode_indices"][:]
        teacher_frames = teacher_f["frame_indices"][:]
        teacher_seq_lengths = teacher_f["seq_lengths"][:]
        teacher_hidden_states = teacher_f["hidden_states"]

        # Index teacher states by episode
        print("\nIndexing teacher states by episode...")
        episode_to_indices = {}
        for idx in range(len(teacher_episodes)):
            ep = int(teacher_episodes[idx])
            if ep not in episode_to_indices:
                episode_to_indices[ep] = []
            episode_to_indices[ep].append(idx)

        print(f"✓ Indexed {len(episode_to_indices)} episodes")

        # Process each episode (keeps only one episode in RAM at a time)
        print("\nProcessing episodes...")
        for episode_idx in tqdm.tqdm(sorted(episode_lengths.keys()), desc="Episodes"):
            num_frames = episode_lengths[episode_idx]

            # Load and aggregate ONLY this episode's teacher states
            episode_teacher_states = {}

            if episode_idx in episode_to_indices:
                for sample_idx in episode_to_indices[episode_idx]:
                    frame_idx = int(teacher_frames[sample_idx])
                    seq_len = int(teacher_seq_lengths[sample_idx])

                    # Read and reshape
                    hidden_state_flat = np.array(teacher_hidden_states[sample_idx], dtype=np.float32)
                    hidden_state_2d = hidden_state_flat.reshape(seq_len, HIDDEN_DIM)

                    # Aggregate
                    aggregated = aggregate_sequence(hidden_state_2d, AGGREGATION_METHOD)
                    episode_teacher_states[frame_idx] = aggregated

            if not episode_teacher_states:
                continue

            # Collect all frames for this episode
            episode_states_list = []
            global_indices_list = []
            supervision_list = []

            for frame_idx in range(num_frames):
                state, has_supervision = interpolate_teacher_state(
                    episode_teacher_states, frame_idx, TEACHER_STRIDE
                )

                # Look up global index for this (episode, frame) pair
                key = (episode_idx, frame_idx)
                if key in global_index_map:
                    global_idx = global_index_map[key]
                else:
                    # If not found, use -1 as sentinel (shouldn't happen)
                    global_idx = -1

                episode_states_list.append(state)
                global_indices_list.append(global_idx)
                supervision_list.append(has_supervision)

                if has_supervision:
                    supervised_count += 1
                else:
                    interpolated_count += 1
                total_samples += 1

            # Write entire episode to disk at once
            if episode_states_list:
                current_size = len(out_f["teacher_states"])
                num_frames_to_write = len(episode_states_list)

                out_f["teacher_states"].resize(current_size + num_frames_to_write, axis=0)
                out_f["global_indices"].resize(current_size + num_frames_to_write, axis=0)
                out_f["has_supervision"].resize(current_size + num_frames_to_write, axis=0)

                out_f["teacher_states"][current_size:current_size+num_frames_to_write] = np.array(episode_states_list, dtype=np.float32)
                out_f["global_indices"][current_size:current_size+num_frames_to_write] = np.array(global_indices_list, dtype=np.int32)
                out_f["has_supervision"][current_size:current_size+num_frames_to_write] = np.array(supervision_list, dtype=bool)

            # Clear this episode's data before moving to next
            episode_teacher_states.clear()

        print(f"\n✓ Created {total_samples} total samples")
        print(f"  - {supervised_count} directly supervised (frame_idx % 12 == 0)")
        print(f"  - {interpolated_count} interpolated")
        print(f"  Output: {OUTPUT_FILE}")

    # Create supervised-only dataset
    create_supervised_only_dataset(OUTPUT_FILE, OUTPUT_FILE_SUPERVISED)


def create_supervised_only_dataset(interpolated_file: Path, output_file: Path):
    """
    Create a new H5 file containing only the supervised frames (directly sampled from teacher).
    These are the frames with has_supervision=True, corresponding to the original LAPA frames.
    """
    print(f"\nCreating supervised-only dataset...")
    print(f"  Reading from: {interpolated_file}")
    print(f"  Writing to: {output_file}")

    with h5py.File(interpolated_file, "r") as in_f, \
         h5py.File(output_file, "w") as out_f:

        # Read all data
        teacher_states = in_f["teacher_states"][:]
        global_indices = in_f["global_indices"][:]
        has_supervision = in_f["has_supervision"][:]

        # Filter to only supervised frames
        supervised_mask = has_supervision
        supervised_states = teacher_states[supervised_mask]
        supervised_global_indices = global_indices[supervised_mask]

        # Create output datasets
        out_f.create_dataset("teacher_states", data=supervised_states.astype(np.float32),
                            dtype=np.float32, compression="gzip")
        out_f.create_dataset("global_indices", data=supervised_global_indices.astype(np.int32),
                            dtype=np.int32, compression="gzip")

        print(f"\n✓ Created supervised-only dataset")
        print(f"  Total samples: {len(supervised_states)}")
        print(f"  Shape: {supervised_states.shape}")
        print(f"  Global indices range: {supervised_global_indices.min()} to {supervised_global_indices.max()}")
        print(f"  Output: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute teacher dataset with aggregation and interpolation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interpolated", "supervised", "both"],
        default="both",
        help="Which dataset to create: 'interpolated' (all frames), 'supervised' (original frames only), or 'both' (default)"
    )
    args = parser.parse_args()

    if args.mode in ["interpolated", "both"]:
        main()

    if args.mode in ["supervised", "both"]:
        create_supervised_only_dataset(OUTPUT_FILE, OUTPUT_FILE_SUPERVISED)
