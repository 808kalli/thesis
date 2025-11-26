"""
Precompute supervised teacher dataset with aggregated hidden states.

MEMORY EFFICIENT: Streams data and only keeps one episode in RAM at a time.

This script:
1. Loads aggregated teacher hidden states from lapa_hidden_states.h5 (teacher frames at 0, 12, 24, ...)
2. Loads the libero_spatial_noops dataset
3. For each episode:
   - Aggregate teacher frame sequences [seq_len, 4096] → [4096]
   - Keep ONLY frames with teacher supervision (multiples of 12)
4. Saves a new HDF5 with [batch, 4096] supervised teacher states

Output: teacher_dataset_supervised.h5
  - hidden_states: [total_samples, 4096]
  - global_indices: [total_samples] (index into dataset)

Usage:
  python precompute_teacher_dataset.py [--aggregation_method mean|last]
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import tqdm
import argparse
from typing import Dict, Tuple

# Paths
TEACHER_H5 = Path("/home/elias/Thesis/lapa_hidden_states/lapa_hidden_states_seq.h5")
DATASET_DIR = Path("/home/elias/Thesis/raw_datasets/libero_spatial_noops")
OUTPUT_FILE_SUPERVISED = Path("/home/elias/Thesis/lapa_hidden_states/teacher_dataset_supervised_last.h5")

TEACHER_STRIDE = 12  # Frames are sampled every 12 (0, 12, 24, ...)
HIDDEN_DIM = 4096
AGGREGATION_METHOD = "last"  # Options: "last" or "mean"


def aggregate_sequence(seq: np.ndarray, method: str = "last") -> np.ndarray:
    """Aggregate [seq_len, 4096] → [4096]"""
    if method == "last":
        return seq[-1, :].astype(np.float32)  # Last token
    elif method == "mean":
        return seq.mean(axis=0).astype(np.float32)  # Mean of all tokens
    else:
        raise ValueError(f"Unknown aggregation: {method}")


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




def main():
    """Precompute supervised teacher dataset - MEMORY EFFICIENT VERSION."""

    if not TEACHER_H5.exists():
        raise FileNotFoundError(f"Teacher H5 not found: {TEACHER_H5}")

    print(f"Opening teacher H5: {TEACHER_H5}")
    print(f"Creating output file: {OUTPUT_FILE_SUPERVISED}\n")

    # Build mapping from (episode, frame) to global sequential index
    global_index_map = build_global_index_map()

    total_samples = 0

    with h5py.File(TEACHER_H5, "r") as teacher_f, \
         h5py.File(OUTPUT_FILE_SUPERVISED, "w") as out_f:

        # Create output datasets
        out_f.create_dataset("hidden_states", (0, HIDDEN_DIM), maxshape=(None, HIDDEN_DIM),
                            dtype=np.float32, chunks=(1000, HIDDEN_DIM), compression="gzip")
        out_f.create_dataset("global_indices", (0,), maxshape=(None,),
                            dtype=np.int32, chunks=10000, compression="gzip")

        # Read teacher metadata
        teacher_episodes = teacher_f["episode_indices"][:]
        teacher_frames = teacher_f["frame_indices"][:]
        teacher_seq_lengths = teacher_f["seq_lengths"][:]
        teacher_hidden_states = teacher_f["hidden_states"]

        # Process all supervised frames directly (these are already at stride 12)
        print("Processing supervised frames...")
        for idx in tqdm.tqdm(range(len(teacher_episodes)), desc="Frames"):
            episode_idx = int(teacher_episodes[idx])
            frame_idx = int(teacher_frames[idx])
            seq_len = int(teacher_seq_lengths[idx])

            # Read and reshape
            hidden_state_flat = np.array(teacher_hidden_states[idx], dtype=np.float32)
            hidden_state_2d = hidden_state_flat.reshape(seq_len, HIDDEN_DIM)

            # Aggregate
            aggregated = aggregate_sequence(hidden_state_2d, AGGREGATION_METHOD)

            # Look up global index for this (episode, frame) pair
            key = (episode_idx, frame_idx)
            if key in global_index_map:
                global_idx = global_index_map[key]
            else:
                # If not found, use -1 as sentinel (shouldn't happen)
                global_idx = -1

            # Write to disk
            current_size = len(out_f["hidden_states"])
            out_f["hidden_states"].resize(current_size + 1, axis=0)
            out_f["global_indices"].resize(current_size + 1, axis=0)

            out_f["hidden_states"][current_size] = aggregated
            out_f["global_indices"][current_size] = global_idx

            total_samples += 1

        print(f"\n✓ Created {total_samples} supervised samples")
        print(f"  Aggregation method: {AGGREGATION_METHOD}")
        print(f"  Output: {OUTPUT_FILE_SUPERVISED}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute supervised teacher dataset with aggregation"
    )
    parser.add_argument(
        "--aggregation_method",
        type=str,
        choices=["last", "mean"],
        default="mean",
        help="Aggregation method for sequence: 'last' (last token) or 'mean' (mean of all tokens, default)"
    )
    args = parser.parse_args()

    # Update global aggregation method if provided
    if args.aggregation_method:
        globals()["AGGREGATION_METHOD"] = args.aggregation_method

    main()
