"""
inspect_lapa_hidden_states.py

Read the extracted LAPA hidden states HDF5 file and display hidden state shapes
for a given episode.

Usage:
python inspect_lapa_hidden_states.py \
    --h5_file /path/to/lapa_hidden_states.h5 \
    --episode_idx 0
"""

import argparse
from pathlib import Path
import numpy as np

try:
    import h5py
except ImportError:
    print("❌ h5py not installed. Install with: pip install h5py")
    exit(1)


def inspect_episode(h5_file, episode_idx):
    """
    Read HDF5 file and display hidden state shapes for a specific episode.

    Args:
        h5_file: Path to lapa_hidden_states.h5
        episode_idx: Episode index to inspect
    """
    h5_file = Path(h5_file)

    if not h5_file.exists():
        print(f"❌ File not found: {h5_file}")
        return

    print(f"\n{'='*70}")
    print(f"Reading: {h5_file}")
    print(f"{'='*70}")

    with h5py.File(h5_file, 'r') as f:
        print(f"\n📊 Dataset Info:")
        print(f"  - Total samples: {len(f['hidden_states'])}")
        print(f"  - Keys: {list(f.keys())}")

        # Find all samples for this episode
        episode_indices = f['episode_indices'][:]
        unique_episodes = np.unique(episode_indices)
        print(f"  - Episodes with data: {len(unique_episodes)}")
        print(f"  - Episode range: {unique_episodes.min()} to {unique_episodes.max()}")
        print(f"  - Missing episodes: {432 - len(unique_episodes)} (likely failed or had no video)")

        matching_indices = np.where(episode_indices == episode_idx)[0]

        if len(matching_indices) == 0:
            print(f"\n❌ No samples found for episode {episode_idx}")
            print(f"   Available episodes: {np.unique(episode_indices).tolist()}")
            return

        print(f"\n✅ Found {len(matching_indices)} samples for episode {episode_idx}")
        print(f"\n{'='*70}")
        print(f"Episode {episode_idx} - Hidden State Shapes")
        print(f"{'='*70}\n")

        # Print details for each sample in this episode
        print(f"{'Index':<8} {'Frame':<8} {'Seq Len':<10} {'Shape':<20} {'Task'}")
        print(f"{'-'*70}")

        for sample_idx in matching_indices:
            # Read metadata
            frame_idx = f['frame_indices'][sample_idx]
            seq_len = f['seq_lengths'][sample_idx]
            task_desc = f['task_descriptions'][sample_idx]

            # Reconstruct 2D hidden state
            hidden_state_flat = f['hidden_states'][sample_idx]
            hidden_state_2d = hidden_state_flat.reshape(seq_len, 4096)

            # Decode task description if bytes
            if isinstance(task_desc, bytes):
                task_desc = task_desc.decode('utf-8')

            # Truncate task description for display
            task_desc_short = task_desc[:30] + "..." if len(task_desc) > 30 else task_desc

            shape_str = str(hidden_state_2d.shape)
            print(f"{sample_idx:<8} {frame_idx:<8} {seq_len:<10} {shape_str:<20} {task_desc_short}")

        # Statistics
        print(f"\n{'='*70}")
        print(f"Statistics for Episode {episode_idx}:")
        print(f"{'='*70}")

        seq_lens = f['seq_lengths'][matching_indices]
        print(f"  - Sequence length range: {seq_lens.min()} to {seq_lens.max()}")
        print(f"  - Average sequence length: {seq_lens.mean():.1f}")
        print(f"  - Hidden dimension: 4096")

        # Concatenate all hidden states for this episode and show stats
        all_hidden_states = []
        for sample_idx in matching_indices:
            hidden_state_flat = f['hidden_states'][sample_idx]
            seq_len = f['seq_lengths'][sample_idx]
            hidden_state_2d = hidden_state_flat.reshape(seq_len, 4096)
            all_hidden_states.append(hidden_state_2d)

        all_hidden_states_concat = np.vstack(all_hidden_states)
        print(f"\n  - Total vectors in episode: {len(all_hidden_states_concat)}")
        print(f"  - Min value: {all_hidden_states_concat.min():.6f}")
        print(f"  - Max value: {all_hidden_states_concat.max():.6f}")
        print(f"  - Mean value: {all_hidden_states_concat.mean():.6f}")
        print(f"  - Std value: {all_hidden_states_concat.std():.6f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect LAPA hidden states HDF5 file"
    )
    parser.add_argument(
        "--h5_file",
        type=str,
        required=True,
        help="Path to lapa_hidden_states.h5"
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=0,
        help="Episode index to inspect (default: 0)"
    )

    args = parser.parse_args()
    inspect_episode(args.h5_file, args.episode_idx)
