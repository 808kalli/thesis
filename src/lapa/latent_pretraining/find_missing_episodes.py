"""
find_missing_episodes.py

Identify which episodes are missing from the extraction and how many samples each has.
"""

import h5py
from pathlib import Path
import numpy as np
import argparse


def analyze_episodes(h5_file):
    """Analyze which episodes have data and show sample counts."""
    h5_file = Path(h5_file)

    if not h5_file.exists():
        print(f"❌ File not found: {h5_file}")
        return

    with h5py.File(h5_file, 'r') as f:
        episode_indices = f['episode_indices'][:]

        unique_episodes, counts = np.unique(episode_indices, return_counts=True)

        print(f"\n{'='*70}")
        print(f"Episode Extraction Summary")
        print(f"{'='*70}")
        print(f"\nTotal episodes with data: {len(unique_episodes)}/432")
        print(f"Total samples: {len(episode_indices)}")
        print(f"Average samples per episode: {len(episode_indices) / len(unique_episodes):.1f}")

        print(f"\nSample distribution:")
        print(f"  Min: {counts.min()}")
        print(f"  Max: {counts.max()}")
        print(f"  Mean: {counts.mean():.1f}")
        print(f"  Std: {counts.std():.1f}")

        # Find missing episodes
        all_episodes = set(range(432))
        extracted_episodes = set(unique_episodes.tolist())
        missing_episodes = sorted(all_episodes - extracted_episodes)

        print(f"\nMissing episodes: {len(missing_episodes)}")
        if len(missing_episodes) <= 50:
            print(f"  {missing_episodes}")
        else:
            print(f"  First 20: {missing_episodes[:20]}")
            print(f"  Last 20: {missing_episodes[-20:]}")

        # Find where the cutoff is
        if len(missing_episodes) > 0:
            cutoff = min(missing_episodes)
            print(f"\nCutoff point: Episode {cutoff} is the first missing episode")
            print(f"Episodes 0-{cutoff-1} were processed ({cutoff} episodes)")
            print(f"Episodes {cutoff}-431 are missing ({432-cutoff} episodes)")

        # Show episodes with few samples
        print(f"\n{'='*70}")
        print(f"Episodes with fewer than 5 samples:")
        print(f"{'='*70}")

        for ep, count in zip(unique_episodes, counts):
            if count < 5:
                print(f"  Episode {ep}: {count} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze episode extraction")
    parser.add_argument("--h5_file", type=str, required=True, help="Path to lapa_hidden_states.h5")

    args = parser.parse_args()
    analyze_episodes(args.h5_file)
