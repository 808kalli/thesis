"""
Compare Hidden State Distributions Between Vanilla and Distilled Models

This script loads hidden states extracted from vanilla and distilled models,
matches them by global index, and compares their distributions.

Usage:
    python scripts/compare_hidden_state_distributions.py \
        --vanilla_dir runs/hidden_states_vanilla \
        --distilled_dir runs/hidden_states_distilled \
        --output_dir analysis/distribution_comparison
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance


def load_hidden_states(directory):
    """
    Load all hidden states from a directory.

    Returns:
        dict: global_index -> hidden_states array
    """
    directory = Path(directory)
    hidden_states_by_idx = {}

    batch_files = sorted(directory.glob("batch_*.npz"))
    print(f"Loading {len(batch_files)} batches from {directory}...")

    for batch_file in batch_files:
        data = np.load(batch_file)
        global_indices = data["global_indices"]
        student_hidden = data["student_hidden"]  # [batch_size, hidden_dim] (already aggregated)

        # Store by global index
        for i, global_idx in enumerate(global_indices):
            hidden_states_by_idx[int(global_idx)] = student_hidden[i]

    print(f"  Loaded {len(hidden_states_by_idx)} samples")
    return hidden_states_by_idx


def match_hidden_states(vanilla_states, distilled_states):
    """
    Match hidden states by global index.

    Returns:
        vanilla_matched: list of arrays
        distilled_matched: list of arrays
        matched_indices: list of global indices
    """
    common_indices = set(vanilla_states.keys()) & set(distilled_states.keys())
    common_indices = sorted(common_indices)

    print(f"\nMatching samples:")
    print(f"  Vanilla samples: {len(vanilla_states)}")
    print(f"  Distilled samples: {len(distilled_states)}")
    print(f"  Common samples: {len(common_indices)}")

    vanilla_matched = [vanilla_states[idx] for idx in common_indices]
    distilled_matched = [distilled_states[idx] for idx in common_indices]

    return vanilla_matched, distilled_matched, common_indices


def aggregate_hidden_states(hidden_states_list, method="mean"):
    """
    Stack hidden states (already aggregated during extraction).

    Args:
        hidden_states_list: List of [hidden_dim] arrays (already aggregated)
        method: Ignored (kept for compatibility)

    Returns:
        np.array: [num_samples, hidden_dim]
    """
    # Hidden states are already aggregated, just stack them
    return np.stack(hidden_states_list)


def compute_statistics(vanilla_agg, distilled_agg):
    """Compute distribution statistics."""
    stats = {}

    # Per-dimension statistics
    stats["vanilla_mean"] = vanilla_agg.mean(axis=0)
    stats["vanilla_std"] = vanilla_agg.std(axis=0)
    stats["distilled_mean"] = distilled_agg.mean(axis=0)
    stats["distilled_std"] = distilled_agg.std(axis=0)

    # Norms
    stats["vanilla_norms"] = np.linalg.norm(vanilla_agg, axis=1)
    stats["distilled_norms"] = np.linalg.norm(distilled_agg, axis=1)

    # Pairwise cosine similarities (sample to sample)
    stats["cosine_similarities"] = []
    for v, d in zip(vanilla_agg, distilled_agg):
        cos_sim = 1 - cosine(v, d)
        stats["cosine_similarities"].append(cos_sim)
    stats["cosine_similarities"] = np.array(stats["cosine_similarities"])

    # L2 distances (sample to sample)
    stats["l2_distances"] = np.linalg.norm(vanilla_agg - distilled_agg, axis=1)

    return stats


def plot_distributions(stats, output_dir):
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Norm distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(stats["vanilla_norms"], bins=50, alpha=0.6, label="Vanilla", color="blue")
    axes[0].hist(stats["distilled_norms"], bins=50, alpha=0.6, label="Distilled", color="red")
    axes[0].set_xlabel("Hidden State Norm")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Hidden State Norms")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Cosine similarity distribution
    axes[1].hist(stats["cosine_similarities"], bins=50, color="green", alpha=0.7)
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Vanilla vs Distilled Cosine Similarity\n(Mean: {stats['cosine_similarities'].mean():.4f})")
    axes[1].axvline(stats["cosine_similarities"].mean(), color="red", linestyle="--", linewidth=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "norm_and_similarity_distributions.png", dpi=150)
    plt.close()

    print(f"✓ Saved: {output_dir / 'norm_and_similarity_distributions.png'}")

    # Plot 3: L2 distance distribution
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(stats["l2_distances"], bins=50, color="purple", alpha=0.7)
    ax.set_xlabel("L2 Distance")
    ax.set_ylabel("Frequency")
    ax.set_title(f"L2 Distance: Vanilla vs Distilled\n(Mean: {stats['l2_distances'].mean():.4f})")
    ax.axvline(stats["l2_distances"].mean(), color="red", linestyle="--", linewidth=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "l2_distance_distribution.png", dpi=150)
    plt.close()

    print(f"✓ Saved: {output_dir / 'l2_distance_distribution.png'}")

    # Plot 4: Mean per-dimension comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    dims = np.arange(len(stats["vanilla_mean"]))
    ax.plot(dims, stats["vanilla_mean"], label="Vanilla Mean", alpha=0.7, linewidth=1)
    ax.plot(dims, stats["distilled_mean"], label="Distilled Mean", alpha=0.7, linewidth=1)
    ax.fill_between(dims, stats["vanilla_mean"] - stats["vanilla_std"],
                     stats["vanilla_mean"] + stats["vanilla_std"], alpha=0.2)
    ax.fill_between(dims, stats["distilled_mean"] - stats["distilled_std"],
                     stats["distilled_mean"] + stats["distilled_std"], alpha=0.2)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Value")
    ax.set_title("Per-Dimension Mean ± Std")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "per_dimension_mean_std.png", dpi=150)
    plt.close()

    print(f"✓ Saved: {output_dir / 'per_dimension_mean_std.png'}")


def print_summary(stats):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("DISTRIBUTION COMPARISON SUMMARY")
    print("=" * 80)

    print("\nNorm Statistics:")
    print(f"  Vanilla  - Mean: {stats['vanilla_norms'].mean():.4f}, Std: {stats['vanilla_norms'].std():.4f}")
    print(f"  Distilled - Mean: {stats['distilled_norms'].mean():.4f}, Std: {stats['distilled_norms'].std():.4f}")

    print("\nCosine Similarity (Vanilla vs Distilled):")
    print(f"  Mean: {stats['cosine_similarities'].mean():.4f}")
    print(f"  Std:  {stats['cosine_similarities'].std():.4f}")
    print(f"  Min:  {stats['cosine_similarities'].min():.4f}")
    print(f"  Max:  {stats['cosine_similarities'].max():.4f}")

    print("\nL2 Distance (Vanilla vs Distilled):")
    print(f"  Mean: {stats['l2_distances'].mean():.4f}")
    print(f"  Std:  {stats['l2_distances'].std():.4f}")
    print(f"  Min:  {stats['l2_distances'].min():.4f}")
    print(f"  Max:  {stats['l2_distances'].max():.4f}")

    print("\nInterpretation:")
    avg_cos_sim = stats['cosine_similarities'].mean()
    if avg_cos_sim > 0.95:
        print("  → Very high similarity: Distributions are nearly identical")
    elif avg_cos_sim > 0.85:
        print("  → High similarity: Distillation has minor effect on distribution")
    elif avg_cos_sim > 0.70:
        print("  → Moderate similarity: Distillation has noticeable effect")
    else:
        print("  → Low similarity: Distillation significantly changes distribution")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare hidden state distributions")
    parser.add_argument("--vanilla_dir", type=str, required=True,
                        help="Directory with vanilla model hidden states")
    parser.add_argument("--distilled_dir", type=str, required=True,
                        help="Directory with distilled model hidden states")
    parser.add_argument("--output_dir", type=str, default="analysis/distribution_comparison",
                        help="Output directory for plots and results")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "last"],
                        help="Method to aggregate sequence dimension")

    args = parser.parse_args()

    # Load hidden states
    print("=" * 80)
    print("Loading Hidden States")
    print("=" * 80)
    vanilla_states = load_hidden_states(args.vanilla_dir)
    distilled_states = load_hidden_states(args.distilled_dir)

    # Match by global index
    vanilla_matched, distilled_matched, matched_indices = match_hidden_states(
        vanilla_states, distilled_states
    )

    if len(matched_indices) == 0:
        print("ERROR: No common samples found!")
        return

    # Aggregate sequences
    print(f"\nAggregating sequences using '{args.aggregation}' method...")
    vanilla_agg = aggregate_hidden_states(vanilla_matched, method=args.aggregation)
    distilled_agg = aggregate_hidden_states(distilled_matched, method=args.aggregation)

    print(f"  Vanilla shape:   {vanilla_agg.shape}")
    print(f"  Distilled shape: {distilled_agg.shape}")

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(vanilla_agg, distilled_agg)

    # Print summary
    print_summary(stats)

    # Create plots
    print("\nGenerating plots...")
    plot_distributions(stats, args.output_dir)

    # Save numerical results
    output_dir = Path(args.output_dir)
    np.savez(
        output_dir / "statistics.npz",
        vanilla_norms=stats["vanilla_norms"],
        distilled_norms=stats["distilled_norms"],
        cosine_similarities=stats["cosine_similarities"],
        l2_distances=stats["l2_distances"],
        matched_indices=matched_indices,
    )
    print(f"✓ Saved: {output_dir / 'statistics.npz'}")

    print("\n" + "=" * 80)
    print("✓ Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
