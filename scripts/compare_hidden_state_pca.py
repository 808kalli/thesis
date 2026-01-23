"""
Compare LAPA Mean, Distilled, and Vanilla hidden states using various methods.

This script:
1. Loads hidden states from all 3 sources
2. Aligns samples using global_indices (intersection of available indices)
3. Performs various analyses:
   - PCA 2D scatter plots (overlay)
   - PCA component histograms (1D distributions)
   - L2 norm distributions

Usage:
    python compare_hidden_state_pca.py --plot pca_2d
    python compare_hidden_state_pca.py --plot pca_hist
    python compare_hidden_state_pca.py --plot l2_norm
    python compare_hidden_state_pca.py --plot all
"""

import numpy as np
import h5py
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse

# Paths
LAPA_H5 = Path("/home/elias/Thesis/lapa_hidden_states/lapa_mean_hidden_states_eps0-431_stride1.h5")
DISTILLED_DIR = Path("/home/elias/Thesis/lapa_hidden_states/distilled")
VANILLA_DIR = Path("/home/elias/Thesis/lapa_hidden_states/vanilla")
BEFORE_TRAINING_DIR = Path("/home/elias/Thesis/lapa_hidden_states/before_training")
OUTPUT_DIR = Path("/home/elias/Thesis/figures")


def load_lapa_mean():
    """Load LAPA mean hidden states from H5."""
    print("Loading LAPA mean hidden states...")
    with h5py.File(LAPA_H5, 'r') as f:
        hidden_states = f['hidden_states'][:]
        global_indices = f['global_indices'][:]

    # Create mapping from global_index -> hidden_state
    index_to_hidden = {int(idx): hidden_states[i] for i, idx in enumerate(global_indices)}
    print(f"  Loaded {len(index_to_hidden)} samples")
    return index_to_hidden


def load_npz_batches(batch_dir):
    """Load hidden states from NPZ batch files."""
    print(f"Loading batches from {batch_dir.name}...")
    batch_files = sorted(glob.glob(str(batch_dir / "batch_*.npz")))

    index_to_hidden = {}

    for batch_file in tqdm(batch_files, desc="  Loading batches"):
        batch = np.load(batch_file)
        student_hidden = batch['student_hidden']
        global_indices = batch['global_indices']

        for i, idx in enumerate(global_indices):
            idx = int(idx)
            # Keep first occurrence if duplicate
            if idx not in index_to_hidden:
                index_to_hidden[idx] = student_hidden[i]

    print(f"  Loaded {len(index_to_hidden)} unique samples")
    return index_to_hidden


def align_samples(lapa_dict, distilled_dict, vanilla_dict, before_training_dict=None):
    """Align samples to have the same global indices across all sources."""
    print("\nAligning samples...")

    # Find common indices
    lapa_indices = set(lapa_dict.keys())
    distilled_indices = set(distilled_dict.keys())
    vanilla_indices = set(vanilla_dict.keys())

    common_indices = lapa_indices & distilled_indices & vanilla_indices
    print(f"  LAPA indices: {len(lapa_indices)}")
    print(f"  Distilled indices: {len(distilled_indices)}")
    print(f"  Vanilla indices: {len(vanilla_indices)}")

    if before_training_dict is not None:
        before_training_indices = set(before_training_dict.keys())
        common_indices = common_indices & before_training_indices
        print(f"  Before Training indices: {len(before_training_indices)}")

    print(f"  Common indices: {len(common_indices)}")

    # Sort indices for consistent ordering
    common_indices = sorted(common_indices)

    # Extract aligned arrays
    lapa_hidden = np.array([lapa_dict[idx] for idx in common_indices])
    distilled_hidden = np.array([distilled_dict[idx] for idx in common_indices])
    vanilla_hidden = np.array([vanilla_dict[idx] for idx in common_indices])

    if before_training_dict is not None:
        before_training_hidden = np.array([before_training_dict[idx] for idx in common_indices])
        return lapa_hidden, distilled_hidden, vanilla_hidden, before_training_hidden, common_indices

    return lapa_hidden, distilled_hidden, vanilla_hidden, common_indices


def plot_pca_2d(lapa_hidden, distilled_hidden, vanilla_hidden, output_dir, before_training_hidden=None):
    """Plot 2D PCA scatter comparison (overlay only)."""
    print("\nPerforming 2D PCA scatter plot...")

    # Combine all data for fitting PCA
    datasets = [lapa_hidden, distilled_hidden, vanilla_hidden]
    if before_training_hidden is not None:
        datasets.append(before_training_hidden)
    all_data = np.vstack(datasets)

    pca = PCA(n_components=2)
    pca.fit(all_data)

    # Transform each dataset
    lapa_pca = pca.transform(lapa_hidden)
    distilled_pca = pca.transform(distilled_hidden)
    vanilla_pca = pca.transform(vanilla_hidden)
    if before_training_hidden is not None:
        before_training_pca = pca.transform(before_training_hidden)

    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")

    # Single overlay plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(lapa_pca[:, 0], lapa_pca[:, 1], alpha=0.2, s=1, c='blue', label='LAPA Mean')
    ax.scatter(distilled_pca[:, 0], distilled_pca[:, 1], alpha=0.2, s=1, c='red', label='Distilled')
    ax.scatter(vanilla_pca[:, 0], vanilla_pca[:, 1], alpha=0.2, s=1, c='green', label='Vanilla')
    if before_training_hidden is not None:
        ax.scatter(before_training_pca[:, 0], before_training_pca[:, 1], alpha=0.2, s=1, c='orange', label='Before Training')
    ax.set_title('Hidden States PCA Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "pca_2d_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_pca_histograms(lapa_hidden, distilled_hidden, vanilla_hidden, output_dir):
    """Plot PCA component histograms (1D distributions of PC1)."""
    from scipy.stats import gaussian_kde

    print("\nPlotting PC1 histogram...")

    # Combine all data for fitting PCA
    all_data = np.vstack([lapa_hidden, distilled_hidden, vanilla_hidden])

    pca = PCA(n_components=1)
    pca.fit(all_data)

    # Transform each dataset
    lapa_pca = pca.transform(lapa_hidden).flatten()
    distilled_pca = pca.transform(distilled_hidden).flatten()
    vanilla_pca = pca.transform(vanilla_hidden).flatten()

    print(f"  PC1 explained variance: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  Teacher (LAPA) - mean: {lapa_pca.mean():.2f}, std: {lapa_pca.std():.2f}")
    print(f"  Student (Distilled) - mean: {distilled_pca.mean():.2f}, std: {distilled_pca.std():.2f}")
    print(f"  Vanilla (Initial) - mean: {vanilla_pca.mean():.2f}, std: {vanilla_pca.std():.2f}")

    # Create x range for KDE
    x_min = min(vanilla_pca.min(), lapa_pca.min(), distilled_pca.min())
    x_max = max(vanilla_pca.max(), lapa_pca.max(), distilled_pca.max())
    x = np.linspace(x_min, x_max, 500)

    # Compute KDEs
    kde_vanilla = gaussian_kde(vanilla_pca)
    kde_lapa = gaussian_kde(lapa_pca)
    kde_distilled = gaussian_kde(distilled_pca)

    y_vanilla = kde_vanilla(x)
    y_lapa = kde_lapa(x)
    y_distilled = kde_distilled(x)

    # === Plot 1: KDE curves (true density) ===
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, y_vanilla, color='gray', linewidth=2,
            label=f'Vanilla (μ={vanilla_pca.mean():.1f}, σ={vanilla_pca.std():.1f})')
    ax.fill_between(x, y_vanilla, alpha=0.3, color='gray')

    ax.plot(x, y_lapa, color='blue', linewidth=2,
            label=f'Teacher (μ={lapa_pca.mean():.1f}, σ={lapa_pca.std():.1f})')
    ax.fill_between(x, y_lapa, alpha=0.3, color='blue')

    ax.plot(x, y_distilled, color='red', linewidth=2,
            label=f'Student (μ={distilled_pca.mean():.1f}, σ={distilled_pca.std():.1f})')
    ax.fill_between(x, y_distilled, alpha=0.3, color='red')

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "pca_histograms_kde.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # === Plot 2: Normalized to same max height ===
    # Normalize each KDE to have max=1
    y_vanilla_norm = y_vanilla / y_vanilla.max()
    y_lapa_norm = y_lapa / y_lapa.max()
    y_distilled_norm = y_distilled / y_distilled.max()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Colors matching bar plot style: darkslategray for Student, tan for Vanilla
    ax.fill_between(x, y_vanilla_norm, alpha=0.6, color='tan',
                    label=f'Vanilla (μ={vanilla_pca.mean():.1f}, σ={vanilla_pca.std():.1f})')

    ax.fill_between(x, y_lapa_norm, alpha=0.6, color='red',
                    label=f'Teacher (μ={lapa_pca.mean():.1f}, σ={lapa_pca.std():.1f})')

    ax.fill_between(x, y_distilled_norm, alpha=0.6, color='darkslategray',
                    label=f'Student (μ={distilled_pca.mean():.1f}, σ={distilled_pca.std():.1f})')

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('Normalized Density', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "pca_histograms_normalized.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_l2_norm(lapa_hidden, distilled_hidden, vanilla_hidden, output_dir):
    """Plot L2 norm distributions."""
    print("\nPlotting L2 norm distributions...")

    # Compute L2 norms
    lapa_l2 = np.linalg.norm(lapa_hidden, axis=1)
    distilled_l2 = np.linalg.norm(distilled_hidden, axis=1)
    vanilla_l2 = np.linalg.norm(vanilla_hidden, axis=1)

    print(f"  LAPA L2 norm - mean: {lapa_l2.mean():.2f}, std: {lapa_l2.std():.2f}")
    print(f"  Distilled L2 norm - mean: {distilled_l2.mean():.2f}, std: {distilled_l2.std():.2f}")
    print(f"  Vanilla L2 norm - mean: {vanilla_l2.mean():.2f}, std: {vanilla_l2.std():.2f}")

    # Single overlay plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lapa_l2, bins=100, alpha=0.6, color='blue', label=f'LAPA Mean (μ={lapa_l2.mean():.1f})', density=True)
    ax.hist(distilled_l2, bins=100, alpha=0.6, color='red', label=f'Distilled (μ={distilled_l2.mean():.1f})', density=True)
    ax.hist(vanilla_l2, bins=100, alpha=0.6, color='green', label=f'Vanilla (μ={vanilla_l2.mean():.1f})', density=True)
    ax.set_title('L2 Norm Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('L2 Norm')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "l2_norm_distribution.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare hidden state distributions")
    parser.add_argument('--plot', type=str, default='all',
                        choices=['pca_2d', 'pca_hist', 'l2_norm', 'all'],
                        help='Which plot(s) to generate')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COMPARING HIDDEN STATE DISTRIBUTIONS")
    print("="*80)

    # Load all sources
    lapa_dict = load_lapa_mean()
    distilled_dict = load_npz_batches(DISTILLED_DIR)
    vanilla_dict = load_npz_batches(VANILLA_DIR)
    before_training_dict = load_npz_batches(BEFORE_TRAINING_DIR)

    # Align to common indices
    lapa_hidden, distilled_hidden, vanilla_hidden, before_training_hidden, common_indices = align_samples(
        lapa_dict, distilled_dict, vanilla_dict, before_training_dict
    )

    print(f"\nFinal aligned dataset:")
    print(f"  Shape: {lapa_hidden.shape}")
    print(f"  Total samples: {len(common_indices)}")

    # Create visualizations based on flag
    plots_generated = []

    if args.plot in ['pca_2d', 'all']:
        plot_pca_2d(lapa_hidden, distilled_hidden, vanilla_hidden, OUTPUT_DIR, before_training_hidden)
        plots_generated.append("pca_2d_comparison.png")

    if args.plot in ['pca_hist', 'all']:
        plot_pca_histograms(lapa_hidden, distilled_hidden, vanilla_hidden, OUTPUT_DIR)
        plots_generated.extend(["pca_histograms_kde.png", "pca_histograms_normalized.png"])

    if args.plot in ['l2_norm', 'all']:
        plot_l2_norm(lapa_hidden, distilled_hidden, vanilla_hidden, OUTPUT_DIR)
        plots_generated.append("l2_norm_distribution.png")

    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Generated plots:")
    for p in plots_generated:
        print(f"  - {p}")
    print("="*80)


if __name__ == "__main__":
    main()
