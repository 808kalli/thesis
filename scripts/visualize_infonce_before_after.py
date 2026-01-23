"""
Visualize InfoNCE cross-modal similarity matrix between before_training and LAPA.

Creates a figure with 1 plot showing the similarity matrix.

Usage:
    python visualize_infonce_before_after.py [options]

Controls:
    Right Arrow / Space: Next batch
    Left Arrow / Backspace: Previous batch
    q: Quit
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import glob
from tqdm import tqdm
import sys

# Paths
LAPA_H5 = Path("/home/elias/Thesis/lapa_hidden_states/lapa_mean_hidden_states_eps0-431_stride1.h5")
BEFORE_TRAINING_DIR = Path("/home/elias/Thesis/lapa_hidden_states/before_training")


def scan_lapa_indices():
    """Get LAPA indices and build index->position mapping."""
    print("Scanning LAPA H5 file...")
    with h5py.File(LAPA_H5, 'r') as f:
        global_indices = f['global_indices'][:]
    index_to_pos = {int(idx): i for i, idx in enumerate(global_indices)}
    print(f"  Found {len(index_to_pos)} indices")
    return index_to_pos


def scan_npz_folder(folder, name=""):
    """Scan NPZ folder to get indices and their file locations."""
    print(f"Scanning {name}...")
    batch_files = sorted(glob.glob(str(folder / "batch_*.npz")))

    index_to_loc = {}
    for batch_file in tqdm(batch_files, desc="  Scanning"):
        data = np.load(batch_file)
        for i, idx in enumerate(data['global_indices']):
            idx = int(idx)
            if idx not in index_to_loc:
                index_to_loc[idx] = (batch_file, i)

    print(f"  Found {len(index_to_loc)} unique indices")
    return index_to_loc


def load_lapa_batch(indices, index_to_pos):
    """Load hidden states from LAPA H5 for given indices."""
    positions = [index_to_pos[idx] for idx in indices]
    with h5py.File(LAPA_H5, 'r') as f:
        hidden = f['hidden_states'][positions]
    return hidden


def load_npz_batch(indices, index_to_loc):
    """Load hidden states from NPZ files for given indices."""
    result = np.zeros((len(indices), 4096), dtype=np.float32)

    # Group by file
    file_groups = {}
    for i, idx in enumerate(indices):
        fpath, pos = index_to_loc[idx]
        if fpath not in file_groups:
            file_groups[fpath] = []
        file_groups[fpath].append((i, pos))

    # Load from each file
    for fpath, items in file_groups.items():
        data = np.load(fpath)
        hidden = data['student_hidden']
        for result_i, file_pos in items:
            result[result_i] = hidden[file_pos]

    return result


def compute_infonce_matrix(student, teacher, normalize=True, temperature=0.1):
    """Compute cross-modal InfoNCE similarity matrix."""
    if normalize:
        student = student / (np.linalg.norm(student, axis=1, keepdims=True) + 1e-8)
        teacher = teacher / (np.linalg.norm(teacher, axis=1, keepdims=True) + 1e-8)

    cross_modal = student @ teacher.T
    cross_modal = cross_modal / temperature
    return cross_modal


class InteractiveBrowser:
    """Interactive browser for before_training - LAPA similarity matrix."""

    def __init__(self, common_indices, lapa_pos, before_loc,
                 batch_size=20, num_batches=50, temperature=0.1, normalize=True):
        self.common_indices = common_indices
        self.lapa_pos = lapa_pos
        self.before_loc = before_loc
        self.batch_size = batch_size
        self.temperature = temperature
        self.normalize = normalize

        self.current_batch = 0
        self.max_batch = min(num_batches - 1, len(common_indices) // batch_size - 1)

        self.fig = None
        self.ax = None
        self.cbar = None

    def create_figure(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(7, 6))
        plt.tight_layout(pad=3.0)

    def update_batch(self, batch_idx):
        if batch_idx < 0 or batch_idx > self.max_batch:
            return False

        # Get batch indices
        start = batch_idx * self.batch_size
        end = start + self.batch_size
        batch_indices = self.common_indices[start:end]

        # Load hidden states
        teacher = load_lapa_batch(batch_indices, self.lapa_pos)
        before = load_npz_batch(batch_indices, self.before_loc)

        # Compute similarity matrix
        sim_matrix = compute_infonce_matrix(before, teacher, self.normalize, self.temperature)

        # Diagonal mean
        diag_mean = np.diag(sim_matrix).mean()

        # Clear and redraw
        self.ax.clear()
        im = self.ax.imshow(sim_matrix, cmap='viridis', aspect='auto')
        self.ax.set_xlabel('Teacher Sample Index', fontsize=10)
        self.ax.set_ylabel('Student Sample Index', fontsize=10)
        if self.cbar:
            self.cbar.remove()
        self.cbar = plt.colorbar(im, ax=self.ax)

        self.fig.suptitle(
            f"InfoNCE Similarity | Temp: {self.temperature} | "
            f"Batch {batch_idx}/{self.max_batch} | "
            f"[Arrows to navigate, q to quit]",
            fontsize=11
        )

        self.current_batch = batch_idx
        self.fig.canvas.draw()
        return True

    def on_key(self, event):
        if event.key in ["right", " "]:
            if self.current_batch < self.max_batch:
                self.update_batch(self.current_batch + 1)
        elif event.key in ["left", "backspace"]:
            if self.current_batch > 0:
                self.update_batch(self.current_batch - 1)
        elif event.key == "q":
            plt.close("all")
            sys.exit(0)

    def run(self):
        print(f"\nBatches available: {self.max_batch + 1} (0-{self.max_batch})")
        print(f"Batch size: {self.batch_size}")
        print(f"Temperature: {self.temperature}")
        print("\nControls: Arrow keys to navigate, q to quit\n")

        self.create_figure()
        self.update_batch(0)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_batches', type=int, default=50)
    parser.add_argument('--normalize', type=lambda x: x.lower() == 'true', default=True)
    args = parser.parse_args()

    print("=" * 70)
    print("INFONCE SIMILARITY: BEFORE TRAINING - LAPA")
    print("=" * 70)

    # Step 1: Scan sources for indices (no hidden states loaded yet)
    lapa_pos = scan_lapa_indices()
    before_loc = scan_npz_folder(BEFORE_TRAINING_DIR, "Before Training")

    # Step 2: Find common indices
    common = set(lapa_pos.keys()) & set(before_loc.keys())
    common_indices = sorted(common)
    print(f"\nCommon indices: {len(common_indices)}")
    print(f"Can create {len(common_indices) // args.batch_size} batches of size {args.batch_size}")

    if len(common_indices) < args.batch_size:
        print("Error: Not enough common indices")
        return

    # Step 3: Run interactive browser
    browser = InteractiveBrowser(
        common_indices, lapa_pos, before_loc,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        temperature=args.temperature,
        normalize=args.normalize,
    )
    browser.run()


if __name__ == "__main__":
    main()
