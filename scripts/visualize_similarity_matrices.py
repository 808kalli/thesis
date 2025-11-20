"""
Visualize student and teacher similarity matrices from saved hidden states.

Usage:
    python visualize_similarity_matrices.py --normalize [True/False]

Controls:
    Right Arrow / Space: Next batch
    Left Arrow / Backspace: Previous batch
    q: Quit
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Default path
DEFAULT_HIDDEN_STATES_DIR = Path("/home/elias/Thesis/hidden_states_logs/")


def load_batch(batch_idx, hidden_states_dir):
    """Load student and teacher hidden states for a batch."""
    batch_file = hidden_states_dir / f"batch_{batch_idx:04d}.npz"

    if not batch_file.exists():
        return None

    data = np.load(batch_file)
    return {
        "student_hidden": data["student_hidden"],
        "teacher_hidden": data["teacher_hidden"],
        "batch_idx": int(data["batch_idx"]),
        "aggregation_method": str(data["aggregation_method"]),
        "batch_size": int(data["batch_size"]),
    }


def compute_similarity_matrix(hidden_states, normalize=True):
    """
    Compute similarity matrix from hidden states.

    Args:
        hidden_states: [batch_size, hidden_dim]
        normalize: Whether to L2 normalize before computing similarity

    Returns:
        similarity_matrix: [batch_size, batch_size]
    """
    if normalize:
        # L2 normalize
        hidden_states = hidden_states / (np.linalg.norm(hidden_states, axis=1, keepdims=True) + 1e-8)

    # Compute similarity matrix: H @ H.T
    sim_matrix = hidden_states @ hidden_states.T

    return sim_matrix


def apply_temperature_and_masking(sim_matrix, temperature=1.0, mask_diagonal=False):
    """
    Apply temperature scaling and optional diagonal masking to similarity matrix.

    Args:
        sim_matrix: [batch_size, batch_size] similarity matrix
        temperature: Temperature for scaling (temperature > 1 makes softmax softer)
        mask_diagonal: If True, set diagonal to -inf (softmax will give 0)

    Returns:
        processed_matrix: similarity matrix with temperature applied and optional diagonal masking
    """
    # Apply temperature scaling
    processed = sim_matrix / temperature if temperature != 1.0 else sim_matrix.copy()

    # Mask diagonal if requested
    if mask_diagonal:
        # Set diagonal to -inf so softmax gives 0
        np.fill_diagonal(processed, -np.inf)

    return processed


class InteractiveBrowser:
    """Interactive browser for similarity matrices."""

    def __init__(self, hidden_states_dir, normalize=True, temperature=1.0, mask_diagonal=False):
        self.hidden_states_dir = Path(hidden_states_dir)
        self.normalize = normalize
        self.temperature = temperature
        self.mask_diagonal = mask_diagonal
        self.current_batch = 0
        self.max_batch = self._find_max_batch()

        # Figure and axes (reused)
        self.fig = None
        self.axes = None
        self.im1 = None
        self.im2 = None
        self.cbar1 = None
        self.cbar2 = None

    def _find_max_batch(self):
        """Find the maximum batch index available."""
        batch_files = sorted(self.hidden_states_dir.glob("batch_*.npz"))
        if not batch_files:
            print(f"Error: No batch files found in {self.hidden_states_dir}")
            return -1
        return len(batch_files) - 1

    def create_figure(self):
        """Create the figure and axes once."""
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))
        plt.tight_layout(pad=3.0)
        return self.fig

    def update_batch(self, batch_idx):
        """Update the displayed batch by modifying existing figure."""
        if batch_idx < 0 or batch_idx > self.max_batch:
            print(f"Invalid batch index: {batch_idx} (valid range: 0-{self.max_batch})")
            return False

        batch_data = load_batch(batch_idx, self.hidden_states_dir)
        if batch_data is None:
            print(f"Failed to load batch {batch_idx}")
            return False

        # Debug: Check if teacher hidden states are actually different
        teacher_hidden = batch_data["teacher_hidden"]
        print(f"\n=== Batch {batch_idx} ===")
        print(f"Teacher hidden states shape: {teacher_hidden.shape}")

        # Check if all rows are identical
        unique_rows = np.unique(teacher_hidden, axis=0)
        print(f"Unique teacher states: {len(unique_rows)} (total: {len(teacher_hidden)})")

        if len(unique_rows) == 1:
            print("⚠️  WARNING: All teacher states in batch are IDENTICAL!")
        else:
            # Show pairwise differences
            print(f"Pairwise differences (L2 norm):")
            for i in range(min(5, len(teacher_hidden))):
                for j in range(i+1, min(i+3, len(teacher_hidden))):
                    diff = np.linalg.norm(teacher_hidden[i] - teacher_hidden[j])
                    print(f"  State {i} vs {j}: {diff:.6f}")

        # Compute similarity matrices
        student_sim = compute_similarity_matrix(batch_data["student_hidden"], normalize=self.normalize)
        teacher_sim = compute_similarity_matrix(batch_data["teacher_hidden"], normalize=self.normalize)

        # Apply temperature scaling and optional diagonal masking
        student_sim = apply_temperature_and_masking(student_sim, temperature=self.temperature, mask_diagonal=self.mask_diagonal)
        teacher_sim = apply_temperature_and_masking(teacher_sim, temperature=self.temperature, mask_diagonal=self.mask_diagonal)

        # Update left plot (student)
        self.axes[0].clear()
        self.im1 = self.axes[0].imshow(student_sim, cmap="viridis", aspect="auto")
        self.axes[0].set_title(f"Student Similarity Matrix\n(Batch {batch_data['batch_idx']}, Size {batch_data['batch_size']})")
        self.axes[0].set_xlabel("Sample Index")
        self.axes[0].set_ylabel("Sample Index")
        if self.cbar1 is not None:
            self.cbar1.remove()
        self.cbar1 = plt.colorbar(self.im1, ax=self.axes[0])

        # Update right plot (teacher)
        self.axes[1].clear()
        self.im2 = self.axes[1].imshow(teacher_sim, cmap="viridis", aspect="auto")
        self.axes[1].set_title("Teacher Similarity Matrix")
        self.axes[1].set_xlabel("Sample Index")
        self.axes[1].set_ylabel("Sample Index")
        if self.cbar2 is not None:
            self.cbar2.remove()
        self.cbar2 = plt.colorbar(self.im2, ax=self.axes[1])

        # Update title
        self.fig.suptitle(
            f"Aggregation: {batch_data['aggregation_method']} | "
            f"Normalize: {self.normalize} | "
            f"Temperature: {self.temperature} | "
            f"Batch {batch_idx}/{self.max_batch} | "
            f"Use arrow keys to navigate, q to quit",
            fontsize=10,
        )

        self.current_batch = batch_idx
        self.fig.canvas.draw()
        return True

    def next_batch(self):
        """Move to next batch."""
        if self.current_batch < self.max_batch:
            self.update_batch(self.current_batch + 1)
        else:
            print(f"Already at last batch ({self.max_batch})")

    def prev_batch(self):
        """Move to previous batch."""
        if self.current_batch > 0:
            self.update_batch(self.current_batch - 1)
        else:
            print("Already at first batch (0)")

    def on_key(self, event):
        """Handle keyboard events."""
        if event.key in ["right", " "]:
            self.next_batch()
        elif event.key in ["left", "backspace"]:
            self.prev_batch()
        elif event.key == "q":
            print("Quitting...")
            plt.close("all")
            sys.exit(0)

    def run(self):
        """Start interactive browser."""
        if self.max_batch < 0:
            return

        print(f"Loading batches from: {self.hidden_states_dir}")
        print(f"Found {self.max_batch + 1} batches (0-{self.max_batch})")
        print(f"Normalization: {self.normalize}")
        print(f"Temperature: {self.temperature}")
        print(f"Diagonal Masking: {self.mask_diagonal}")
        print("\nControls:")
        print("  Right Arrow / Space: Next batch")
        print("  Left Arrow / Backspace: Previous batch")
        print("  q: Quit")
        print()

        # Create figure once
        self.create_figure()

        # Show first batch
        self.update_batch(0)

        # Connect key press event
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Show and block
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize student and teacher similarity matrices from hidden states logs."
    )
    parser.add_argument(
        "--hidden_states_dir",
        type=str,
        default=str(DEFAULT_HIDDEN_STATES_DIR),
        help=f"Path to hidden_states_logs directory (default: {DEFAULT_HIDDEN_STATES_DIR})"
    )
    parser.add_argument(
        "--normalize",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to L2 normalize hidden states before computing similarity (default: True)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for scaling similarity matrices (higher = softer softmax, default: 1.0)"
    )
    parser.add_argument(
        "--mask_diagonal",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to mask diagonal with -inf (softmax will give 0 on diagonal, default: False)"
    )

    args = parser.parse_args()

    # Verify directory exists
    hidden_states_dir = Path(args.hidden_states_dir)
    if not hidden_states_dir.exists():
        print(f"Error: Directory not found: {hidden_states_dir}")
        sys.exit(1)

    # Start browser
    browser = InteractiveBrowser(
        hidden_states_dir,
        normalize=args.normalize,
        temperature=args.temperature,
        mask_diagonal=args.mask_diagonal
    )
    browser.run()


if __name__ == "__main__":
    main()
