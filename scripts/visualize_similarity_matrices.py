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
DEFAULT_HIDDEN_STATES_DIR = Path("/home/elias/Thesis/checkpoints/openvla_distilled/hidden_states_logs")


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


def plot_batch(batch_data, normalize=True):
    """Plot student and teacher similarity matrices for a batch."""
    if batch_data is None:
        return None

    student_sim = compute_similarity_matrix(batch_data["student_hidden"], normalize=normalize)
    teacher_sim = compute_similarity_matrix(batch_data["teacher_hidden"], normalize=normalize)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Student similarity matrix
    im1 = axes[0].imshow(student_sim, cmap="viridis", aspect="auto")
    axes[0].set_title(f"Student Similarity Matrix\n(Batch {batch_data['batch_idx']}, Size {batch_data['batch_size']})")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Sample Index")
    plt.colorbar(im1, ax=axes[0])

    # Teacher similarity matrix
    im2 = axes[1].imshow(teacher_sim, cmap="viridis", aspect="auto")
    axes[1].set_title("Teacher Similarity Matrix")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Sample Index")
    plt.colorbar(im2, ax=axes[1])

    # Add metadata
    fig.suptitle(
        f"Aggregation: {batch_data['aggregation_method']} | "
        f"Normalize: {normalize} | "
        f"Use arrow keys to navigate, q to quit",
        fontsize=10,
        y=0.98
    )

    plt.tight_layout()
    return fig, (student_sim, teacher_sim)


class InteractiveBrowser:
    """Interactive browser for similarity matrices."""

    def __init__(self, hidden_states_dir, normalize=True):
        self.hidden_states_dir = Path(hidden_states_dir)
        self.normalize = normalize
        self.current_batch = 0
        self.max_batch = self._find_max_batch()
        self.fig = None
        self.similarity_matrices = None

        # Setup plot
        plt.ion()  # Interactive mode

    def _find_max_batch(self):
        """Find the maximum batch index available."""
        batch_files = sorted(self.hidden_states_dir.glob("batch_*.npz"))
        if not batch_files:
            print(f"Error: No batch files found in {self.hidden_states_dir}")
            return -1
        return len(batch_files) - 1

    def show_batch(self, batch_idx):
        """Show a specific batch."""
        if batch_idx < 0 or batch_idx > self.max_batch:
            print(f"Invalid batch index: {batch_idx} (valid range: 0-{self.max_batch})")
            return False

        batch_data = load_batch(batch_idx, self.hidden_states_dir)
        if batch_data is None:
            print(f"Failed to load batch {batch_idx}")
            return False

        # Clear previous figure
        if self.fig is not None:
            plt.close(self.fig)

        result = plot_batch(batch_data, normalize=self.normalize)
        if result is None:
            return False

        self.fig, self.similarity_matrices = result
        self.current_batch = batch_idx

        plt.draw()
        plt.pause(0.001)
        return True

    def next_batch(self):
        """Move to next batch."""
        if self.current_batch < self.max_batch:
            self.show_batch(self.current_batch + 1)
        else:
            print(f"Already at last batch ({self.max_batch})")

    def prev_batch(self):
        """Move to previous batch."""
        if self.current_batch > 0:
            self.show_batch(self.current_batch - 1)
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
        print("\nControls:")
        print("  Right Arrow / Space: Next batch")
        print("  Left Arrow / Backspace: Previous batch")
        print("  q: Quit")
        print()

        # Show first batch
        self.show_batch(0)

        # Connect key press event
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

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

    args = parser.parse_args()

    # Verify directory exists
    hidden_states_dir = Path(args.hidden_states_dir)
    if not hidden_states_dir.exists():
        print(f"Error: Directory not found: {hidden_states_dir}")
        sys.exit(1)

    # Start browser
    browser = InteractiveBrowser(hidden_states_dir, normalize=args.normalize)
    browser.run()


if __name__ == "__main__":
    main()
