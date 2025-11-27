"""
Visualize student and teacher similarity matrices from saved hidden states.

Supports two loss types:
1. KL-Divergence: Shows self-similarity matrices for student and teacher
2. InfoNCE: Shows cross-modal similarity matrix between student and teacher

Usage:
    python visualize_similarity_matrices.py [options]

Arguments:
    --loss_type: Type of loss to visualize - 'kl_divergence' (default) or 'infonce'
    --normalize: L2 normalize hidden states before computing similarity (default: True)
    --common_scale: Use common color scale for both plots (default: False)

    KL-Divergence options:
        --temperature_student: Temperature for student similarity matrices (default: 1.0)
        --temperature_teacher: Temperature for teacher similarity matrices (default: 1.0)
        --mask_diagonal: Mask diagonal with -inf (default: False)
        --apply_softmax: Apply softmax to similarity matrices (default: True)

    InfoNCE options:
        --temperature_infonce: Temperature for cross-modal similarities (default: 0.1)

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
from scipy.special import softmax, logsumexp

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


def apply_temperature_and_masking(sim_matrix, temperature=1.0, mask_diagonal=False, apply_softmax=True):
    """
    Apply temperature scaling, optional diagonal masking, and optional softmax to similarity matrix.

    Args:
        sim_matrix: [batch_size, batch_size] similarity matrix
        temperature: Temperature for scaling (temperature > 1 makes softmax softer, only used if apply_softmax=True)
        mask_diagonal: If True, set diagonal to -inf (softmax will give 0)
        apply_softmax: If True, apply softmax to get probability distribution

    Returns:
        processed_matrix: similarity matrix with temperature, masking, and optional softmax applied
    """
    processed = sim_matrix.copy()

    # Mask diagonal if requested (before temperature scaling and softmax)
    if mask_diagonal:
        # Set diagonal to -inf so softmax gives 0
        np.fill_diagonal(processed, -np.inf)

    if apply_softmax:
        # Apply temperature scaling
        processed = processed / temperature if temperature != 1.0 else processed

        # Apply softmax across each row to get probability distribution
        processed = softmax(processed, axis=1)

    return processed


def compute_infonce_cross_modal_matrix(student_hidden, teacher_hidden, normalize=True, temperature=0.1):
    """
    Compute cross-modal InfoNCE similarity matrix between student and teacher.

    Args:
        student_hidden: [batch_size, hidden_dim] - student embeddings
        teacher_hidden: [batch_size, hidden_dim] - teacher embeddings
        normalize: Whether to L2 normalize before computing similarity
        temperature: Temperature for scaling (lower = sharper distribution)

    Returns:
        cross_modal_matrix: [batch_size, batch_size] cross-modal similarity matrix
                           where (i,j) = student_i similarity to teacher_j
    """
    if normalize:
        # L2 normalize
        student_hidden = student_hidden / (np.linalg.norm(student_hidden, axis=1, keepdims=True) + 1e-8)
        teacher_hidden = teacher_hidden / (np.linalg.norm(teacher_hidden, axis=1, keepdims=True) + 1e-8)

    # Compute cross-modal similarity: S @ T.T
    cross_modal = student_hidden @ teacher_hidden.T

    # Apply temperature scaling
    cross_modal = cross_modal / temperature

    # Return raw similarity scores (temperature-scaled cross-modal similarities)
    return cross_modal


class InteractiveBrowser:
    """Interactive browser for similarity matrices."""

    def __init__(self, hidden_states_dir, normalize=True, temperature_student=1.0, temperature_teacher=1.0, mask_diagonal=False, common_scale=False, apply_softmax=True, loss_type="kl_divergence", temperature_infonce=0.1):
        self.hidden_states_dir = Path(hidden_states_dir)
        self.normalize = normalize
        self.temperature_student = temperature_student
        self.temperature_teacher = temperature_teacher
        self.mask_diagonal = mask_diagonal
        self.common_scale = common_scale
        self.apply_softmax = apply_softmax
        self.loss_type = loss_type  # "kl_divergence" or "infonce"
        self.temperature_infonce = temperature_infonce
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
        # Create 1 subplot for InfoNCE, 2 subplots for KL divergence
        num_plots = 1 if self.loss_type == "infonce" else 2
        figsize = (7, 6) if num_plots == 1 else (14, 6)
        self.fig, self.axes = plt.subplots(1, num_plots, figsize=figsize)

        # Ensure axes is always a list for consistency
        if num_plots == 1:
            self.axes = [self.axes]

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

        # Compute similarity matrices based on loss type
        if self.loss_type == "kl_divergence":
            student_sim = compute_similarity_matrix(batch_data["student_hidden"], normalize=self.normalize)
            teacher_sim = compute_similarity_matrix(batch_data["teacher_hidden"], normalize=self.normalize)

            # Apply temperature scaling, optional diagonal masking, and optional softmax (with separate temperatures)
            student_sim = apply_temperature_and_masking(student_sim, temperature=self.temperature_student, mask_diagonal=self.mask_diagonal, apply_softmax=self.apply_softmax)
            teacher_sim = apply_temperature_and_masking(teacher_sim, temperature=self.temperature_teacher, mask_diagonal=self.mask_diagonal, apply_softmax=self.apply_softmax)

            plot_title_left = "Student Similarity Matrix (Self-similarity)"
            plot_title_right = "Teacher Similarity Matrix (Self-similarity)"

        elif self.loss_type == "infonce":
            # Compute cross-modal InfoNCE similarity matrix
            student_sim = compute_infonce_cross_modal_matrix(
                batch_data["student_hidden"],
                batch_data["teacher_hidden"],
                normalize=self.normalize,
                temperature=self.temperature_infonce
            )
            # For InfoNCE, both "plots" show the same cross-modal matrix (student→teacher)
            teacher_sim = student_sim

            plot_title_left = "InfoNCE Cross-Modal Matrix (log-probs)"
            plot_title_right = "InfoNCE Cross-Modal Matrix (log-probs)"
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Determine color scale
        if self.common_scale:
            # Use common min/max across both matrices
            vmin = min(student_sim.min(), teacher_sim.min())
            vmax = max(student_sim.max(), teacher_sim.max())
        else:
            vmin = None
            vmax = None

        if self.loss_type == "infonce":
            # InfoNCE: Single cross-modal matrix
            self.axes[0].clear()
            self.im1 = self.axes[0].imshow(student_sim, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
            self.axes[0].set_title(f"{plot_title_left}\n(Batch {batch_data['batch_idx']}, Size {batch_data['batch_size']})")
            self.axes[0].set_xlabel("Teacher Sample Index")
            self.axes[0].set_ylabel("Student Sample Index")
            if self.cbar1 is not None:
                self.cbar1.remove()
            self.cbar1 = plt.colorbar(self.im1, ax=self.axes[0])
        else:
            # KL Divergence: Two self-similarity matrices
            # Update left plot (student)
            self.axes[0].clear()
            self.im1 = self.axes[0].imshow(student_sim, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
            self.axes[0].set_title(f"{plot_title_left}\n(Batch {batch_data['batch_idx']}, Size {batch_data['batch_size']})")
            self.axes[0].set_xlabel("Sample Index")
            self.axes[0].set_ylabel("Sample Index")
            if self.cbar1 is not None:
                self.cbar1.remove()
            self.cbar1 = plt.colorbar(self.im1, ax=self.axes[0])

            # Update right plot (teacher)
            self.axes[1].clear()
            self.im2 = self.axes[1].imshow(teacher_sim, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
            self.axes[1].set_title(plot_title_right)
            self.axes[1].set_xlabel("Sample Index")
            self.axes[1].set_ylabel("Sample Index")
            if self.cbar2 is not None:
                self.cbar2.remove()
            self.cbar2 = plt.colorbar(self.im2, ax=self.axes[1])

        # Update title - show different parameters based on loss type
        if self.loss_type == "kl_divergence":
            title = (
                f"Loss Type: KL-Divergence | "
                f"Aggregation: {batch_data['aggregation_method']} | "
                f"Normalize: {self.normalize} | "
                f"Apply Softmax: {self.apply_softmax} | "
                f"Temp (Student): {self.temperature_student} | "
                f"Temp (Teacher): {self.temperature_teacher} | "
                f"Mask Diagonal: {self.mask_diagonal} | "
                f"Common Scale: {self.common_scale} | "
                f"Batch {batch_idx}/{self.max_batch} | "
                f"Use arrow keys to navigate, q to quit"
            )
        else:  # infonce
            title = (
                f"Loss Type: InfoNCE | "
                f"Aggregation: {batch_data['aggregation_method']} | "
                f"Normalize: {self.normalize} | "
                f"Temperature: {self.temperature_infonce} | "
                f"Common Scale: {self.common_scale} | "
                f"Batch {batch_idx}/{self.max_batch} | "
                f"Use arrow keys to navigate, q to quit"
            )

        self.fig.suptitle(title, fontsize=10)

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
        print(f"Loss Type: {self.loss_type}")
        print(f"Normalization: {self.normalize}")

        if self.loss_type == "kl_divergence":
            print(f"Apply Softmax: {self.apply_softmax}")
            print(f"Temperature (Student): {self.temperature_student}")
            print(f"Temperature (Teacher): {self.temperature_teacher}")
            print(f"Diagonal Masking: {self.mask_diagonal}")
        elif self.loss_type == "infonce":
            print(f"Temperature (InfoNCE): {self.temperature_infonce}")

        print(f"Common Scale: {self.common_scale}")
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
        "--temperature_student",
        type=float,
        default=1.0,
        help="Temperature for student similarity matrices (higher = softer softmax, default: 1.0)"
    )
    parser.add_argument(
        "--temperature_teacher",
        type=float,
        default=1.0,
        help="Temperature for teacher similarity matrices (higher = softer softmax, default: 1.0)"
    )
    parser.add_argument(
        "--mask_diagonal",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to mask diagonal with -inf (softmax will give 0 on diagonal, default: False)"
    )
    parser.add_argument(
        "--common_scale",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to use common color scale for both student and teacher matrices (default: False)"
    )
    parser.add_argument(
        "--apply_softmax",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to apply softmax to similarity matrices (default: True)"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="kl_divergence",
        choices=["kl_divergence", "infonce"],
        help="Type of loss visualization: 'kl_divergence' (self-similarity) or 'infonce' (cross-modal, default: kl_divergence)"
    )
    parser.add_argument(
        "--temperature_infonce",
        type=float,
        default=0.1,
        help="Temperature for InfoNCE loss visualization (lower = sharper, default: 0.1)"
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
        temperature_student=args.temperature_student,
        temperature_teacher=args.temperature_teacher,
        mask_diagonal=args.mask_diagonal,
        common_scale=args.common_scale,
        apply_softmax=args.apply_softmax,
        loss_type=args.loss_type,
        temperature_infonce=args.temperature_infonce
    )
    browser.run()


if __name__ == "__main__":
    main()
