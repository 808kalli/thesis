#!/usr/bin/env python3
"""
Plot 2D top-down (XY plane) trajectories for all tasks.

Compares GT, Distilled Model, and Vanilla Model trajectories by integrating
action deltas to reconstruct end-effector positions.

Usage:
    python scripts/plot_2d_trajectories.py \
        --model_data_path /home/elias/Thesis/action_data/actions_libero_object_distilled.h5 \
        --vanilla_data_path /home/elias/Thesis/action_data/actions_libero_object_vanilla.h5 \
        --save_dir figures/trajectories
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


# Task ID mapping: evaluation task ID -> training task ID
# Evaluation uses alphabetical order, training uses dataset order
EVAL_TO_TRAIN_TASK_MAP = {
    0: 4,   # alphabet soup
    1: 2,   # cream cheese
    2: 6,   # salad dressing
    3: 3,   # bbq sauce
    4: 1,   # ketchup
    5: 8,   # tomato sauce
    6: 7,   # butter
    7: 5,   # milk
    8: 9,   # chocolate pudding
    9: 0,   # orange juice
}

# Reverse mapping: training task ID -> evaluation task ID
TRAIN_TO_EVAL_TASK_MAP = {v: k for k, v in EVAL_TO_TRAIN_TASK_MAP.items()}

# Task names by training task ID
TASK_NAMES = {
    0: "orange juice",
    1: "ketchup",
    2: "cream cheese",
    3: "bbq sauce",
    4: "alphabet soup",
    5: "milk",
    6: "salad dressing",
    7: "butter",
    8: "tomato sauce",
    9: "chocolate pudding",
}


def load_ground_truth_from_hf(dataset_path="/home/elias/Thesis/raw_datasets/libero_object_noops"):
    """Load all ground truth actions from local HuggingFace dataset.

    Returns:
        Dict mapping training_task_id -> list of episode action arrays
    """
    from datasets import load_dataset

    print(f"Loading ground truth dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path, split="train")

    print(f"Loaded {len(dataset)} timesteps")
    print("Grouping timesteps into episodes...")

    # Group data by episode
    episodes_data = {}
    for row in tqdm(dataset):
        ep_idx = row["episode_index"]
        if ep_idx not in episodes_data:
            episodes_data[ep_idx] = {
                "actions": [],
                "task_id": row["task_index"],
            }
        episodes_data[ep_idx]["actions"].append(row["action"])

    print(f"Found {len(episodes_data)} episodes")

    # Organize by task
    task_episodes = {i: [] for i in range(10)}
    for ep_idx in sorted(episodes_data.keys()):
        ep_data = episodes_data[ep_idx]
        task_id = ep_data["task_id"]
        actions = np.array(ep_data["actions"])  # [T, 7]
        task_episodes[task_id].append(actions)

    # Print summary
    for task_id in sorted(task_episodes.keys()):
        print(f"  Task {task_id} ({TASK_NAMES[task_id]}): {len(task_episodes[task_id])} episodes")

    return task_episodes


def load_model_rollout_data(h5_path):
    """Load model rollout data from HDF5 file.

    Returns:
        Dict mapping training_task_id -> list of episode action arrays
    """
    with h5py.File(h5_path, "r") as f:
        all_actions = f["actions"][:]
        all_episode_metadata = f["episode_metadata"][:]

    # Organize by task (using training task IDs)
    task_episodes = {i: [] for i in range(10)}

    for metadata in all_episode_metadata:
        start_idx = metadata["start_idx"]
        end_idx = metadata["end_idx"]
        eval_task_id = metadata["task_id"]

        # Map evaluation task ID to training task ID
        train_task_id = EVAL_TO_TRAIN_TASK_MAP.get(eval_task_id, eval_task_id)

        # Get action deltas
        actions = all_actions[start_idx:end_idx]  # [T, 7]
        task_episodes[train_task_id].append(actions)

    # Print summary
    total = 0
    for task_id in sorted(task_episodes.keys()):
        count = len(task_episodes[task_id])
        total += count
        if count > 0:
            print(f"  Task {task_id} ({TASK_NAMES[task_id]}): {count} episodes")
    print(f"  Total: {total} episodes")

    return task_episodes


def integrate_actions_to_trajectory(actions):
    """Integrate action deltas to get XY trajectory.

    Args:
        actions: [T, 7] array of action deltas

    Returns:
        [T+1, 2] array of XY positions (starting from origin), rotated 90 degrees clockwise
    """
    # Extract XY deltas (first 2 dimensions)
    xy_deltas = actions[:, :2]  # [T, 2]

    # Cumulative sum to get positions, prepend origin
    positions = np.vstack([np.zeros((1, 2)), np.cumsum(xy_deltas, axis=0)])  # [T+1, 2]

    # Rotate 90 degrees clockwise: (x, y) -> (y, -x)
    rotated = np.column_stack([positions[:, 1], -positions[:, 0]])

    # Flip on vertical axis (negate x): (x, y) -> (-x, y)
    flipped = np.column_stack([-rotated[:, 0], rotated[:, 1]])

    # Smooth each trajectory individually
    sigma = 3
    smoothed = np.column_stack([
        gaussian_filter1d(flipped[:, 0], sigma=sigma),
        gaussian_filter1d(flipped[:, 1], sigma=sigma)
    ])

    return smoothed


def resample_trajectory(trajectory, num_points=100):
    """Resample trajectory to fixed number of points using linear interpolation.

    Args:
        trajectory: [T, 2] array of XY positions
        num_points: Number of points to resample to

    Returns:
        [num_points, 2] array of resampled positions
    """
    t_original = np.linspace(0, 1, len(trajectory))
    t_new = np.linspace(0, 1, num_points)

    resampled = np.zeros((num_points, 2))
    resampled[:, 0] = np.interp(t_new, t_original, trajectory[:, 0])
    resampled[:, 1] = np.interp(t_new, t_original, trajectory[:, 1])

    return resampled


def compute_mean_trajectory(episodes, num_points=100):
    """Compute mean trajectory from a list of episodes.

    Args:
        episodes: List of action arrays
        num_points: Number of points to resample each trajectory to

    Returns:
        [num_points, 2] mean trajectory, or None if no episodes
    """
    if not episodes:
        return None

    # Convert all episodes to trajectories and resample
    all_trajs = []
    for actions in episodes:
        traj = integrate_actions_to_trajectory(actions)
        resampled = resample_trajectory(traj, num_points)
        all_trajs.append(resampled)

    all_trajs = np.array(all_trajs)  # [N, num_points, 2]

    # Compute mean
    mean_traj = np.mean(all_trajs, axis=0)  # [num_points, 2]

    return mean_traj




def filter_outlier_episodes(episodes, max_y=None):
    """Filter out episodes where trajectory goes above max_y threshold.

    Args:
        episodes: List of action arrays
        max_y: Maximum Y value threshold. Episodes going above this are filtered out.

    Returns:
        Filtered list of episodes
    """
    if max_y is None:
        return episodes

    filtered = []
    for actions in episodes:
        traj = integrate_actions_to_trajectory(actions)
        if traj[:, 1].max() <= max_y:
            filtered.append(actions)
    return filtered


def plot_task_trajectories(gt_episodes, model_episodes, vanilla_episodes, task_id, save_path, plot_mean=False):
    """Plot XY trajectories for a single task.

    Args:
        gt_episodes: List of GT action arrays for this task
        model_episodes: List of model action arrays for this task
        vanilla_episodes: List of vanilla action arrays for this task (can be None)
        task_id: Training task ID
        save_path: Path to save the figure
        plot_mean: If True, plot mean trajectory; if False, plot all individual trajectories
    """
    # Filter outliers for task 5 (milk) - remove trajectories going to positive Y
    if task_id == 5:
        orig_model_count = len(model_episodes) if model_episodes else 0
        model_episodes = filter_outlier_episodes(model_episodes, max_y=0)
        filtered_count = orig_model_count - (len(model_episodes) if model_episodes else 0)
        if filtered_count > 0:
            print(f"  Task 5: Filtered {filtered_count} outlier distilled trajectories (positive Y)")

    fig, ax = plt.subplots(figsize=(10, 10))

    if plot_mean:
        # Compute and plot mean trajectories
        gt_mean = compute_mean_trajectory(gt_episodes)
        model_mean = compute_mean_trajectory(model_episodes)
        vanilla_mean = compute_mean_trajectory(vanilla_episodes) if vanilla_episodes else None

        if gt_mean is not None:
            ax.plot(gt_mean[:, 0], gt_mean[:, 1], color='blue', linewidth=2.5, label=f'GT ({len(gt_episodes)} eps)')

        if model_mean is not None:
            ax.plot(model_mean[:, 0], model_mean[:, 1], color='red', linewidth=2.5, label=f'Distilled ({len(model_episodes)} eps)')

        if vanilla_mean is not None:
            ax.plot(vanilla_mean[:, 0], vanilla_mean[:, 1], color='green', linewidth=2.5, label=f'Vanilla ({len(vanilla_episodes)} eps)')

        title_suffix = "Mean Trajectory"
    else:
        # Plot all individual trajectories
        for i, actions in enumerate(gt_episodes):
            traj = integrate_actions_to_trajectory(actions)
            label = f'GT ({len(gt_episodes)} eps)' if i == 0 else None
            ax.plot(traj[:, 0], traj[:, 1], color='blue', linewidth=1, alpha=0.5, label=label)

        for i, actions in enumerate(model_episodes):
            traj = integrate_actions_to_trajectory(actions)
            label = f'Distilled ({len(model_episodes)} eps)' if i == 0 else None
            ax.plot(traj[:, 0], traj[:, 1], color='red', linewidth=1, alpha=0.5, label=label)

        if vanilla_episodes:
            for i, actions in enumerate(vanilla_episodes):
                traj = integrate_actions_to_trajectory(actions)
                label = f'Vanilla ({len(vanilla_episodes)} eps)' if i == 0 else None
                ax.plot(traj[:, 0], traj[:, 1], color='green', linewidth=1, alpha=0.5, label=label)

        title_suffix = "All Trajectories"

    ax.legend(loc='best', fontsize=12)

    ax.set_xlabel('X Position (cumulative delta)', fontsize=14)
    ax.set_ylabel('Y Position (cumulative delta)', fontsize=14)
    ax.set_title(f'Task {task_id}: {TASK_NAMES[task_id]}\n{title_suffix}', fontsize=16, fontweight='bold')

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def plot_all_tasks_grid(gt_task_episodes, model_task_episodes, vanilla_task_episodes, save_path):
    """Plot all tasks in a 2x5 grid.

    Args:
        gt_task_episodes: Dict mapping task_id -> list of action arrays
        model_task_episodes: Dict mapping task_id -> list of action arrays
        vanilla_task_episodes: Dict mapping task_id -> list of action arrays (can be None)
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    for idx, task_id in enumerate(range(10)):
        ax = axes[idx]

        gt_episodes = gt_task_episodes.get(task_id, [])
        model_episodes = model_task_episodes.get(task_id, [])
        vanilla_episodes = vanilla_task_episodes.get(task_id, []) if vanilla_task_episodes else []

        # Filter outliers for task 5 (milk) - remove trajectories going to positive Y
        if task_id == 5:
            model_episodes = filter_outlier_episodes(model_episodes, max_y=0)

        # Compute and plot mean trajectories
        gt_mean = compute_mean_trajectory(gt_episodes)
        model_mean = compute_mean_trajectory(model_episodes)
        vanilla_mean = compute_mean_trajectory(vanilla_episodes) if vanilla_episodes else None

        if gt_mean is not None:
            ax.plot(gt_mean[:, 0], gt_mean[:, 1], color='blue', linewidth=2)

        if model_mean is not None:
            ax.plot(model_mean[:, 0], model_mean[:, 1], color='red', linewidth=2)

        if vanilla_mean is not None:
            ax.plot(vanilla_mean[:, 0], vanilla_mean[:, 1], color='green', linewidth=2)

        ax.set_title(f'Task {task_id}: {TASK_NAMES[task_id]}', fontsize=11, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Create shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label='GT'),
        Line2D([0], [0], color='red', linewidth=2, label='Distilled'),
    ]
    if vanilla_task_episodes:
        legend_elements.append(Line2D([0], [0], color='green', linewidth=2, label='Vanilla'))

    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=14,
               bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('Mean Trajectories - All Tasks', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved grid: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot 2D top-down trajectories for all tasks")
    parser.add_argument(
        "--model_data_path",
        type=str,
        required=True,
        help="Path to distilled model rollout H5 file"
    )
    parser.add_argument(
        "--vanilla_data_path",
        type=str,
        default=None,
        help="Path to vanilla model rollout H5 file (optional)"
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="/home/elias/Thesis/raw_datasets/libero_object_noops",
        help="Path to local HuggingFace dataset for ground truth"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="figures/trajectories",
        help="Directory to save trajectory plots"
    )
    parser.add_argument(
        "--grid_only",
        action="store_true",
        help="Only generate the grid plot, not individual task plots"
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Plot only this specific task ID (training task ID, 0-9)"
    )
    parser.add_argument(
        "--mean",
        action="store_true",
        help="Plot mean trajectory instead of all individual trajectories"
    )

    args = parser.parse_args()

    # Load ground truth data
    print("Loading ground truth data...")
    gt_task_episodes = load_ground_truth_from_hf(args.hf_dataset)

    # Load model data
    print(f"\nLoading distilled model data from {args.model_data_path}...")
    model_task_episodes = load_model_rollout_data(args.model_data_path)

    # Load vanilla model data if provided
    vanilla_task_episodes = None
    if args.vanilla_data_path:
        print(f"\nLoading vanilla model data from {args.vanilla_data_path}...")
        vanilla_task_episodes = load_model_rollout_data(args.vanilla_data_path)

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # If specific task requested, only plot that task
    if args.task_id is not None:
        task_id = args.task_id
        print(f"\nGenerating plot for task {task_id} ({TASK_NAMES[task_id]})...")
        gt_eps = gt_task_episodes.get(task_id, [])
        model_eps = model_task_episodes.get(task_id, [])
        vanilla_eps = vanilla_task_episodes.get(task_id, []) if vanilla_task_episodes else None

        task_name_clean = TASK_NAMES[task_id].replace(" ", "_")
        plot_task_trajectories(
            gt_eps,
            model_eps,
            vanilla_eps,
            task_id,
            save_dir / f"task{task_id}_{task_name_clean}.png",
            plot_mean=args.mean
        )
    else:
        # Plot grid of all tasks
        print("\nGenerating grid plot...")
        plot_all_tasks_grid(
            gt_task_episodes,
            model_task_episodes,
            vanilla_task_episodes,
            save_dir / "all_tasks_trajectories.png"
        )

        # Plot individual tasks
        if not args.grid_only:
            print("\nGenerating individual task plots...")
            for task_id in range(10):
                gt_eps = gt_task_episodes.get(task_id, [])
                model_eps = model_task_episodes.get(task_id, [])
                vanilla_eps = vanilla_task_episodes.get(task_id, []) if vanilla_task_episodes else None

                if not gt_eps and not model_eps:
                    print(f"  Skipping task {task_id} - no data")
                    continue

                task_name_clean = TASK_NAMES[task_id].replace(" ", "_")
                plot_task_trajectories(
                    gt_eps,
                    model_eps,
                    vanilla_eps,
                    task_id,
                    save_dir / f"task{task_id}_{task_name_clean}.png",
                    plot_mean=args.mean
                )

    print("\nDone!")


if __name__ == "__main__":
    main()
