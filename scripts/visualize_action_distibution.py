"""
Visualize action distributions using histograms or UMAP.

Compares GT, Model, and Vanilla Model data using either:
1. Histograms: Action distributions with bar plots showing overlaid histograms
2. UMAP: Action distributions using UMAP dimensionality reduction

Example usage:
    # Histogram mode - shows overlaid bar plots (3-way comparison, all tasks)
    python scripts/visualize_action_distibution.py \
        --model_data_path /home/elias/Thesis/action_data/actions_libero_spatial_checkpoints_seed7.h5 \
        --vanilla_data_path /home/elias/Thesis/action_data/actions_libero_spatial_checkpoints_seed7_vanilla.h5 \
        --num_episodes 100 \
        --mode histogram \
        --save_path action_histograms_3way.png

    # UMAP mode - shows end-effector pose distributions (3-way comparison, all tasks)
    python scripts/visualize_action_distibution.py \
        --model_data_path /home/elias/Thesis/action_data/actions_libero_spatial_checkpoints_seed7.h5 \
        --vanilla_data_path /home/elias/Thesis/action_data/actions_libero_spatial_checkpoints_seed7_vanilla.h5 \
        --num_episodes 100 \
        --mode umap \
        --save_path eef_umap_3way.png

    # Filter to only task 1 (single task visualization)
    python scripts/visualize_action_distibution.py \
        --model_data_path /home/elias/Thesis/action_data/actions_libero_spatial_checkpoints_seed7.h5 \
        --vanilla_data_path /home/elias/Thesis/action_data/actions_libero_spatial_checkpoints_seed7_vanilla.h5 \
        --num_episodes 432 \
        --mode histogram \
        --task_id 1 \
        --save_path action_histograms_task1.png
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import umap


def load_ground_truth_from_hf(dataset_name="aopolin-lv/libero_spatial_no_noops_lerobot_v21", max_episodes=None, load_states=False):
    """Load ground truth actions from HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset name
        max_episodes: Maximum total number of episodes to load (will be distributed across tasks)
        load_states: If True, also load states for UMAP visualization

    Returns:
        Tuple of (episodes_list, episodes_per_task_dict)
    """
    from datasets import load_dataset

    print(f"Loading ground truth dataset from HuggingFace: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

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
            if load_states:
                episodes_data[ep_idx]["states"] = []
        episodes_data[ep_idx]["actions"].append(row["action"])
        if load_states:
            episodes_data[ep_idx]["states"].append(row["observation.state"])

    print(f"Found {len(episodes_data)} episodes")

    # Count episodes per task first
    task_episode_counts = {}
    for ep_idx in sorted(episodes_data.keys()):
        task_id = episodes_data[ep_idx]["task_id"]
        if task_id not in task_episode_counts:
            task_episode_counts[task_id] = 0
        task_episode_counts[task_id] += 1

    print(f"Episodes per task in dataset: {task_episode_counts}")

    # If max_episodes specified, limit episodes per task proportionally
    episodes_per_task = {}
    if max_episodes is not None:
        total_episodes = sum(task_episode_counts.values())
        if total_episodes > max_episodes:
            # Distribute proportionally, ensuring we hit exactly max_episodes
            task_ids = sorted(task_episode_counts.keys())

            # First pass: distribute proportionally with floor
            for task_id in task_ids:
                count = task_episode_counts[task_id]
                episodes_per_task[task_id] = int(count * max_episodes / total_episodes)

            # Second pass: distribute remainder to tasks with highest fractional parts
            current_total = sum(episodes_per_task.values())
            remainder = max_episodes - current_total

            if remainder > 0:
                # Calculate fractional parts
                fractional_parts = []
                for task_id in task_ids:
                    count = task_episode_counts[task_id]
                    exact = count * max_episodes / total_episodes
                    fractional = exact - int(exact)
                    fractional_parts.append((fractional, task_id))

                # Sort by fractional part (descending) and add 1 episode to top tasks
                fractional_parts.sort(reverse=True)
                for i in range(remainder):
                    _, task_id = fractional_parts[i]
                    episodes_per_task[task_id] += 1
        else:
            episodes_per_task = task_episode_counts.copy()
    else:
        episodes_per_task = task_episode_counts.copy()

    print(f"Will load {sum(episodes_per_task.values())} episodes: {episodes_per_task}")

    # Now load episodes according to the distribution
    episodes_list = []
    task_counts = {tid: 0 for tid in episodes_per_task.keys()}

    for ep_idx in tqdm(sorted(episodes_data.keys())):
        ep_data = episodes_data[ep_idx]
        task_id = ep_data["task_id"]

        # Skip if we've loaded enough for this task
        if task_counts[task_id] >= episodes_per_task[task_id]:
            continue

        actions = np.array(ep_data["actions"])  # [T, 7] - action deltas

        episode_dict = {
            "episode_idx": ep_idx,
            "task_id": task_id,
            "actions": actions,  # Action deltas
        }

        if load_states and "states" in ep_data:
            states = np.array(ep_data["states"])  # [T, 8]: [eef_pos(3), eef_euler(3), gripper_qpos(2)]
            episode_dict["eef_positions"] = states[:, :3]      # [T, 3] - xyz positions
            episode_dict["eef_orientations"] = states[:, 3:6]  # [T, 3] - roll, pitch, yaw
            episode_dict["gripper_states"] = states[:, 6:8]    # [T, 2] - gripper qpos

        episodes_list.append(episode_dict)
        task_counts[task_id] += 1

    print(f"Loaded {len(episodes_list)} episodes: {task_counts}")
    return episodes_list, episodes_per_task


def load_model_rollout_data(h5_path, episodes_per_task=None):
    """Load model rollout data from HDF5 file.

    Args:
        h5_path: Path to HDF5 file
        episodes_per_task: Optional dict mapping task_id -> num_episodes to select.
                          If provided, will balance episodes across tasks to match GT.

    Returns:
        List of episode dicts
    """
    with h5py.File(h5_path, "r") as f:
        all_actions = f["actions"][:]
        all_episode_metadata = f["episode_metadata"][:]

    if episodes_per_task is None:
        # Load all episodes
        episodes_list = []
        for i, metadata in enumerate(all_episode_metadata):
            start_idx = metadata["start_idx"]
            end_idx = metadata["end_idx"]

            # Get action deltas directly
            action_deltas = all_actions[start_idx:end_idx]  # [T, 7]

            episodes_list.append({
                "episode_idx": metadata["episode_idx"],
                "task_id": metadata["task_id"],
                "actions": action_deltas,  # [T, 7] - action deltas
                "success": metadata["success"],
            })
    else:
        # Balance episodes per task to match GT distribution
        print(f"Balancing model episodes to match GT distribution...")
        episodes_list = []

        for task_id, num_episodes_needed in sorted(episodes_per_task.items()):
            # Find all episodes for this task
            task_mask = all_episode_metadata["task_id"] == task_id
            task_episodes_indices = np.where(task_mask)[0]

            if len(task_episodes_indices) == 0:
                print(f"  Warning: Task {task_id} not in model data, skipping")
                continue

            # Take first num_episodes_needed for this task
            selected_indices = task_episodes_indices[:num_episodes_needed]

            if len(selected_indices) < num_episodes_needed:
                print(f"  Warning: Task {task_id} has only {len(selected_indices)} episodes, need {num_episodes_needed}")

            # Add episodes for this task
            for ep_idx in selected_indices:
                metadata = all_episode_metadata[ep_idx]
                start_idx = metadata["start_idx"]
                end_idx = metadata["end_idx"]

                # Get action deltas directly
                action_deltas = all_actions[start_idx:end_idx]  # [T, 7]

                episodes_list.append({
                    "episode_idx": metadata["episode_idx"],
                    "task_id": metadata["task_id"],
                    "actions": action_deltas,  # [T, 7] - action deltas
                    "success": metadata["success"],
                })

            print(f"  Task {task_id}: selected {len(selected_indices)} episodes")

    print(f"Loaded {len(episodes_list)} model episodes total")
    return episodes_list


def plot_action_histograms(gt_episodes, model_episodes, vanilla_episodes=None, save_path=None):
    """
    Create histogram plots comparing GT, model, and vanilla model actions across all dimensions.

    Args:
        gt_episodes: List of ground truth episode dicts (already balanced)
        model_episodes: List of model rollout episode dicts (already balanced)
        vanilla_episodes: Optional list of vanilla model rollout episode dicts (already balanced)
        save_path: Path to save the figure
    """
    n_models = 2 if vanilla_episodes is None else 3
    print(f"\nProcessing {len(gt_episodes)} GT episodes and {len(model_episodes)} model episodes" +
          (f" and {len(vanilla_episodes)} vanilla episodes..." if vanilla_episodes else "..."))

    # Collect all actions
    gt_actions_list = []
    model_actions_list = []
    vanilla_actions_list = [] if vanilla_episodes else None

    print("Collecting actions from episodes...")
    for idx in tqdm(range(len(gt_episodes))):
        gt_actions_list.append(gt_episodes[idx]["actions"])
        model_actions_list.append(model_episodes[idx]["actions"])
        if vanilla_episodes:
            vanilla_actions_list.append(vanilla_episodes[idx]["actions"])

    # Concatenate all actions
    gt_actions = np.vstack(gt_actions_list)  # [N, 7]
    model_actions = np.vstack(model_actions_list)  # [M, 7]
    vanilla_actions = np.vstack(vanilla_actions_list) if vanilla_episodes else None  # [V, 7]

    print(f"\nGT actions: {gt_actions.shape}")
    print(f"Model actions: {model_actions.shape}")
    if vanilla_episodes:
        print(f"Vanilla actions: {vanilla_actions.shape}")

    # Action dimension labels
    action_labels = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'dgripper']

    # Create figure with 7 subplots (one per action dimension)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    print("\nPlotting histograms...")
    for i, (ax, label) in enumerate(zip(axes[:7], action_labels)):
        # Extract data for this dimension
        gt_data = gt_actions[:, i]
        model_data = model_actions[:, i]

        # Compute shared bins for all distributions
        if vanilla_episodes:
            vanilla_data = vanilla_actions[:, i]
            all_data = np.concatenate([gt_data, model_data, vanilla_data])
        else:
            all_data = np.concatenate([gt_data, model_data])
        bins = np.linspace(all_data.min(), all_data.max(), 100)

        # Plot histograms with transparency
        ax.hist(gt_data, bins=bins, alpha=0.5, label='GT', color='blue',
                density=True, edgecolor='black', linewidth=0.5)
        ax.hist(model_data, bins=bins, alpha=0.5, label='Model', color='red',
                density=True, edgecolor='black', linewidth=0.5)
        if vanilla_episodes:
            ax.hist(vanilla_data, bins=bins, alpha=0.5, label='Vanilla', color='green',
                    density=True, edgecolor='black', linewidth=0.5)

        # Compute and display statistics
        gt_mean = np.mean(gt_data)
        model_mean = np.mean(model_data)
        gt_std = np.std(gt_data)
        model_std = np.std(model_data)

        # Add statistics to plot
        if vanilla_episodes:
            vanilla_mean = np.mean(vanilla_data)
            vanilla_std = np.std(vanilla_data)
            stats_text = (
                f"GT: μ={gt_mean:.4f}, σ={gt_std:.4f}\n"
                f"Model: μ={model_mean:.4f}, σ={model_std:.4f}\n"
                f"Vanilla: μ={vanilla_mean:.4f}, σ={vanilla_std:.4f}"
            )
        else:
            stats_text = (
                f"GT: μ={gt_mean:.4f}, σ={gt_std:.4f}\n"
                f"Model: μ={model_mean:.4f}, σ={model_std:.4f}"
            )

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(f'{label} value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Histogram: {label}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    # Remove extra subplots
    for i in range(7, 9):
        fig.delaxes(axes[i])

    # Title with appropriate info
    title_parts = [f'{len(gt_episodes)} balanced episodes', f'{len(gt_actions)} GT timesteps',
                   f'{len(model_actions)} model timesteps']
    if vanilla_episodes:
        title_parts.append(f'{len(vanilla_actions)} vanilla timesteps')

    plt.suptitle(
        f'Action Distribution Comparison: Histograms\n'
        f'({", ".join(title_parts)})',
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        # Create parent directory if it doesn't exist
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")

    plt.close()


def reconstruct_eef_from_actions(initial_position, initial_orientation, actions):
    """
    Reconstruct EEF poses by integrating action deltas.

    Args:
        initial_position: [3] array with [x, y, z]
        initial_orientation: [3] array with [roll, pitch, yaw]
        actions: [T, 7] array of deltas [dx, dy, dz, droll, dpitch, dyaw, dgripper]

    Returns:
        eef_poses: [T+1, 6] array of [x, y, z, roll, pitch, yaw] (excluding gripper)
    """
    poses = []
    current_pos = initial_position.copy()
    current_orient = initial_orientation.copy()

    # Add initial pose
    poses.append(np.concatenate([current_pos, current_orient]))

    # Integrate deltas
    for action in actions:
        current_pos = current_pos + action[:3]  # dx, dy, dz
        current_orient = current_orient + action[3:6]  # droll, dpitch, dyaw
        poses.append(np.concatenate([current_pos, current_orient]))

    return np.array(poses)  # [T+1, 6]


def plot_umap_comparison(gt_episodes, model_episodes, vanilla_episodes=None, save_path=None):
    """
    Create UMAP visualization comparing GT, model, and vanilla model action deltas.

    Args:
        gt_episodes: List of ground truth episode dicts (already balanced)
        model_episodes: List of model rollout episode dicts (already balanced)
        vanilla_episodes: Optional list of vanilla model rollout episode dicts (already balanced)
        save_path: Path to save the figure
    """
    print(f"\nProcessing {len(gt_episodes)} GT episodes and {len(model_episodes)} model episodes" +
          (f" and {len(vanilla_episodes)} vanilla episodes..." if vanilla_episodes else "..."))

    # Verify lengths match
    n_episodes = len(gt_episodes)
    if len(model_episodes) != n_episodes:
        print(f"Warning: model has {len(model_episodes)} episodes, expected {n_episodes}")
        n_episodes = min(n_episodes, len(model_episodes))
    if vanilla_episodes and len(vanilla_episodes) != n_episodes:
        print(f"Warning: vanilla has {len(vanilla_episodes)} episodes, expected {n_episodes}")
        n_episodes = min(n_episodes, len(vanilla_episodes))

    gt_episodes = gt_episodes[:n_episodes]
    model_episodes = model_episodes[:n_episodes]
    if vanilla_episodes:
        vanilla_episodes = vanilla_episodes[:n_episodes]

    gt_actions_list = []
    model_actions_list = []
    vanilla_actions_list = [] if vanilla_episodes else None

    print(f"Collecting action deltas from {n_episodes} episodes...")

    for i in tqdm(range(n_episodes)):
        gt_ep = gt_episodes[i]
        model_ep = model_episodes[i]

        # Collect action deltas directly
        gt_actions_list.append(gt_ep["actions"])  # [T, 7]
        model_actions_list.append(model_ep["actions"])  # [T, 7]

        if vanilla_episodes:
            vanilla_ep = vanilla_episodes[i]
            vanilla_actions_list.append(vanilla_ep["actions"])  # [T, 7]

    # Concatenate all episodes
    gt_actions = np.vstack(gt_actions_list)  # [N, 7]
    model_actions = np.vstack(model_actions_list)  # [M, 7]
    if vanilla_episodes:
        vanilla_actions = np.vstack(vanilla_actions_list)  # [V, 7]

    print(f"\nGT actions: {gt_actions.shape}")
    print(f"Model actions: {model_actions.shape}")
    if vanilla_episodes:
        print(f"Vanilla actions: {vanilla_actions.shape}")

    # Combine for joint UMAP
    if vanilla_episodes:
        all_actions = np.vstack([gt_actions, model_actions, vanilla_actions])
        n_gt = len(gt_actions)
        n_model = len(model_actions)
    else:
        all_actions = np.vstack([gt_actions, model_actions])
        n_gt = len(gt_actions)

    print(f"\nComputing UMAP embedding...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    embedding = reducer.fit_transform(all_actions)

    # Split back
    gt_embedding = embedding[:n_gt]
    if vanilla_episodes:
        model_embedding = embedding[n_gt:n_gt+n_model]
        vanilla_embedding = embedding[n_gt+n_model:]
    else:
        model_embedding = embedding[n_gt:]

    print(f"Creating plot...")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot GT actions
    ax.scatter(gt_embedding[:, 0], gt_embedding[:, 1],
               alpha=0.2, s=20, c='blue', label='GT Actions', edgecolors='none')

    # Plot model actions
    ax.scatter(model_embedding[:, 0], model_embedding[:, 1],
               alpha=0.2, s=20, c='red', label='Model Actions', edgecolors='none')

    # Plot vanilla actions if provided
    if vanilla_episodes:
        ax.scatter(vanilla_embedding[:, 0], vanilla_embedding[:, 1],
                   alpha=0.2, s=20, c='green', label='Vanilla Actions', edgecolors='none')

    ax.set_xlabel('UMAP 1', fontsize=14)
    ax.set_ylabel('UMAP 2', fontsize=14)

    title = 'Action Distribution: GT vs Model'
    if vanilla_episodes:
        title += ' vs Vanilla'
    title += f'\n({n_episodes} balanced episodes, {len(gt_actions)} GT timesteps)'

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12, markerscale=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        # Create parent directory if it doesn't exist
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")

    plt.close()


def plot_eef_position_planes(gt_episodes, model_episodes, vanilla_episodes=None, save_path=None):
    """
    Plot 3 scatter plots showing end-effector positions in XY, XZ, and YZ planes.

    Args:
        gt_episodes: List of ground truth episode dicts (already balanced)
        model_episodes: List of model rollout episode dicts (already balanced)
        vanilla_episodes: Optional list of vanilla model rollout episode dicts (already balanced)
        save_path: Path to save the figure
    """
    print(f"\nProcessing {len(gt_episodes)} GT episodes and {len(model_episodes)} model episodes" +
          (f" and {len(vanilla_episodes)} vanilla episodes..." if vanilla_episodes else "..."))

    # Verify lengths match
    n_episodes = len(gt_episodes)
    if len(model_episodes) != n_episodes:
        print(f"Warning: model has {len(model_episodes)} episodes, expected {n_episodes}")
        n_episodes = min(n_episodes, len(model_episodes))
    if vanilla_episodes and len(vanilla_episodes) != n_episodes:
        print(f"Warning: vanilla has {len(vanilla_episodes)} episodes, expected {n_episodes}")
        n_episodes = min(n_episodes, len(vanilla_episodes))

    gt_episodes = gt_episodes[:n_episodes]
    model_episodes = model_episodes[:n_episodes]
    if vanilla_episodes:
        vanilla_episodes = vanilla_episodes[:n_episodes]

    # Collect all EEF positions by integrating action deltas
    gt_positions_list = []
    model_positions_list = []
    vanilla_positions_list = [] if vanilla_episodes else None

    print(f"Reconstructing end-effector positions from {n_episodes} episodes...")

    for i in tqdm(range(n_episodes)):
        # GT positions
        gt_actions = gt_episodes[i]["actions"][:, :3]  # [T, 3] - only dx, dy, dz
        gt_cumulative = np.cumsum(gt_actions, axis=0)  # [T, 3] - integrate to get positions
        gt_positions_list.append(gt_cumulative)

        # Model positions
        model_actions = model_episodes[i]["actions"][:, :3]  # [T, 3]
        model_cumulative = np.cumsum(model_actions, axis=0)  # [T, 3]
        model_positions_list.append(model_cumulative)

        # Vanilla positions
        if vanilla_episodes:
            vanilla_actions = vanilla_episodes[i]["actions"][:, :3]  # [T, 3]
            vanilla_cumulative = np.cumsum(vanilla_actions, axis=0)  # [T, 3]
            vanilla_positions_list.append(vanilla_cumulative)

    # Concatenate all positions
    gt_positions = np.vstack(gt_positions_list)  # [N, 3]
    model_positions = np.vstack(model_positions_list)  # [M, 3]
    if vanilla_episodes:
        vanilla_positions = np.vstack(vanilla_positions_list)  # [V, 3]

    print(f"\nGT positions: {gt_positions.shape}")
    print(f"Model positions: {model_positions.shape}")
    if vanilla_episodes:
        print(f"Vanilla positions: {vanilla_positions.shape}")

    print(f"Creating plane plots...")

    # Create figure with 3 subplots (XY, XZ, YZ planes)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define planes: [(x_idx, y_idx, xlabel, ylabel, title), ...]
    planes = [
        (0, 1, 'X Position', 'Y Position', 'XY Plane'),
        (0, 2, 'X Position', 'Z Position', 'XZ Plane'),
        (1, 2, 'Y Position', 'Z Position', 'YZ Plane'),
    ]

    for ax, (x_idx, y_idx, xlabel, ylabel, title) in zip(axes, planes):
        # Plot GT positions
        ax.scatter(gt_positions[:, x_idx], gt_positions[:, y_idx],
                   alpha=0.3, s=10, c='blue', label='GT', edgecolors='none')

        # Plot model positions
        ax.scatter(model_positions[:, x_idx], model_positions[:, y_idx],
                   alpha=0.3, s=10, c='red', label='Model', edgecolors='none')

        # Plot vanilla positions if provided
        if vanilla_episodes:
            ax.scatter(vanilla_positions[:, x_idx], vanilla_positions[:, y_idx],
                       alpha=0.3, s=10, c='green', label='Vanilla', edgecolors='none')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10, markerscale=2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    # Overall title
    title = 'End-Effector Position Distribution: GT vs Model'
    if vanilla_episodes:
        title += ' vs Vanilla'
    title += f'\n({n_episodes} balanced episodes, {len(gt_positions)} GT timesteps)'

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        # Create parent directory if it doesn't exist
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize action/pose distributions with histograms or UMAP")
    parser.add_argument(
        "--model_data_path",
        type=str,
        required=True,
        help="Path to model rollout H5 file"
    )
    parser.add_argument(
        "--vanilla_data_path",
        type=str,
        default=None,
        help="Path to vanilla model rollout H5 file (optional, for 3-way comparison)"
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="aopolin-lv/libero_spatial_no_noops_lerobot_v21",
        help="HuggingFace dataset name for ground truth"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to visualize"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["histogram", "umap", "planes"],
        default="histogram",
        help="Visualization mode: 'histogram' for overlaid histograms, 'umap' for action UMAP, 'planes' for EEF position planes (XY, XZ, YZ)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save visualization"
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Optional: Filter to visualize only episodes from a specific task (1-10 for libero_spatial)"
    )

    args = parser.parse_args()

    # Load ground truth data FIRST to determine episode distribution
    print(f"Loading ground truth data from HuggingFace...")
    gt_episodes, episodes_per_task = load_ground_truth_from_hf(
        dataset_name=args.hf_dataset,
        max_episodes=args.num_episodes,
        load_states=False  # No longer need states - using action deltas directly
    )

    # Load model data matching the GT distribution
    print(f"\nLoading model rollout data from {args.model_data_path}...")
    model_episodes = load_model_rollout_data(args.model_data_path, episodes_per_task=episodes_per_task)

    # Load vanilla model data if provided
    vanilla_episodes = None
    if args.vanilla_data_path:
        print(f"\nLoading vanilla model rollout data from {args.vanilla_data_path}...")
        vanilla_episodes = load_model_rollout_data(args.vanilla_data_path, episodes_per_task=episodes_per_task)

    # Filter by task_id if specified
    if args.task_id is not None:
        print(f"\nFiltering episodes for task_id={args.task_id}...")
        gt_episodes = [ep for ep in gt_episodes if ep["task_id"] == args.task_id]
        model_episodes = [ep for ep in model_episodes if ep["task_id"] == args.task_id]
        if vanilla_episodes:
            vanilla_episodes = [ep for ep in vanilla_episodes if ep["task_id"] == args.task_id]

        print(f"After filtering: {len(gt_episodes)} GT episodes, {len(model_episodes)} model episodes" +
              (f", {len(vanilla_episodes)} vanilla episodes" if vanilla_episodes else ""))

        if len(gt_episodes) == 0:
            print(f"Error: No episodes found for task_id={args.task_id}")
            return

    # Visualize based on mode
    if args.mode == "histogram":
        save_path = args.save_path or "action_histograms_comparison.png"
        plot_action_histograms(
            gt_episodes,
            model_episodes,
            vanilla_episodes=vanilla_episodes,
            save_path=save_path
        )
    elif args.mode == "umap":
        save_path = args.save_path or "eef_umap_comparison.png"
        plot_umap_comparison(
            gt_episodes,
            model_episodes,
            vanilla_episodes=vanilla_episodes,
            save_path=save_path
        )
    elif args.mode == "planes":
        save_path = args.save_path or "eef_position_planes.png"
        plot_eef_position_planes(
            gt_episodes,
            model_episodes,
            vanilla_episodes=vanilla_episodes,
            save_path=save_path
        )


if __name__ == "__main__":
    main()
