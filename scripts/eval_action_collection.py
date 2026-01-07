"""
Evaluation script that collects actions and joint states from LIBERO rollouts.

This script evaluates a SmolVLA policy on LIBERO tasks and saves:
1. Actions taken by the model at each timestep
2. Joint positions of the robot (7-DOF Franka Emika Panda arm)

The data is saved efficiently in a single compressed HDF5 file to minimize storage.
"""

import dataclasses
import logging
import math
import os
from pathlib import Path

import cv2
import draccus
import h5py
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def normalize_gripper_action(action, binarize=True):
    """Changes gripper action from [0,1] to [-1,+1]."""
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """Flips the sign of the gripper action."""
    action[..., -1] = action[..., -1] * -1.0
    return action


@dataclasses.dataclass
class Args:
    """Evaluation arguments for action collection."""

    # --- Hugging Face arguments ---
    policy_path: str = "outputs/train/libero_smolvla/checkpoints/070000/pretrained_model/"
    """Path to the pretrained policy."""

    # --- LIBERO environment-specific parameters ---
    task_suite_name: str = "libero_spatial"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize."""
    num_trials_per_task: int = 50
    """Number of rollouts per task."""

    # --- Output arguments ---
    output_dir: str = "/home/elias/Thesis/action_data"
    """Directory to save action collection data."""
    output_filename: str = None
    """Output filename. If None, auto-generated from config."""

    # --- Evaluation arguments ---
    device: str = "cuda"
    """Device to use for evaluation."""
    seed: int = 7
    """Random seed."""


@draccus.wrap()
def collect_actions(args: Args) -> None:
    """Main evaluation loop that collects actions and joint states."""

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename from config
    policy_name = Path(args.policy_path).parent.parent.name  # e.g., "070000"
    if args.output_filename is None:
        args.output_filename = f"actions_{args.task_suite_name}_{policy_name}_seed{args.seed}_vanilla.h5"

    output_path = output_dir / args.output_filename
    print(f"Will save action data to: {output_path}")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load Policy ---
    print("--- Loading policy ---")
    print(f"Loading policy from {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.config.n_action_steps = 1
    policy.to(args.device)
    policy.eval()

    # --- Initialize LIBERO task suite ---
    benchmark_dict = benchmark.get_benchmark_dict()
    try:
        task_suite = benchmark_dict[args.task_suite_name]()
    except KeyError:
        raise ValueError(
            f"Unknown task suite: {args.task_suite_name}. "
            f"Available options are: {list(benchmark_dict.keys())}"
        )
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {args.task_suite_name} ({num_tasks_in_suite} tasks)")

    # Determine max steps based on task suite
    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        max_steps = 520

    # Prepare HDF5 file for efficient storage
    # We'll store data in a flat structure for efficiency:
    # - actions: [N, 7] where N is total timesteps across all episodes
    # - joint_positions: [N, 7] where N is total timesteps across all episodes
    # - episode_metadata: dataset containing episode start/end indices, task info, success

    print("Starting data collection...")

    # Accumulate data in memory first (more efficient than incremental HDF5 writes)
    all_actions = []
    all_joint_positions = []
    episode_metadata = []

    total_timesteps = 0
    total_episodes = 0

    # --- Evaluation Loop ---
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):
        # Get task
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize environment
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Run episodes for this task
        for episode_idx in tqdm(
            range(min(args.num_trials_per_task, len(initial_states))),
            desc=f"Task {task_id}: {task.language}",
            leave=False,
        ):
            # Reset environment and policy
            env.reset()
            policy.reset()

            # Set initial state
            obs = env.set_init_state(initial_states[episode_idx])

            # Wait for objects to stabilize
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            # Track episode data
            episode_actions = []
            episode_joint_positions = []
            episode_start_idx = total_timesteps

            t = 0
            done = False

            while t < max_steps:
                try:
                    # Extract robot joint positions (7-DOF for Franka Emika Panda)
                    # Note: robot0_joint_pos is available in LIBERO observations
                    joint_pos = obs.get("robot0_joint_pos", None)

                    if joint_pos is None:
                        # Fallback: some environments might not expose this directly
                        # Try to access via sim data
                        try:
                            joint_pos = env.sim.data.qpos[:7]  # First 7 qpos are robot joints
                        except:
                            print(f"Warning: Could not extract joint positions at step {t}")
                            joint_pos = np.zeros(7)

                    # Get preprocessed images
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

                    # Prepare observation dict
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )
                    observation = {
                        "observation.images.image": torch.from_numpy(agentview_image / 255.0)
                            .permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                        "observation.images.wrist_image": torch.from_numpy(wrist_img / 255.0)
                            .permute(2, 0, 1).to(torch.float32).to(args.device).unsqueeze(0),
                        "observation.state": torch.from_numpy(state)
                            .to(torch.float32).to(args.device).unsqueeze(0),
                        "task": task_description,
                    }

                    # Query model to get action
                    with torch.inference_mode():
                        action_tensor = policy.select_action(observation)
                    action = action_tensor.cpu().numpy()[0]
                    action = normalize_gripper_action(action, binarize=False)
                    action = invert_gripper_action(action)

                    # Store action and joint position
                    episode_actions.append(action.copy())
                    episode_joint_positions.append(joint_pos.copy())

                    # Execute action
                    obs, _, done, _ = env.step(action)

                    if done:
                        break
                    t += 1

                except Exception as e:
                    print(f"Error at task {task_id}, episode {episode_idx}, step {t}: {e}")
                    break

            # Store episode data
            episode_end_idx = total_timesteps + len(episode_actions)

            episode_metadata.append({
                "task_id": task_id,
                "task_description": task_description,
                "episode_idx": episode_idx,
                "start_idx": episode_start_idx,
                "end_idx": episode_end_idx,
                "num_steps": len(episode_actions),
                "success": done,
                "seed": args.seed,
            })

            # Accumulate data
            all_actions.extend(episode_actions)
            all_joint_positions.extend(episode_joint_positions)

            total_timesteps += len(episode_actions)
            total_episodes += 1

    # Convert to numpy arrays
    all_actions = np.array(all_actions, dtype=np.float32)  # [N, 7]
    all_joint_positions = np.array(all_joint_positions, dtype=np.float32)  # [N, 7]

    print(f"\nData collection complete!")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Actions shape: {all_actions.shape}")
    print(f"  Joint positions shape: {all_joint_positions.shape}")

    # Compute storage size estimate
    actions_size_mb = all_actions.nbytes / (1024 * 1024)
    joints_size_mb = all_joint_positions.nbytes / (1024 * 1024)
    total_size_mb = actions_size_mb + joints_size_mb

    print(f"\nStorage requirements:")
    print(f"  Actions: {actions_size_mb:.2f} MB")
    print(f"  Joint positions: {joints_size_mb:.2f} MB")
    print(f"  Total (uncompressed): {total_size_mb:.2f} MB")

    # Save to HDF5 with compression
    print(f"\nSaving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        # Store actions and joint positions with gzip compression
        f.create_dataset(
            "actions",
            data=all_actions,
            compression="gzip",
            compression_opts=6,  # Balanced compression level
        )
        f.create_dataset(
            "joint_positions",
            data=all_joint_positions,
            compression="gzip",
            compression_opts=6,
        )

        # Store episode metadata
        # Convert to structured array for efficient storage
        metadata_dtype = np.dtype([
            ("task_id", np.int32),
            ("episode_idx", np.int32),
            ("start_idx", np.int64),
            ("end_idx", np.int64),
            ("num_steps", np.int32),
            ("success", np.bool_),
            ("seed", np.int32),
        ])

        metadata_array = np.array(
            [
                (m["task_id"], m["episode_idx"], m["start_idx"], m["end_idx"],
                 m["num_steps"], m["success"], m["seed"])
                for m in episode_metadata
            ],
            dtype=metadata_dtype
        )

        f.create_dataset("episode_metadata", data=metadata_array)

        # Store task descriptions separately (variable length strings)
        task_descriptions = [m["task_description"] for m in episode_metadata]
        f.create_dataset(
            "task_descriptions",
            data=np.array(task_descriptions, dtype=h5py.string_dtype()),
        )

        # Store metadata attributes
        f.attrs["task_suite_name"] = args.task_suite_name
        f.attrs["policy_path"] = args.policy_path
        f.attrs["num_episodes"] = total_episodes
        f.attrs["num_timesteps"] = total_timesteps
        f.attrs["seed"] = args.seed

    # Check actual file size
    actual_size_mb = output_path.stat().st_size / (1024 * 1024)
    compression_ratio = total_size_mb / actual_size_mb if actual_size_mb > 0 else 0

    print(f"\nFile saved successfully!")
    print(f"  Compressed size: {actual_size_mb:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Location: {output_path}")

    # Print summary statistics
    print(f"\nSummary statistics:")
    success_rate = np.mean([m["success"] for m in episode_metadata])
    avg_episode_length = np.mean([m["num_steps"] for m in episode_metadata])
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Average episode length: {avg_episode_length:.1f} steps")

    print("\nDone!")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment."""
    import pathlib
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    collect_actions()
