"""
Evaluation script that collects actions and joint states from LIBERO rollouts using OpenVLA.

This script evaluates an OpenVLA policy on LIBERO tasks and saves:
1. Actions taken by the model at each timestep
2. Joint positions of the robot (7-DOF Franka Emika Panda arm)

The data is saved efficiently in a single compressed HDF5 file to minimize storage.

Usage:
    # Distilled model
    python scripts/eval_action_collection_openvla.py \
        --pretrained_checkpoint /path/to/distilled/checkpoint \
        --task_suite_name libero_object \
        --output_filename actions_libero_object_distilled.h5

    # Vanilla (baseline) model
    python scripts/eval_action_collection_openvla.py \
        --pretrained_checkpoint /path/to/vanilla/checkpoint \
        --task_suite_name libero_object \
        --output_filename actions_libero_object_vanilla.h5 \
        --is_baseline True
"""

import dataclasses
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import draccus
import h5py
import numpy as np
import tensorflow as tf
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# Add openvla experiments path
sys.path.insert(0, "/home/elias/Thesis/src/openvla")
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
ACTION_DIM = 7
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


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


def quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def resize_image(img, resize_size):
    """Resize image using the same scheme as RLDS dataset wrapper."""
    assert isinstance(resize_size, tuple)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # Rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def crop_and_resize(image, crop_scale, batch_size):
    """Center-crops an image and resizes back to original size."""
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = crop_and_resize(image, crop_scale, batch_size)

        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action


def load_vla_model(pretrained_checkpoint, load_in_8bit=False, load_in_4bit=False):
    """Loads and returns a VLA model from checkpoint."""
    print("[*] Instantiating Pretrained VLA model")

    # Register OpenVLA model to HF Auto Classes
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if not load_in_8bit and not load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats for action un-normalization
    dataset_statistics_path = os.path.join(pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print("WARNING: No local dataset_statistics.json file found for current checkpoint.")

    return vla


@dataclasses.dataclass
class Args:
    """Evaluation arguments for action collection with OpenVLA."""

    # --- Model arguments ---
    pretrained_checkpoint: Union[str, Path] = ""
    """Path to the pretrained OpenVLA checkpoint."""
    load_in_8bit: bool = False
    """Load with 8-bit quantization."""
    load_in_4bit: bool = False
    """Load with 4-bit quantization."""
    center_crop: bool = True
    """Center crop? (if trained w/ random crop image aug)."""
    is_baseline: bool = False
    """Whether this is a vanilla baseline model (no distillation)."""

    # --- LIBERO environment-specific parameters ---
    task_suite_name: str = "libero_object"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize."""
    num_trials_per_task: int = 10
    """Number of rollouts per task."""

    # --- Output arguments ---
    output_dir: str = "/home/elias/Thesis/action_data"
    """Directory to save action collection data."""
    output_filename: Optional[str] = None
    """Output filename. If None, auto-generated from config."""

    # --- Evaluation arguments ---
    seed: int = 7
    """Random seed."""


@draccus.wrap()
def collect_actions(args: Args) -> None:
    """Main evaluation loop that collects actions and joint states."""

    assert args.pretrained_checkpoint, "pretrained_checkpoint must be provided!"
    assert not (args.load_in_8bit and args.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename from config
    if args.output_filename is None:
        model_type = "vanilla" if args.is_baseline else "distilled"
        args.output_filename = f"actions_{args.task_suite_name}_{model_type}.h5"

    output_path = output_dir / args.output_filename
    print(f"Will save action data to: {output_path}")

    # Set random seeds
    set_seed_everywhere(args.seed)

    # --- Load Model ---
    print("--- Loading OpenVLA model ---")
    print(f"Loading model from {args.pretrained_checkpoint}")
    model = load_vla_model(
        args.pretrained_checkpoint,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    # Set action un-normalization key
    unnorm_key = f"openvla_{args.task_suite_name}" if args.task_suite_name == "libero_spatial" else "openvla_libero_spatial"

    # Check for _no_noops variant
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    print(f"Using unnorm_key: {unnorm_key}")
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA norm_stats!"

    # Get processor
    processor = AutoProcessor.from_pretrained(args.pretrained_checkpoint, trust_remote_code=True)

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
    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_map.get(args.task_suite_name, 520)

    print("Starting data collection...")

    # Accumulate data in memory
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
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(args.seed)

        # Run episodes for this task
        for episode_idx in tqdm(
            range(min(args.num_trials_per_task, len(initial_states))),
            desc=f"Task {task_id}: {task.language}",
            leave=False,
        ):
            # Reset environment
            env.reset()

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
                    joint_pos = obs.get("robot0_joint_pos", None)

                    if joint_pos is None:
                        try:
                            joint_pos = env.sim.data.qpos[:7]
                        except:
                            print(f"Warning: Could not extract joint positions at step {t}")
                            joint_pos = np.zeros(7)

                    # Get preprocessed image
                    img = get_libero_image(obs, LIBERO_ENV_RESOLUTION)

                    # Prepare observation dict for OpenVLA
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                    }

                    # Query model to get action
                    with torch.inference_mode():
                        action = get_vla_action(
                            model, processor, observation, task_description,
                            unnorm_key=unnorm_key, center_crop=args.center_crop
                        )

                    # Process action for environment
                    action = normalize_gripper_action(action, binarize=True)

                    # Store action and joint position
                    episode_actions.append(action.copy())
                    episode_joint_positions.append(joint_pos.copy())

                    # Execute action
                    obs, _, done, _ = env.step(action.tolist())

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

        env.close()

    # Convert to numpy arrays
    all_actions = np.array(all_actions, dtype=np.float32)
    all_joint_positions = np.array(all_joint_positions, dtype=np.float32)

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
            compression_opts=6,
        )
        f.create_dataset(
            "joint_positions",
            data=all_joint_positions,
            compression="gzip",
            compression_opts=6,
        )

        # Store episode metadata
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

        # Store task descriptions separately
        task_descriptions = [m["task_description"] for m in episode_metadata]
        f.create_dataset(
            "task_descriptions",
            data=np.array(task_descriptions, dtype=h5py.string_dtype()),
        )

        # Store metadata attributes
        f.attrs["task_suite_name"] = args.task_suite_name
        f.attrs["pretrained_checkpoint"] = str(args.pretrained_checkpoint)
        f.attrs["num_episodes"] = total_episodes
        f.attrs["num_timesteps"] = total_timesteps
        f.attrs["seed"] = args.seed
        f.attrs["is_baseline"] = args.is_baseline

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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    collect_actions()
