"""
This script demonstrates how to evaluate a pretrained smolVLA policy on the LIBERO benchmark.
https://github.com/huggingface/lerobot/issues/1316
"""

import collections
import dataclasses
import logging
import math
import pathlib
import os
from pathlib import Path
from datetime import datetime

import cv2
import draccus
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' to enable logging.")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


@dataclasses.dataclass
class Args:
    """
    Evaluation arguments for smolVLA on LIBERO.
    """

    # --- Hugging Face arguments ---
    policy_path: str = "outputs/train/libero_smolvla/checkpoints/070000/pretrained_model/"
    """Path to the pretrained policy on the Hugging Face Hub or local directory."""

    # --- LIBERO environment-specific parameters ---
    task_suite_name: str = "libero_spatial"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 50
    """Number of rollouts per task."""

    # --- Evaluation arguments ---
    video_out_path: str = None
    """Path to save videos. If None, will be auto-generated based on config."""
    device: str = "cuda"
    """Device to use for evaluation."""

    # --- Logging arguments ---
    use_wandb: bool = False
    """Whether to log results to Weights & Biases."""
    wandb_project: str = "smolvla-libero-eval"
    """W&B project name."""
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    """W&B entity name."""
    run_id_note: str = None
    """Optional note to add to run ID."""

    is_baseline: bool = False
    """Whether this is a vanilla baseline model (no distillation)."""

    seed: int = 7
    """Random Seed (for reproducibility) - will be overridden for multi-seed eval"""


@draccus.wrap()
def eval_libero(args: Args) -> None:
    # Load training config from checkpoint directory if it exists (skip for baseline)
    training_config = {}
    if not args.is_baseline:
        # Try to find training_config.yaml in checkpoint directory
        # Handle both cases: policy_path = "checkpoints/070000/pretrained_model/" or "checkpoints/070000/"
        policy_path = Path(args.policy_path)

        # If policy_path ends with pretrained_model/, go up one level to checkpoint dir
        if policy_path.name == "pretrained_model":
            checkpoint_dir = policy_path.parent
        else:
            checkpoint_dir = policy_path

        training_config_path = checkpoint_dir / "training_config.yaml"
        if training_config_path.exists():
            import yaml
            with open(training_config_path, 'r') as f:
                training_config = yaml.safe_load(f)
            print(f"Loaded training config from {training_config_path}")
        else:
            print(f"Warning: No training config found at {training_config_path}")
            print(f"  Searched in: {checkpoint_dir}")

    # Create descriptive name from training config
    if args.is_baseline:
        config_name = "finetune_full"
    elif training_config:
        # Extract key parameters from training config
        loss_type = training_config.get('distill_loss_type', 'unknown')
        if loss_type == 'infonce':
            temp = training_config.get('infonce_temperature', 0.0)
        else:
            temp = training_config.get('distill_temperature', 0.0)
        weight = training_config.get('distill_weight', 0.0)

        config_name = f"distill_{loss_type}_temp_{temp}_weight_{weight}_full"
    else:
        config_name = f"EVAL-{args.task_suite_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Set up logging directories
    log_dir = Path("/home/elias/Thesis/logs/smolvla/real")
    os.makedirs(log_dir, exist_ok=True)
    local_log_filepath = log_dir / f"{config_name}.txt"
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Create rollout directory with same name
    rollout_base_dir = Path("/home/elias/Thesis/rollouts/smolvla/real")
    rollout_dir = rollout_base_dir / config_name
    os.makedirs(rollout_dir, exist_ok=True)
    print(f"Saving rollouts to: {rollout_dir}")

    # Override video_out_path if not specified
    if args.video_out_path is None:
        args.video_out_path = str(rollout_dir)

    run_id = config_name

    # Initialize Weights & Biases logging
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for logging. Install with 'pip install wandb'")
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_id,
        )

        # Log training config parameters to wandb
        if training_config:
            wandb.config.update({
                "training_config": training_config,
                "checkpoint_path": str(args.policy_path),
                "task_suite_name": args.task_suite_name,
                "num_trials_per_task": args.num_trials_per_task,
            })
            print("Logged training config to wandb")

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
    logging.info(f"Task suite: {args.task_suite_name}")
    log_file.write(f"Task suite: {args.task_suite_name}\n")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        # Fallback for custom task suites
        max_steps = 520

    # --- Evaluation Loop - run with multiple seeds and average ---
    seeds = [7, 20, 100]
    all_seeds_results = {}  # {task_description: [success_rates across seeds]}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n{'='*80}")
        print(f"Running evaluation with seed: {seed}")
        print(f"{'='*80}\n")
        log_file.write(f"\n{'='*80}\n")
        log_file.write(f"Seed: {seed}\n")
        log_file.write(f"{'='*80}\n\n")

        total_episodes, total_successes = 0, 0

        for task_id in tqdm(range(num_tasks_in_suite), desc=f"Tasks (Seed {seed})"):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)

            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm(
                range(min(args.num_trials_per_task, len(initial_states))),
                desc=f"Task {task_id}: {task.language}",
                leave=False,
            ):
                logging.info(f"\nTask: {task_description}")
                log_file.write(f"\nTask: {task_description}\n")

                # Reset environment and policy
                env.reset()
                policy.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                for _ in range(args.num_steps_wait):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

                # Setup
                t = 0
                frames = []
                done = False

                logging.info(f"Starting episode {task_episodes+1}...")
                log_file.write(f"Starting episode {task_episodes+1}...\n")

                while t < max_steps:
                    try:
                        # Get preprocessed image
                        # IMPORTANT: rotate 180 degrees to match train preprocessing
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        frames.append(agentview_image)

                        # Prepare observations dict
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

                        # Execute action in environment
                        obs, _, done, _ = env.step(action)
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        logging.error(f"Caught exception: {e}")
                        log_file.write(f"Caught exception: {e}\n")
                        break

                task_episodes += 1
                total_episodes += 1

                # Save a replay video of the episode
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_").replace("/", "_")
                video_path = (
                    pathlib.Path(args.video_out_path) / f"seed_{seed}_task_{task_id}_episode_{episode_idx}_{task_segment}_{suffix}.mp4"
                )
                fps = 30
                writer = imageio.get_writer(video_path, fps=fps)

                for image in frames:
                    writer.append_data(image)
                writer.close()
                logging.info(f"Saved video to {video_path}")

                # Log current results
                print(f"Success: {done}")
                print(f"# episodes completed so far: {total_episodes}")
                print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                log_file.write(f"Success: {done}\n")
                log_file.write(f"# episodes completed so far: {total_episodes}\n")
                log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
                log_file.flush()

            # Log per-task results for this seed
            task_success_rate = float(task_successes) / float(task_episodes)
            print(f"Current task success rate: {task_success_rate:.3f}")
            print(f"Current total success rate: {float(total_successes) / float(total_episodes):.3f}")
            log_file.write(f"Current task success rate: {task_success_rate:.3f}\n")
            log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes):.3f}\n")
            log_file.flush()

            # Store results for averaging across seeds
            if task_description not in all_seeds_results:
                all_seeds_results[task_description] = []
            all_seeds_results[task_description].append(task_success_rate)

    # Compute averages across seeds
    print(f"\n{'='*80}")
    print(f"AVERAGED RESULTS ACROSS {len(seeds)} SEEDS: {seeds}")
    print(f"{'='*80}\n")
    log_file.write(f"\n{'='*80}\n")
    log_file.write(f"AVERAGED RESULTS ACROSS {len(seeds)} SEEDS: {seeds}\n")
    log_file.write(f"{'='*80}\n\n")

    task_avg_results = {}
    all_task_success_rates = []
    for task_desc, success_rates in all_seeds_results.items():
        avg_success = np.mean(success_rates)
        std_success = np.std(success_rates)
        task_avg_results[task_desc] = avg_success
        all_task_success_rates.append(avg_success)

        print(f"Task: {task_desc}")
        print(f"  Success rates per seed: {success_rates}")
        print(f"  Average: {avg_success:.3f} ± {std_success:.3f}\n")
        log_file.write(f"Task: {task_desc}\n")
        log_file.write(f"  Success rates per seed: {success_rates}\n")
        log_file.write(f"  Average: {avg_success:.3f} ± {std_success:.3f}\n\n")

    # Overall average
    overall_avg = np.mean(all_task_success_rates)
    overall_std = np.std(all_task_success_rates)
    print(f"Overall average success rate: {overall_avg:.3f} ± {overall_std:.3f}")
    log_file.write(f"Overall average success rate: {overall_avg:.3f} ± {overall_std:.3f}\n")

    # Save local log file
    log_file.close()

    # Push averaged metrics to wandb
    if args.use_wandb:
        # Log per-task averaged results
        for task_desc, avg_success in task_avg_results.items():
            wandb.log({
                f"success_rate/{task_desc}": avg_success,
                f"num_trials_per_task": args.num_trials_per_task,
                f"num_seeds": len(seeds),
            })

        # Log overall average
        wandb.log({
            "success_rate/total": overall_avg,
            "num_seeds": len(seeds),
        })
        wandb.save(str(local_log_filepath))

    logging.info("--- Evaluation finished ---")
    print(f"\nFinal Results:")
    print(f"  Overall average success rate: {overall_avg:.3f} ± {overall_std:.3f}")
    print(f"  Log saved to: {local_log_filepath}")
    print(f"  Videos saved to: {rollout_dir}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("evaluation_log.txt"),
            logging.StreamHandler()  # Optional: keeps logging in the terminal too
        ]
    )
    eval_libero()
