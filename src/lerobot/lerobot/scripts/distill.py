#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Knowledge distillation training script for SmolVLA student from LAPA teacher.
"""

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

# Import SmolVLA distillation utilities
from lerobot.scripts.smolvla_distillation_utils import (
    AggregationMethod,
    StudentSequenceProjectionMLP,
    SimilarityMatrixDistillationLoss,
    InfoNCEDistillationLoss,
)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    # Distillation components
    teacher_hidden_states: dict = None,
    student_projection_mlp: nn.Module = None,
    distillation_loss_fn: nn.Module = None,
    distill_weight: float = 0.0,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    if student_projection_mlp is not None:
        student_projection_mlp.train()

    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        # Forward pass through student policy
        task_loss, output_dict = policy.forward(batch)

        # Compute action L1 loss for logging (accuracy-like metric for continuous actions)
        if "actions" in output_dict and "action" in batch:
            action_preds = output_dict["actions"]
            action_gt = batch["action"]
            action_loss = torch.nn.functional.l1_loss(action_preds, action_gt)
            output_dict["action_loss"] = action_loss.item()

        # Add distillation loss if enabled
        if teacher_hidden_states is not None and student_projection_mlp is not None and distillation_loss_fn is not None:
            # Extract student hidden states (scene understanding from SmolVLA VLM)
            if "prefix_out" not in output_dict:
                raise KeyError("prefix_out not found in output_dict! SmolVLA must expose prefix_out.")

            student_hidden_states = output_dict["prefix_out"]  # [batch, 149, 960]

            # Project student hidden states: [batch, 149, 960] → [batch, 4096]
            student_projected = student_projection_mlp(student_hidden_states)

            # Get teacher hidden states for this batch using efficient list indexing
            global_indices = batch.get("index")
            if global_indices is None:
                raise KeyError("Batch must contain 'index' field with global frame indices")

            # Gather teacher hidden states using list-based indexing (same as OpenVLA)
            # teacher_hidden_states is a tensor [num_samples, 4096] loaded from H5 into CPU RAM
            # We index directly using global_indices to get the corresponding teacher states
            teacher_batch_hidden_states = []
            for global_idx in global_indices:
                idx_int = int(global_idx.item())
                # Direct indexing since states are stored as list (0, 1, 2, ...)
                teacher_state = teacher_hidden_states[idx_int]
                teacher_batch_hidden_states.append(teacher_state)

            if len(teacher_batch_hidden_states) > 0:
                teacher_batch_hidden_states = torch.stack(teacher_batch_hidden_states, dim=0)  # [batch, 4096]
                # Move to GPU for loss computation
                teacher_batch_hidden_states = teacher_batch_hidden_states.to(device)

                # Compute distillation loss
                distill_loss = distillation_loss_fn(student_projected[:len(teacher_batch_hidden_states)], teacher_batch_hidden_states)

                # Combined loss
                loss = task_loss + distill_weight * distill_loss

                # Log separate losses
                output_dict["task_loss"] = task_loss.item()
                output_dict["distill_loss"] = distill_loss.item()
            else:
                loss = task_loss
                output_dict["task_loss"] = task_loss.item()
                output_dict["distill_loss"] = 0.0
        else:
            loss = task_loss

    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time

    # Add custom metrics to train_metrics for WandB logging
    if "action_loss" in output_dict:
        train_metrics.action_loss = output_dict["action_loss"]
    if "task_loss" in output_dict:
        train_metrics.task_loss = output_dict["task_loss"]
    if "distill_loss" in output_dict:
        train_metrics.distill_loss = output_dict["distill_loss"]

    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    # Distillation setup
    teacher_hidden_states = None
    teacher_global_indices = None
    student_projection_mlp = None
    distillation_loss_fn = None
    distill_weight = 0.0

    # Check if distillation config is provided
    if hasattr(cfg, 'distillation') and cfg.distillation is not None:
        logging.info("=== Distillation Configuration ===")
        distill_cfg = cfg.distillation

        # Load teacher hidden states from H5 file (same LAPA file used for OpenVLA)
        teacher_path = distill_cfg.teacher_hidden_states_path
        if teacher_path:
            logging.info(f"Loading teacher hidden states from {teacher_path}")
            logging.info("Loading from H5 file into CPU RAM (efficient list-based indexing)...")

            # Load from H5 file format: {"global_indices": [...], "hidden_states": [...]}
            teacher_dataset_file = h5py.File(str(teacher_path), "r")

            # Load entire dataset into CPU RAM as numpy arrays, then convert to torch tensors
            # This is done once at the start for fast batch lookups during training
            teacher_global_indices = teacher_dataset_file["global_indices"][:]
            teacher_hidden_states = torch.from_numpy(
                np.array(teacher_dataset_file["hidden_states"][:], dtype=np.float32)
            )
            teacher_dataset_file.close()

            logging.info(f"✓ Loaded {len(teacher_hidden_states)} teacher hidden states")
            logging.info(f"✓ Teacher hidden states shape: {teacher_hidden_states.shape}")
            logging.info(f"✓ Teacher states stored in CPU RAM for efficient batch lookup")

            # Create student projection MLP: [batch, 149, 960] → [batch, 4096]
            aggregation_method = AggregationMethod.MEAN if distill_cfg.aggregation_method == "mean" else AggregationMethod.LAST
            student_projection_mlp = StudentSequenceProjectionMLP(
                input_dim=distill_cfg.student_hidden_dim,  # 960
                output_dim=distill_cfg.teacher_hidden_dim,  # 4096
                bottleneck_dim=distill_cfg.bottleneck_dim,  # 512
                aggregation_method=aggregation_method,
            ).to(device)

            # Create distillation loss
            if distill_cfg.distill_loss_type == "similarity":
                distillation_loss_fn = SimilarityMatrixDistillationLoss(
                    student_hidden_dim=distill_cfg.teacher_hidden_dim,  # After projection: 4096
                    teacher_hidden_dim=distill_cfg.teacher_hidden_dim,  # 4096
                    temperature_student=distill_cfg.distill_temperature,
                    temperature_teacher=distill_cfg.distill_temperature,
                    normalize=distill_cfg.normalize,
                    mask_diagonal=distill_cfg.mask_diagonal,
                    apply_softmax=True,
                ).to(device)
            elif distill_cfg.distill_loss_type == "infonce":
                distillation_loss_fn = InfoNCEDistillationLoss(
                    student_hidden_dim=distill_cfg.teacher_hidden_dim,  # After projection: 4096
                    teacher_hidden_dim=distill_cfg.teacher_hidden_dim,  # 4096
                    temperature=distill_cfg.infonce_temperature,
                    normalize=distill_cfg.normalize,
                ).to(device)
            else:
                raise ValueError(f"Unknown distill_loss_type: {distill_cfg.distill_loss_type}")

            distill_weight = distill_cfg.distill_weight

            logging.info(f"Student projection MLP: {distill_cfg.student_hidden_dim} → {distill_cfg.bottleneck_dim} → {distill_cfg.teacher_hidden_dim}")
            logging.info(f"Distillation loss: {distill_cfg.distill_loss_type}, weight: {distill_weight}")

    logging.info("Creating optimizer and scheduler")
    # Include student projection MLP parameters in optimizer if distillation is enabled
    if student_projection_mlp is not None:
        all_parameters = list(policy.parameters()) + list(student_projection_mlp.parameters())
    else:
        all_parameters = list(policy.parameters())

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # Update optimizer to include MLP parameters
    if student_projection_mlp is not None:
        optimizer = torch.optim.AdamW(
            all_parameters,
            lr=cfg.optimizer.lr,
            betas=cfg.optimizer.betas,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        )

    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

        # Load student projection MLP if distillation is enabled
        if student_projection_mlp is not None:
            mlp_path = cfg.checkpoint_path / "student_projection_mlp.pt"
            if mlp_path.exists():
                student_projection_mlp.load_state_dict(torch.load(mlp_path, map_location=device))
                logging.info(f"Loaded student projection MLP from {mlp_path}")
            else:
                logging.warning(f"Student projection MLP checkpoint not found at {mlp_path}")

    # Count parameters including student projection MLP if distillation is enabled
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if student_projection_mlp is not None:
        mlp_learnable_params = sum(p.numel() for p in student_projection_mlp.parameters() if p.requires_grad)
        mlp_total_params = sum(p.numel() for p in student_projection_mlp.parameters())
        num_learnable_params += mlp_learnable_params
        num_total_params += mlp_total_params

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if student_projection_mlp is not None:
        logging.info(f"  ├─ Policy: {sum(p.numel() for p in policy.parameters() if p.requires_grad)} learnable params")
        logging.info(f"  └─ Student Projection MLP: {mlp_learnable_params} learnable params ({format_big_number(mlp_learnable_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grad_norm", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "action_loss": AverageMeter("action_loss", ":.4f"),
        "task_loss": AverageMeter("task_loss", ":.3f"),
        "distill_loss": AverageMeter("distill_loss", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            teacher_hidden_states=teacher_hidden_states,
            student_projection_mlp=student_projection_mlp,
            distillation_loss_fn=distillation_loss_fn,
            distill_weight=distill_weight,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)

            # Print detailed metrics every log step
            metrics_dict = train_tracker.to_dict()
            print(f"\n{'='*80}")
            print(f"STEP {step} METRICS:")
            print(f"{'='*80}")
            print(f"  loss:         {metrics_dict.get('loss', 'N/A'):.4f}")
            print(f"  action_loss:  {metrics_dict.get('action_loss', 'N/A'):.4f}")
            print(f"  task_loss:    {metrics_dict.get('task_loss', 'N/A'):.4f}")
            print(f"  distill_loss: {metrics_dict.get('distill_loss', 'N/A'):.4f}")
            print(f"  grad_norm:    {metrics_dict.get('grad_norm', 'N/A'):.4f}")
            print(f"  lr:           {metrics_dict.get('lr', 'N/A'):.2e}")
            print(f"{'='*80}\n")

            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")

            # Delete previous checkpoint to save only the latest
            checkpoints_dir = cfg.output_dir / "checkpoints"
            if checkpoints_dir.exists():
                import shutil
                for old_checkpoint in checkpoints_dir.iterdir():
                    if old_checkpoint.is_dir() and old_checkpoint.name != "last":
                        shutil.rmtree(old_checkpoint)
                        logging.info(f"Deleted old checkpoint: {old_checkpoint.name}")

            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)

            # Save student projection MLP if distillation is enabled
            if student_projection_mlp is not None:
                mlp_path = checkpoint_dir / "student_projection_mlp.pt"
                torch.save(student_projection_mlp.state_dict(), mlp_path)
                logging.info(f"Saved student projection MLP to {mlp_path}")

            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    init_logging()
    train()
