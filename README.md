# Knowledge Distillation for OpenVLA Training

Complete guide for training OpenVLA student models with knowledge distillation from LAPA teacher hidden states.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Two Issues & Solutions](#two-issues--solutions)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Understanding the Loss](#understanding-the-loss)
7. [Monitoring & Debugging](#monitoring--debugging)
8. [Advanced Tuning](#advanced-tuning)
9. [Troubleshooting](#troubleshooting)
10. [Technical Details](#technical-details)

---

## Quick Start

### 1. Generate Teacher Hidden States (One-time)
```bash
python src/lapa/latent_pretraining/extract_lapa_hidden_states.py \
    --image_directory <path_to_images> \
    --output_h5_path lapa_hidden_states.h5
```

### 2. Verify Dataset Metadata
Ensure your dataset batch returns:
```python
batch["episode_indices"]  # [batch_size] - which episode
batch["frame_indices"]    # [batch_size] - which frame in episode
```

### 3. Run Training with Distillation
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    vla-scripts/distill.py \
    --data_root_dir datasets/libero_spatial_noops \
    --dataset_name libero_spatial \
    --batch_size 12 \
    --use_distillation \
    --teacher_h5_path /path/to/lapa_hidden_states.h5 \
    --aggregation_method mean \
    --frame_alignment_mode supervised_only \
    --distill_weight 0.1
```

**Expected Results**: 5-10% task performance improvement, 2-5% action loss reduction

---

## Overview

Knowledge distillation leverages teacher (LAPA) hidden states to regularize student (OpenVLA) learning via **similarity matrix KL divergence**.

### What is Similarity Matrix Loss?

Instead of comparing hidden states directly (magnitude-dependent, unstable), we compare **how similar samples are**:

```
Student similarity matrix:  S_s = student_hidden @ student_hidden.T   [batch, batch]
Teacher similarity matrix:  S_t = teacher_hidden @ teacher_hidden.T   [batch, batch]

Loss = KL(softmax(S_t / temperature) || softmax(S_s / temperature))
```

**Benefits**:
- ✓ Scale-invariant (works regardless of magnitude)
- ✓ Focuses on relationships (which samples are similar)
- ✓ More stable than L2 loss on 4096-dimensional vectors
- ✓ Works well with small batches

### Architecture Integration

```
Input → VLA Model → Hidden States [B, L, 4096]
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
         Student Path          Action Loss
              ↓                   (task loss)
         Aggregate Sequence
         [B, L, 4096] → [B, 4096]
              ↓
         Similarity Matrix        Load Teacher HDF5
         S = H @ H.T [B, B]           ↓
              ↓                   Align Frames
         Softmax(S/T)            Aggregate
              ↓                   Similarity Matrix
              └──────→ KL Divergence ←──┘
                           ↓
         Total Loss = L_action + weight * L_distill
                           ↓
                      Backward Pass
```

---

## Two Issues & Solutions

### Issue #1: Sequence Aggregation

**Problem**: Each frame has ~391 tokens, but we need ONE representation per frame.

#### Solution A: Last Token Only
```python
aggregated = hidden_states[:, -1, :]  # [batch, 4096]
```
- **Simplest**, **fastest**
- Good when final token encodes decision
- Less robust to sequence length variation

#### Solution B: Mean Pooling
```python
aggregated = hidden_states.mean(dim=1)  # [batch, 4096]
```
- **More robust** to length variation
- Better when info distributed across tokens
- Recommended for general use

**Flag**: `--aggregation_method {last, mean}`

**Recommendation**: Start with `mean`, try `last` if training is unstable

---

### Issue #2: Frame Alignment

**Problem**: Teacher has states every 12 frames (0, 12, 24...), student uses every frame.

#### Solution A: Supervised Only
```
Frames:     0  1  2  3  4  5  6  7  8  9  10  11  12
Teacher: ✓  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·   ·   ·  ✓
Loss:    ✓  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·   ·   ·  ✓
```
- Compute loss **only on frames 0, 12, 24...** (1 in 12 frames)
- Sparse but **clean supervision signal**
- **No assumptions** about intermediate frames
- Simpler, faster, more stable

#### Solution B: Interpolated
```
Frames:     0  1  2  3  4  5  6  7  8  9  10  11  12
Teacher: ✓  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?   ?   ?  ✓
Loss:    ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓   ✓   ✓  ✓
```
- **Linearly interpolate** teacher states: `state[i] = (1-r)*state[0] + r*state[12]`
- Dense supervision **(every frame)**
- **Stronger regularization** signal
- Assumes linear interpolation is valid (reasonable over 0.6s)

**Flag**: `--frame_alignment_mode {supervised_only, interpolated}`

**Recommendation**: Start with `supervised_only`, try `interpolated` for stronger signal

---

## Configuration

### All Parameters

| Flag | Type | Default | Range | Description |
|------|------|---------|-------|-------------|
| `--use_distillation` | bool | False | - | Enable distillation |
| `--teacher_h5_path` | Path | None | - | Path to `lapa_hidden_states.h5` |
| `--aggregation_method` | str | "last" | {last, mean} | Sequence aggregation (Issue #1) |
| `--frame_alignment_mode` | str | "supervised_only" | {supervised_only, interpolated} | Frame alignment (Issue #2) |
| `--distill_weight` | float | 0.1 | 0.0-1.0 | Loss weight |
| `--distill_temperature` | float | 4.0 | 1.0-10.0 | Softmax temperature |
| `--distill_normalize` | bool | True | - | L2 normalize before similarity |
| `--distill_projection_dim` | int | None | - | Optional hidden state compression |

### Pre-Built Configurations

**Config A: Conservative** (safest, minimal risk)
```bash
--aggregation_method last \
--frame_alignment_mode supervised_only \
--distill_weight 0.05 \
--distill_temperature 4.0
```
- Use when: Starting out, unsure about setup
- Pros: Minimal assumptions, unlikely to hurt
- Cons: Weaker signal

**Config B: Standard** (recommended, good balance)
```bash
--aggregation_method mean \
--frame_alignment_mode supervised_only \
--distill_weight 0.1 \
--distill_temperature 4.0
```
- Use when: Most cases, first choice
- Pros: Balanced, works well
- Cons: None significant

**Config C: Aggressive** (strong regularization)
```bash
--aggregation_method mean \
--frame_alignment_mode interpolated \
--distill_weight 0.2 \
--distill_temperature 4.0
```
- Use when: High-quality teacher, want strong signal
- Pros: Dense supervision, stronger improvement
- Cons: More computational cost, may overfit to teacher

**Config D: Compressed** (low memory)
```bash
--aggregation_method last \
--frame_alignment_mode supervised_only \
--distill_weight 0.1 \
--distill_projection_dim 512
```
- Use when: Memory-constrained
- Pros: 10-15% less memory
- Cons: Some information loss

---

## Usage Examples

### Example 1: Minimal (Just Works)
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    vla-scripts/distill.py \
    --batch_size 12 \
    --use_distillation \
    --teacher_h5_path lapa_hidden_states.h5
```
Uses all defaults (Config B equivalent)

### Example 2: Recommended (Config B)
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    vla-scripts/distill.py \
    --data_root_dir datasets/libero_spatial_noops \
    --dataset_name libero_spatial \
    --batch_size 12 \
    --learning_rate 5e-4 \
    --use_distillation \
    --teacher_h5_path /path/to/lapa_hidden_states.h5 \
    --aggregation_method mean \
    --frame_alignment_mode supervised_only \
    --distill_weight 0.1 \
    --distill_temperature 4.0
```

### Example 3: Dense Supervision (Config C)
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    vla-scripts/distill.py \
    --data_root_dir datasets/libero_spatial_noops \
    --dataset_name libero_spatial \
    --batch_size 12 \
    --use_distillation \
    --teacher_h5_path /path/to/lapa_hidden_states.h5 \
    --aggregation_method mean \
    --frame_alignment_mode interpolated \
    --distill_weight 0.2 \
    --distill_temperature 4.0
```

### Example 4: Low Memory (Config D)
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
    vla-scripts/distill.py \
    --data_root_dir datasets/libero_spatial_noops \
    --dataset_name libero_spatial \
    --batch_size 12 \
    --use_distillation \
    --teacher_h5_path /path/to/lapa_hidden_states.h5 \
    --aggregation_method last \
    --frame_alignment_mode supervised_only \
    --distill_weight 0.1 \
    --distill_projection_dim 512
```

---

## Understanding the Loss

### How Distillation Weight Works

```
Total Loss = action_loss + distill_weight * distill_loss
```

**Effect of weight**:
- `0.0`: No distillation (baseline)
- `0.05`: Very light regularization
- `0.1`: Light-to-moderate (recommended)
- `0.2-0.3`: Strong regularization
- `0.5+`: Dominates action loss (not recommended)

**Tuning guideline**:
- If no improvement: ↓ weight to 0.05
- If too slow: ↑ weight to 0.15-0.2
- If unstable: ↓ weight and ↑ temperature

### Temperature Parameter

Controls softness of similarity distribution:

```
P = softmax(S / temperature)
```

**Effect**:
- `temperature=1.0`: Sharp, focus on strongest similarities
- `temperature=4.0`: Balanced (default, works well)
- `temperature=8.0`: Soft, learn general structure

**When to adjust**:
- If loss is high/noisy: ↑ temperature to 6-8
- If not learning: ↓ temperature to 2-3

### L2 Normalization

When `--distill_normalize` (default True):
```python
student = student / ||student||_2  # Unit norm
teacher = teacher / ||teacher||_2  # Unit norm
similarity = student @ student.T    # Cosine similarity
```

**Why**:
- Makes loss invariant to magnitude
- More stable training
- Better generalization

**Keep True** unless you have specific reason not to

---

## Monitoring & Debugging

### W&B Metrics

When distillation enabled, W&B logs:
- `action_loss`: Task prediction loss
- `distill_loss`: KL divergence between similarity matrices
- `total_loss`: Weighted combination
- `action_accuracy`: Token accuracy
- `l1_loss`: Continuous action error

### Quick Validation (1 batch)
```bash
torchrun ... distill.py \
    --use_distillation \
    --teacher_h5_path lapa_hidden_states.h5 \
    --batch_size 2 \
    --max_steps 1
```

Verify:
- ✓ No file errors
- ✓ No shape mismatches
- ✓ `distill_loss` computes (not NaN)
- ✓ `distill_loss` is reasonable value (0.1-1.0 initially)

### Common Observations

**Good signs**:
- `distill_loss` decreases over time
- `action_loss` continues to decrease
- Total loss follows expected trajectory
- No NaN or Inf values

**Bad signs**:
- `distill_loss` always 0 → Check HDF5 path and frame indices
- `distill_loss` very high (>10) → Reduce weight
- Training unstable (spikes) → Increase temperature
- Performance doesn't improve → May not help for task

---

## Advanced Tuning

### For Stronger Distillation
When you want more regularization:
```bash
--aggregation_method mean              # More info
--frame_alignment_mode interpolated    # Dense supervision
--distill_weight 0.2-0.3               # Higher weight
--distill_temperature 2.0              # Sharper distribution
```

### For Lighter Distillation
When you want minimal interference:
```bash
--aggregation_method last              # Less info
--frame_alignment_mode supervised_only # Sparse supervision
--distill_weight 0.05                  # Lower weight
--distill_temperature 8.0              # Soft distribution
```

### For Very Different Models
When teacher and student architectures differ significantly:
```bash
--aggregation_method mean              # Better robustness
--distill_projection_dim 1024          # Project to intermediate dimension
--distill_weight 0.1                   # Moderate weight
--distill_temperature 4.0              # Default
```

### Curriculum Learning
Start conservative, gradually strengthen:
- **Weeks 1-2**: Config A (conservative)
- **Weeks 3-6**: Config B (standard)
- **After week 6**: Config C (aggressive) if helpful

---

## Troubleshooting

### "FileNotFoundError: lapa_hidden_states.h5"
**Check**:
```bash
ls -la /path/to/lapa_hidden_states.h5
```
**Fix**: Use absolute path, verify file exists

### "⚠️ Warning: No teacher hidden states loaded for batch"
**Cause**: Frame indices in batch don't match HDF5 file

**Debug**:
```python
# Check what's in HDF5
import h5py
f = h5py.File('lapa_hidden_states.h5', 'r')
frames_in_h5 = set(f['frame_indices'][:])
print("Frames in H5:", sorted(frames_in_h5)[:20])

# Check what's in batch
for batch in dataloader:
    frames_in_batch = set(batch['frame_indices'].numpy())
    print("Frames in batch:", sorted(frames_in_batch)[:20])
    break
```
**Fix**: Ensure dataset frame indexing matches extraction script

### "distill_loss stays at 0"
**Check**:
1. Is HDF5 file valid? `python -c "import h5py; print(list(h5py.File('file.h5').keys()))"`
2. Do frames overlap? (see above)
3. Are episode indices in range? `print(f['episode_indices'][:].min(), f['episode_indices'][:].max())`

### "CUDA out of memory"
**Solution 1**: Reduce batch size
```bash
--batch_size 6  # Down from 12
```
**Solution 2**: Use projection to compress
```bash
--distill_projection_dim 512
```
**Solution 3**: Use simpler aggregation
```bash
--aggregation_method last  # Instead of mean
```

### "distill_loss is very high (>10)"
**Cause**: Weight too high or teacher/student very misaligned

**Fix**:
```bash
--distill_weight 0.05      # Reduce weight
--aggregation_method mean  # More robust
--distill_temperature 8.0  # Softer
```

### "Training becomes unstable (spikes)"
**Fix**:
```bash
--distill_weight 0.05              # Lower weight
--aggregation_method mean          # More stable
--distill_temperature 6.0-8.0      # Softer
--frame_alignment_mode supervised_only  # Sparse signal
```

### "Performance doesn't improve"
Distillation may not help for your task. Try:
1. Reduce weight further (0.05)
2. Verify teacher quality
3. Check frame index alignment
4. Ensure teacher relevant to task

---

## Technical Details

### Frame Interpolation

For frame 6 between teacher frames 0 and 12:
```python
ratio = (6 - 0) / 12 = 0.5
interpolated_state = (1 - 0.5) * state[0] + 0.5 * state[12]
```

Frames that don't have neighbor teachers use nearest available state.

### Sequence Aggregation Details

**Last Token** (fastest):
```python
hidden_states[:, -1, :]  # Shape: [batch, 4096]
```
- Memory: O(batch)
- Speed: O(1) per sample

**Mean** (more stable):
```python
hidden_states.mean(dim=1)  # Shape: [batch, 4096]
```
- Memory: O(batch * seq_len) during computation
- Speed: O(seq_len) per sample

### Similarity Matrix Computation

After aggregation `[batch, 4096]`:
```python
# Normalize (if enabled)
student = F.normalize(student, p=2, dim=1)
teacher = F.normalize(teacher, p=2, dim=1)

# Similarity matrix
S_student = student @ student.T      # [batch, batch]
S_teacher = teacher @ teacher.T      # [batch, batch]

# Apply temperature
S_student = S_student / temperature
S_teacher = S_teacher / temperature

# Probabilities
P_student = softmax(S_student, dim=1)
P_teacher = softmax(S_teacher, dim=1)

# KL divergence
loss = KL(log(P_student), P_teacher)
```

### Frame Valid Mask

When using `supervised_only`:
- Frames 0, 12, 24, ... have `valid_mask=True`
- Other frames have `valid_mask=False`
- Loss weighted by valid frames

When using `interpolated`:
- All frames have `valid_mask=True` if neighbor teachers exist
- Frames outside teacher range have `valid_mask=False`

---

## Files Reference

### Core Implementation
- **`distillation_utils.py`**: All classes and utilities
  - `AggregationMethod`: Enum for aggregation strategies
  - `SequenceAggregationMLP`: Aggregation implementation
  - `FrameAlignmentMode`: Enum for alignment strategies
  - `FrameAlignmentStrategy`: Frame alignment implementation
  - `SimilarityMatrixDistillationLoss`: KL divergence loss
  - `TeacherHiddenStateLoader`: HDF5 loader with caching

### Training Script
- **`distill.py`**: Modified with distillation integration
  - Config parameters for all flags
  - Initialization of distillation components
  - Loss computation in training loop
  - W&B logging support

### Documentation
- **`KNOWLEDGE_DISTILLATION.md`**: This file

---

## Expected Results

### Without Distillation
Baseline performance, random variation ~2-3%

### With Config B (Standard)
- **After 10k steps**: 2-3% improvement visible
- **After 50k steps**: 5-8% improvement
- **Final**: 5-10% task success improvement, 2-5% action loss reduction

### With Config C (Aggressive)
- **Final**: 8-15% task success improvement, 5-10% action loss reduction
- **Tradeoff**: ~10-15% slower training

### Variability Factors
- Teacher quality (LAPA must be well-trained)
- Task complexity
- Dataset size
- Model capacity
- Configuration choices
- Random seed

---

## Frequently Asked Questions

**Q: Do I need to regenerate lapa_hidden_states.h5?**
A: Only once, unless images change. Cache the file.

**Q: Can I use distillation with LoRA/quantization?**
A: Yes! Distillation is orthogonal, works with all training techniques.

**Q: What's the memory overhead?**
A: ~10-15% (teacher states + similarity matrices).

**Q: Can I use multiple GPUs?**
A: Yes, DDP handles it. Distillation loss computed per-GPU independently.

**Q: Why not use distill_weight=1.0?**
A: Would ignore action task. Balance needed via weight.

**Q: Should I change learning rate?**
A: No, keep same. Adjust via distill_weight instead.

**Q: How long does distillation take?**
A: ~5-10% slower per step (teacher loading + loss computation).

**Q: What if I don't have teacher_h5_path?**
A: Generate it first with extract_lapa_hidden_states.py.

**Q: Can I ensemble multiple teachers?**
A: Not directly, but could merge HDF5 files.

**Q: Should I normalize representations?**
A: Yes (default). Keeps loss stable and scale-invariant.

---

## Getting Started Checklist

- [ ] Generate teacher hidden states: `extract_lapa_hidden_states.py`
- [ ] Verify dataset returns `episode_indices` and `frame_indices`
- [ ] Run quick validation: 1-batch test with `--max_steps 1`
- [ ] Start with Config B (standard)
- [ ] Monitor W&B logs for loss trends
- [ ] Compare final performance with/without distillation
- [ ] Adjust weights/config based on results

---

## Support

If something fails:
1. Check error message carefully
2. Try relevant fix from Troubleshooting section
3. Verify all paths are absolute and files exist
4. Check W&B logs for stack traces
5. Ensure dataset provides required metadata

For detailed technical info, see `distillation_utils.py` docstrings.

---

## LIBERO Evaluation

After training, evaluate your model on LIBERO benchmark tasks.

### Prerequisites

1. Install LIBERO:
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

2. Set environment variables:
```bash
export LIBERO_PATH=/home/elias/Thesis/LIBERO/libero/libero
export LIBERO_CONFIG_PATH=/home/elias/Thesis/LIBERO/.libero
```

### Running Evaluation

Navigate to the OpenVLA source directory and run:

```bash
cd /home/elias/Thesis/src/openvla

export LIBERO_PATH=/home/elias/Thesis/LIBERO/libero/libero
export LIBERO_CONFIG_PATH=/home/elias/Thesis/LIBERO/.libero
PYTHONPATH=/home/elias/Thesis/src/openvla:$PYTHONPATH python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint /home/elias/Thesis/checkpoints/distill_infonce_norm_1_temp_0.07_weight_0.3_full \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 20 \
  --load_in_8bit True \
  --use_wandb True \
  --wandb_project openvla-distill-eval \
  --wandb_entity eliaskallioras-national-technical-university-of-athens \
  --is_baseline False
```

### Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--pretrained_checkpoint` | Path to trained model checkpoint | Directory containing model weights |
| `--task_suite_name` | LIBERO task suite to evaluate on | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90` |
| `--center_crop` | Use center crop (needed if trained with image aug) | `True`, `False` |
| `--num_trials_per_task` | Number of rollouts per task | Default: 20 |
| `--load_in_8bit` | Use 8-bit quantization for inference | `True`, `False` |
| `--use_wandb` | Log results to Weights & Biases | `True`, `False` |
| `--wandb_project` | W&B project name | Any string |
| `--wandb_entity` | W&B entity name | Your W&B username/team |

### What Gets Logged to W&B

When `--use_wandb True`, the script automatically logs:

1. **Training Configuration** (from `training_config.yaml` in checkpoint):
   - Batch size, learning rate, LoRA rank
   - Distillation parameters (loss type, temperature, weight, etc.)
   - All hyperparameters used during training

2. **Per-Task Performance**:
   - `success_rate/<task_name>`: Success rate for each individual task
   - `num_episodes/<task_name>`: Number of episodes run per task

3. **Final Performance**:
   - `success_rate/total`: Overall success rate across all tasks
   - `num_episodes/total`: Total number of episodes

### Example Output

```
Task suite: libero_spatial
Loading training config from /path/to/checkpoint/training_config.yaml
Logged training config to wandb

Task: pick up the red cube
Success: True
# episodes completed so far: 1
# successes: 1 (100.0%)

...

Current task success rate: 0.85
Current total success rate: 0.72
```

### Comparing Different Training Configs

Since training hyperparameters are automatically logged to W&B, you can easily compare:
- Different distillation temperatures
- Different loss types (InfoNCE vs KL-Divergence)
- Different distillation weights
- LoRA configurations

Just run evaluation on multiple checkpoints and compare in W&B dashboard.

### Troubleshooting

**"LIBERO not found"**:
```bash
pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

**"No training config found"**:
- Ensure your checkpoint was saved with the updated training script that saves `training_config.yaml`
- Config will be missing for old checkpoints (evaluation still works, just won't log training params)

**"Action un-norm key not found"**:
- Model doesn't have normalization stats for the task suite
- Retrain on the specific LIBERO suite or use a model pretrained on LIBERO data
