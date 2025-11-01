# LIBERO Spatial Dataset for Franka Panda Robot

## Dataset Description

This dataset contains demonstrations from the LIBERO (Lifelong Benchmarking for Robot Learning) Spatial task suite, converted to RLDS format for OpenVLA finetuning. LIBERO Spatial focuses on spatial reasoning tasks where a **Franka Emika Panda 7-DOF robot** must manipulate objects in specific spatial configurations.

The dataset includes dual-camera observations (context and wrist-mounted views) with 7-DOF robot actions and natural language task instructions.

**Original Dataset:** [LIBERO Benchmark](https://libero-project.github.io/)  
**Conversion Source:** [openvla_libero_spatial on HuggingFace](https://huggingface.co/datasets/mimimimi2002/openvla_libero_spatial)  
**Robot Platform:** Franka Emika Panda (7-DOF arm + 2-finger parallel gripper)

## Task Examples

The LIBERO Spatial suite includes tasks such as:
- "pick up the red block and place it in the drawer"
- "put the cube on the plate"
- "stack the blocks in order"
- Various spatial manipulation and arrangement tasks

## Dataset Statistics

- **Training episodes:** [Your train count]
- **Validation episodes:** [Your val count]
- **Total timesteps:** ~[calculated from episodes]
- **Image resolution:** 256x256 RGB
- **Action space:** 7-DOF (6D end-effector pose delta + 1D gripper command)
- **State space:** 8-DOF (6D EEF pose + 2D gripper state)
- **Joint space:** 7-DOF joint angles
- **Cameras:** Dual camera setup (fixed context view + wrist-mounted view)
- **Robot:** Franka Emika Panda 7-DOF manipulator

## Data Format

Each episode contains:
- **Context Images:** Fixed camera RGB observations (256x256x3, uint8)
- **Wrist Images:** Wrist-mounted camera RGB observations (256x256x3, uint8)
- **Actions:** 7-DOF robot commands (6D end-effector pose delta + 1D gripper)
- **States:** 8-DOF robot state (6D EEF pose + 2D gripper positions)
- **Joint States:** 7-DOF joint angle measurements
- **Language Instructions:** Natural language task descriptions
- **Language Embeddings:** 512-D Universal Sentence Encoder embeddings

### Action Space (7-DOF)
- Dimensions 0-2: End-effector position delta (x, y, z)
- Dimensions 3-5: End-effector orientation delta (rotation)
- Dimension 6: Gripper command (continuous open/close)

### State Space (8-DOF)
- Dimensions 0-2: End-effector position (x, y, z)
- Dimensions 3-5: End-effector orientation (euler angles/rotation)
- Dimensions 6-7: Gripper state (left and right finger positions)

### Joint State (7-DOF)
- 7 joint angles for the Franka Panda arm

## Data Processing

The following processing has been applied during conversion:

1. **Video Frame Extraction:** Original MP4 videos were decoded and extracted frame-by-frame using imageio/ffmpeg
2. **Image Resizing:** All images resized from original resolution to 256x256 using Lanczos interpolation
3. **Frame Synchronization:** Episodes where video frame count didn't match timestep count were truncated to minimum length
4. **Language Embeddings:** Generated using Universal Sentence Encoder (512-D) for each instruction
5. **Episode Validation:** Episodes with missing video files or extraction errors were skipped
6. **Train/Val Split:** Dataset split with stratification to ensure all tasks are represented in both splits (validation ratio: ~15%, minimum 2 episodes per task)

### Skipped/Corrupted Examples
- Episodes with failed video decoding were excluded
- Episodes with frame count mismatches were truncated to valid length
- No data augmentation applied - all demonstrations are original teleoperated trajectories

## Example Trajectories

[Add 2-4 example trajectory images here]

### Context Camera View
![Example trajectory - context view](examples/context_example.png)

### Wrist Camera View  
![Example trajectory - wrist view](examples/wrist_example.png)

## Citation

If you use this dataset, please cite both LIBERO and OpenVLA:
```bibtex