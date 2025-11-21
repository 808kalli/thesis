"""
extract_lapa_hidden_states.py

Extracts aggregated LAPA hidden states from Libero spatial dataset frames.

This script:
1. Loads ALL frames from each episode (no frame skipping)
2. Extracts hidden states from LAPA for each frame
3. Aggregates sequence [seq_len, 4096] → [4096] using specified method
4. Stores aggregated states with episode_id, frame_id, and global_id

The dataset is from HuggingFace (lerobot format, parquet files):
https://huggingface.co/datasets/aopolin-lv/libero_spatial_no_noops_lerobot_v21

Usage:
python latent_pretraining/extract_lapa_hidden_states.py \
    --dataset_dir /workspace/thesis/raw_datasets/libero_spatial \
    --output_dir /workspace/thesis \
    --vqgan_checkpoint lapa_checkpoints/vqgan \
    --load_checkpoint params::lapa_checkpoints/params_sthv2 \
    --aggregation_method mean \
    --num_episodes None \
    --seed 7

Arguments:
    --dataset_dir: Local path to libero_spatial dataset directory (lerobot format with videos and parquet files)
    --output_dir: Directory to save hidden states HDF5 file
    --vqgan_checkpoint: Path to VQGAN checkpoint
    --load_checkpoint: Path to LAPA model checkpoint
    --load_llama_config: LAPA config (default: 7b)
    --aggregation_method: How to aggregate sequence: "last" or "mean" (default: mean)
    --num_episodes: Maximum number of episodes to process (None = all)
    --seed: Random seed

Outputs:
    {output_dir}/lapa_hidden_states.h5 containing:
    - hidden_states: [total_samples, 4096] aggregated hidden states
    - episode_indices: [total_samples] episode IDs
    - frame_indices: [total_samples] frame indices (sampled at stride intervals)
    - global_indices: [total_samples] global sequential indices into full dataset
    - task_descriptions: [total_samples] task description strings
"""

import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import subprocess
import tempfile
import shutil
from io import BytesIO

import draccus
import numpy as np
import tqdm
from PIL import Image

# Prevent JAX from allocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Append current directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_pretraining.sampler_latent_pretrain import DeltaSampler
from latent_pretraining.delta_llama import VideoLLaMAConfig
from tux import JaxDistributedConfig, set_random_seed

# Task descriptions mapped from task_index (libero_spatial suite)
LIBERO_SPATIAL_TASKS = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
]


class FLAGSClass:
    """Simple flags container matching inference.py pattern"""
    def __init__(self, flag_dict):
        for key, value in flag_dict.items():
            setattr(self, key, value)

# ============================================================
# CONFIG
# ============================================================

@dataclass
class ExtractLAPAConfig:
    """Configuration for extracting aggregated LAPA hidden states."""
    # fmt: off

    # Dataset parameters
    dataset_dir: Union[str, Path] = "/path/to/libero_spatial"  # Lerobot format (has both videos and parquet with global indices)

    # LAPA model parameters (same as inference.py)
    vqgan_checkpoint: str = "lapa_checkpoints/vqgan"
    load_checkpoint: str = "params::lapa_checkpoints/params"
    load_llama_config: str = "7b"
    update_llama_config: str = "dict(delta_vocab_size=8,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)"

    # JAX parameters (same as inference.py)
    mesh_dim: str = "1,-1,1,1"
    dtype: str = "bf16"
    vocab_file: str = "lapa_checkpoints/tokenizer.model"
    seed: int = 1234

    # Inference parameters
    tokens_per_delta: int = 4  # Matches inference.py
    multi_image: int = 1
    num_episodes: Optional[int] = None  # If set, only process first N episodes
    aggregation_method: str = "mean"  # How to aggregate: "last" (last token) or "mean" (mean of all)
    frame_stride: int = 2  # Process every Nth frame (stride=1 for all, stride=2 for every 2nd, etc.)

    # Output parameters
    output_dir: Union[str, Path] = "lapa_hidden_states"

    # fmt: on


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def aggregate_sequence(seq: np.ndarray, method: str = "mean") -> np.ndarray:
    """Aggregate [seq_len, 4096] → [4096]"""
    if method == "last":
        return seq[-1, :].astype(np.float32)  # Last token
    elif method == "mean":
        return seq.mean(axis=0).astype(np.float32)  # Mean of all tokens
    else:
        raise ValueError(f"Unknown aggregation: {method}")


def extract_all_frames_from_video(video_path: Path, max_frame_idx: int) -> dict:
    """
    Extract ALL frames from video in a single ffmpeg call for efficiency.

    Instead of calling ffmpeg for each frame (O(n²) decoding cost),
    we dump all frames to disk in one pass (O(n) cost).

    Args:
        video_path: Path to video file
        max_frame_idx: Expected number of frames (0-indexed, so total = max_frame_idx + 1)

    Returns:
        dict mapping frame_idx -> PIL Image
    """
    # Create temporary directory for frame extraction
    temp_dir = tempfile.mkdtemp(prefix="lapa_frames_")

    try:
        # Extract all frames with ffmpeg in a single pass
        # fps=1 means 1 frame per second (adjust if needed for your video)
        frame_pattern = temp_dir + "/frame_%04d.png"
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-f', 'image2',
            '-pix_fmt', 'rgb24',
            frame_pattern
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=30)

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to extract frames from {video_path}. "
                f"stderr: {result.stderr.decode()[:500]}"
            )

        # Load frames from disk into memory
        frames = {}
        frame_files = sorted(Path(temp_dir).glob("frame_*.png"))

        for i, frame_file in enumerate(frame_files):
            try:
                frames[i] = Image.open(frame_file).convert('RGB')
            except Exception as e:
                print(f"Warning: Could not load frame {i}: {e}")

        return frames

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# LAPA HIDDEN STATE EXTRACTOR
# ============================================================

class LAPAHiddenStateExtractor:
    """Wrapper to extract hidden states from LAPA during inference.

    Follows the same initialization pattern as inference.py
    """

    def __init__(self, cfg: ExtractLAPAConfig):
        """Initialize LAPA model for hidden state extraction."""
        self.cfg = cfg
        print("Initializing LAPA hidden state extractor...")

        # Setup JAX (same as inference.py)
        JaxDistributedConfig.initialize(JaxDistributedConfig.get_default_config())
        set_random_seed(cfg.seed)

        # Setup tokenizer and llama config (same as inference.py)
        tokenizer_config = VideoLLaMAConfig.get_tokenizer_config()
        tokenizer_config.vocab_file = cfg.vocab_file
        llama_config = VideoLLaMAConfig.get_default_config()

        # Create flags object (same pattern as inference.py)
        flags_dict = {
            'tokens_per_delta': cfg.tokens_per_delta,
            'vqgan_checkpoint': cfg.vqgan_checkpoint,
            'vocab_file': cfg.vocab_file,
            'multi_image': cfg.multi_image,
            'jax_distributed': JaxDistributedConfig.get_default_config(),
            'seed': cfg.seed,
            'mesh_dim': cfg.mesh_dim,
            'dtype': cfg.dtype,
            'load_llama_config': cfg.load_llama_config,
            'update_llama_config': cfg.update_llama_config,
            'load_checkpoint': cfg.load_checkpoint,
            'tokenizer': tokenizer_config,
            'llama': llama_config,
        }

        flags = FLAGSClass(flags_dict)

        # Initialize DeltaSampler (from sampler_latent_pretrain.py, same as inference.py)
        self.model = DeltaSampler(FLAGS=flags)
        print("✓ LAPA model initialized")

    def extract_hidden_state_from_image(self, image: Image.Image, task_description: str = None) -> np.ndarray:
        """
        Extract LAPA hidden states from a single image.

        This method replicates the inference flow from DeltaSampler.__call__()
        (sampler_latent_pretrain.py:218-222) and generate_video_pred()
        (sampler_latent_pretrain.py:180-216), but calls the model directly
        to capture hidden states instead of using .generate() which discards them.

        Args:
            image: PIL Image to process
            task_description: Optional task description string from dataset. If provided,
                            will be used in the text prompt. If None, uses generic prompt.

        Returns:
            hidden_states: np.ndarray of shape [seq_len, 4096] where seq_len ≈ 395
                          (number of VQGAN-encoded vision tokens)
        """
        import jax

        # STEP 1: VQGAN encode the image → vision token IDs
        # Source: sampler_latent_pretrain.py:78-88 construct_input()
        # This converts the image to ~395 VQGAN token IDs + sentinel token 8193
        prompts = [{'image': [image], 'question': task_description}]
        batch = self.model.construct_input(prompts)  # Returns dict with 'input_ids' key
        # batch['input_ids'].shape = (1, ~396)  where last token is 8193 (sentinel)

        # STEP 2: Create text prompt and tokenize
        # Source: sampler_latent_pretrain.py:218-221 __call__()
        # Format: "<s> <s> You are a helpful assistant. USER: What action should the robot take to `{question}` ASSISTANT: <vision>"
        # We use the actual task description from the dataset if available, replacing underscores with spaces
        task_desc_clean = task_description.replace('_', ' ') if task_description else 'action'
        text_prompt = f"<s> <s> You are a helpful assistant. USER: What action should the robot take to {task_desc_clean} ASSISTANT: <vision>"


        # Tokenize using prefix_tokenizer (left-padding)
        inputs = self.model.prefix_tokenizer(
            [text_prompt],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='np'
        )

        # Continuation after vision tokens
        # Source: sampler_latent_pretrain.py:189-193 generate_video_pred()
        prefix_for_gen = ["</vision> <delta>"]
        inputs_for_gen = self.model.prefix_tokenizer(
            prefix_for_gen,
            return_tensors='np'
        )

        # STEP 3: Assemble full input with proper masking
        # Source: sampler_latent_pretrain.py:195-208 generate_video_pred()
        # Structure: [text_tokens] + [vision_tokens] + [continuation]
        full_input_ids = np.concatenate(
            [inputs.input_ids, batch['input_ids'], inputs_for_gen.input_ids],
            axis=1
        )
        full_attention_mask = np.concatenate(
            [inputs.attention_mask, np.ones(batch['input_ids'].shape, dtype=inputs.attention_mask.dtype),
             inputs_for_gen.attention_mask],
            axis=1
        )

        # Create masks (source: lines 198-207)
        # vision_masks: 1 where vision tokens are, 0 elsewhere
        vision_masks = np.concatenate([
            np.zeros(inputs.input_ids.shape, dtype=bool),
            np.ones(batch['input_ids'].shape, dtype=bool),
            np.zeros(inputs_for_gen.input_ids.shape, dtype=bool)
        ], axis=1)

        # delta_masks: all zeros during prefill (no delta tokens yet)
        delta_masks = np.concatenate([
            np.zeros(inputs.input_ids.shape, dtype=bool),
            np.zeros(batch['input_ids'].shape, dtype=bool),
            np.zeros(inputs_for_gen.input_ids.shape, dtype=bool),
        ], axis=1)

        # STEP 4: Run transformer forward pass
        # Call the model's module.apply() with parameters (Flax pattern)
        # Source: delta_llama.py:410-448 for signature, delta_llama.py:499-500 for module_class
        import jax.numpy as jnp

        with self.model.mesh:
            outputs = self.model.model.module.apply(
                {"params": self.model.params["params"]},
                jnp.array(full_input_ids, dtype=jnp.int32),
                jnp.array(vision_masks, dtype=jnp.bool_),
                jnp.array(delta_masks, dtype=jnp.bool_),
                jnp.array(full_attention_mask, dtype=jnp.int32),
                None,   # segment_ids
                None,   # position_ids
                True,   # deterministic
                False,  # init_cache
                False,  # output_attentions
                True,   # output_hidden_states
                True,   # return_dict
            )


        # STEP 5: Extract all hidden states from final layer (text + vision + continuation)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # outputs.hidden_states[-1] = final transformer layer
            # Shape: [batch=1, seq_len=text+vision+continuation, hidden_dim=4096]
            # Extract entire sequence including text and vision tokens for richer representations
            hidden_states = outputs.hidden_states[-1][0, :, :]
        else:
            raise RuntimeError(
                "Hidden states not in model output. Verify output_hidden_states=True."
            )

        # STEP 6: Move to CPU and convert to float32
        hidden_states = jax.device_get(hidden_states)
        hidden_states = hidden_states.astype(np.float32)

        return hidden_states


# ============================================================
# MAIN EXTRACTION
# ============================================================

@draccus.wrap()
def extract_lapa_hidden_states(cfg: ExtractLAPAConfig) -> None:
    """Extract LAPA hidden states from Libero spatial dataset (HuggingFace parquet format).

    The dataset should be downloaded from:
    https://huggingface.co/datasets/aopolin-lv/libero_spatial_no_noops_lerobot_v21

    Each frame is paired with its task_index, which is mapped to the actual task description
    from the libero_spatial task suite.
    """

    print("\n" + "="*70)
    print("LAPA Aggregated Hidden State Extraction")
    print("="*70 + "\n")

    # Validate dataset directory
    dataset_dir = Path(cfg.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Aggregation method: {cfg.aggregation_method}")
    print(f"Frame stride: {cfg.frame_stride} (processing every {cfg.frame_stride}th frame)\n")

    # Load LAPA extractor
    print("Initializing LAPA...")
    extractor = LAPAHiddenStateExtractor(cfg)

    # Find parquet files without loading entire dataset
    print(f"\nDiscovering dataset structure from: {dataset_dir}")
    import glob
    import pandas as pd

    data_dir = dataset_dir / 'data'
    parquet_files = sorted(glob.glob(str(data_dir / '**' / '*.parquet'), recursive=True))

    if not parquet_files:
        print(f"❌ No parquet files found in {data_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files")

    # Build episode index and global index map by reading parquet files in streaming mode
    print(f"\nBuilding episode index and global index map (streaming)...")
    episodes_metadata = {}  # episode_idx -> {task_index, max_frame_idx}
    global_index_map = {}   # (episode_idx, frame_idx) -> global_idx
    total_records = 0

    for parquet_file in parquet_files:
        # Read necessary columns
        df = pd.read_parquet(parquet_file, columns=['episode_index', 'task_index', 'frame_index', 'index'])
        for idx, row in df.iterrows():
            episode_idx = int(row['episode_index'])
            task_idx = int(row['task_index'])
            frame_idx = int(row['frame_index'])
            global_idx = int(row['index'])

            if episode_idx not in episodes_metadata:
                episodes_metadata[episode_idx] = {
                    'task_index': task_idx,
                    'max_frame_idx': -1
                }

            episodes_metadata[episode_idx]['max_frame_idx'] = max(episodes_metadata[episode_idx]['max_frame_idx'], frame_idx)
            global_index_map[(episode_idx, frame_idx)] = global_idx
            total_records += 1
        del df  # Explicitly delete to free memory

    print(f"✓ Found {len(episodes_metadata)} episodes with {total_records} total records")

    # Process episodes with incremental saving
    hidden_states_list = []
    episode_indices_list = []
    frame_indices_list = []
    global_indices_list = []
    task_descriptions_list = []
    extracted_count = 0
    batch_count = 0
    save_interval = 50  # Save every N episodes

    print(f"\nExtracting aggregated hidden states for ALL frames...")
    print(f"(Auto-saving every {save_interval} episodes to manage RAM)")

    episode_list = sorted(episodes_metadata.items())
    if cfg.num_episodes is not None:
        episode_list = episode_list[:cfg.num_episodes]

    pbar = tqdm.tqdm(total=len(episode_list), desc="Episodes")

    for episode_count, (episode_idx, episode_meta) in enumerate(episode_list, 1):
        pbar.update(1)

        # Get task description from metadata (no record loading needed)
        task_index = episode_meta['task_index']
        max_frame_idx = episode_meta['max_frame_idx']

        if task_index < len(LIBERO_SPATIAL_TASKS):
            task_description = LIBERO_SPATIAL_TASKS[task_index]
        else:
            task_description = 'unknown'

        # Find video file for this episode
        video_path = None
        videos_dir = dataset_dir / 'videos'
        if videos_dir.exists():
            for chunk_dir in sorted(videos_dir.glob('chunk-*')):
                image_dir = chunk_dir / 'observation.images.image'
                if image_dir.exists():
                    potential_video = image_dir / f'episode_{episode_idx:06d}.mp4'
                    if potential_video.exists():
                        video_path = potential_video
                        break

        if not video_path:
            raise RuntimeError(f"❌ FATAL: Video not found for episode {episode_idx}")

        # Extract ALL frames from video in a single efficient pass (not per-frame)
        frames = extract_all_frames_from_video(video_path, max_frame_idx)

        # Process frames at stride intervals: 0, stride, 2*stride, ...
        for frame_idx in range(0, max_frame_idx + 1, cfg.frame_stride):
            if frame_idx not in frames:
                # Frame not extracted (shouldn't happen normally)
                print(f"Warning: Frame {frame_idx} missing from episode {episode_idx}")
                continue

            frame = frames[frame_idx]

            # Extract hidden states [seq_len, 4096]
            hidden_state_seq = extractor.extract_hidden_state_from_image(frame, task_description)

            # Verify shape
            if hidden_state_seq.ndim != 2 or hidden_state_seq.shape[1] != 4096:
                raise RuntimeError(
                    f"❌ FATAL: Unexpected hidden state shape for episode {episode_idx} frame {frame_idx}. "
                    f"Got {hidden_state_seq.shape}, expected [seq_len, 4096]"
                )

            # Aggregate sequence [seq_len, 4096] → [4096]
            aggregated_state = aggregate_sequence(hidden_state_seq, cfg.aggregation_method)

            # Look up global index for this (episode, frame) pair
            key = (episode_idx, frame_idx)
            if key in global_index_map:
                global_idx = global_index_map[key]
            else:
                # If not found, use -1 as sentinel
                global_idx = -1

            hidden_states_list.append(aggregated_state)
            episode_indices_list.append(episode_idx)
            frame_indices_list.append(frame_idx)
            global_indices_list.append(global_idx)
            task_descriptions_list.append(task_description)
            extracted_count += 1

        # Incremental save every N episodes to manage RAM
        if episode_count % save_interval == 0 and hidden_states_list:
            batch_count += 1
            output_file = output_dir / f"lapa_hidden_states_stride_{cfg.frame_stride}.h5"

            try:
                import h5py
            except ImportError:
                print("❌ h5py not installed. Install with: pip install h5py")
                return

            # Append to HDF5 file (store aggregated 1D hidden states [4096])
            with h5py.File(output_file, 'a') as f:
                # Create datasets if they don't exist
                if 'hidden_states' not in f:
                    f.create_dataset('hidden_states', (0, 4096), maxshape=(None, 4096),
                                   dtype=np.float32, chunks=(1000, 4096), compression='gzip')
                    f.create_dataset('episode_indices', (0,), maxshape=(None,),
                                   dtype=np.int32, chunks=10000, compression='gzip')
                    f.create_dataset('frame_indices', (0,), maxshape=(None,),
                                   dtype=np.int32, chunks=10000, compression='gzip')
                    f.create_dataset('global_indices', (0,), maxshape=(None,),
                                   dtype=np.int32, chunks=10000, compression='gzip')
                    f.create_dataset('task_descriptions', (0,), maxshape=(None,),
                                   dtype=h5py.string_dtype(encoding='utf-8'), chunks=10000, compression='gzip')

                # Append new data
                current_size = len(f['hidden_states'])
                f['hidden_states'].resize(current_size + len(hidden_states_list), axis=0)
                f['episode_indices'].resize(current_size + len(hidden_states_list), axis=0)
                f['frame_indices'].resize(current_size + len(hidden_states_list), axis=0)
                f['global_indices'].resize(current_size + len(hidden_states_list), axis=0)
                f['task_descriptions'].resize(current_size + len(hidden_states_list), axis=0)

                f['hidden_states'][current_size:current_size+len(hidden_states_list)] = np.array(hidden_states_list, dtype=np.float32)
                f['episode_indices'][current_size:current_size+len(hidden_states_list)] = np.array(episode_indices_list, dtype=np.int32)
                f['frame_indices'][current_size:current_size+len(hidden_states_list)] = np.array(frame_indices_list, dtype=np.int32)
                f['global_indices'][current_size:current_size+len(hidden_states_list)] = np.array(global_indices_list, dtype=np.int32)
                f['task_descriptions'][current_size:current_size+len(hidden_states_list)] = np.array(task_descriptions_list, dtype=object)

            print(f"\n✅ Batch {batch_count} saved: {current_size + len(hidden_states_list)} total samples to {output_file.name}")

            # Clear lists to free RAM
            hidden_states_list = []
            episode_indices_list = []
            frame_indices_list = []
            global_indices_list = []
            task_descriptions_list = []

    pbar.close()

    # Save any remaining data
    if hidden_states_list:
        output_file = output_dir / "lapa_hidden_states.h5"

        try:
            import h5py
        except ImportError:
            print("❌ h5py not installed. Install with: pip install h5py")
            return

        with h5py.File(output_file, 'a') as f:
            # Create datasets if they don't exist
            if 'hidden_states' not in f:
                f.create_dataset('hidden_states', (0, 4096), maxshape=(None, 4096),
                               dtype=np.float32, chunks=(1000, 4096), compression='gzip')
                f.create_dataset('episode_indices', (0,), maxshape=(None,),
                               dtype=np.int32, chunks=10000, compression='gzip')
                f.create_dataset('frame_indices', (0,), maxshape=(None,),
                               dtype=np.int32, chunks=10000, compression='gzip')
                f.create_dataset('global_indices', (0,), maxshape=(None,),
                               dtype=np.int32, chunks=10000, compression='gzip')
                f.create_dataset('task_descriptions', (0,), maxshape=(None,),
                               dtype=h5py.string_dtype(encoding='utf-8'), chunks=10000, compression='gzip')

            # Append new data
            current_size = len(f['hidden_states'])
            f['hidden_states'].resize(current_size + len(hidden_states_list), axis=0)
            f['episode_indices'].resize(current_size + len(hidden_states_list), axis=0)
            f['frame_indices'].resize(current_size + len(hidden_states_list), axis=0)
            f['global_indices'].resize(current_size + len(hidden_states_list), axis=0)
            f['task_descriptions'].resize(current_size + len(hidden_states_list), axis=0)

            f['hidden_states'][current_size:current_size+len(hidden_states_list)] = np.array(hidden_states_list, dtype=np.float32)
            f['episode_indices'][current_size:current_size+len(hidden_states_list)] = np.array(episode_indices_list, dtype=np.int32)
            f['frame_indices'][current_size:current_size+len(hidden_states_list)] = np.array(frame_indices_list, dtype=np.int32)
            f['global_indices'][current_size:current_size+len(hidden_states_list)] = np.array(global_indices_list, dtype=np.int32)
            f['task_descriptions'][current_size:current_size+len(hidden_states_list)] = np.array(task_descriptions_list, dtype=object)

            final_size = len(f['hidden_states'])

        print(f"\n✅ Final batch saved: {final_size} total samples to {output_file.name}")

    print(f"\n{'='*70}")
    print(f"✅ Extraction Complete")
    print(f"{'='*70}")
    print(f"Successfully extracted: {extracted_count} aggregated hidden states")
    print(f"Aggregation method: {cfg.aggregation_method}")
    print(f"\nSaved to: {output_dir}/lapa_hidden_states.h5 (HDF5 format)")
    print(f"  - hidden_states: [{extracted_count}, 4096] aggregated states")
    print(f"  - episode_indices: [{extracted_count}] episode IDs")
    print(f"  - frame_indices: [{extracted_count}] frame indices (0, 1, 2, ...)")
    print(f"  - global_indices: [{extracted_count}] global dataset indices (0..52969)")
    print(f"  - task_descriptions: [{extracted_count}] task descriptions")
    print(f"\nDirect access: import h5py; f=h5py.File(...); hs=f['hidden_states'][i]; f.close()")


if __name__ == "__main__":
    extract_lapa_hidden_states()
