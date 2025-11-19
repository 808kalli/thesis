"""
extract_lapa_hidden_states.py

Extracts LAPA hidden states from Libero spatial dataset frames.

LAPA was trained on frames ~0.6 seconds apart. Since Libero has 0.05s increments,
we sample every 12 frames (0.6 / 0.05 = 12) to match LAPA's training distribution.

The dataset is from HuggingFace (lerobot format, parquet files):
https://huggingface.co/datasets/aopolin-lv/libero_spatial_no_noops_lerobot_v21

Usage:
python latent_pretraining/extract_lapa_hidden_states.py \
    --dataset_dir /workspace/thesis/raw_datasets/libero_spatial \
    --output_dir /workspace/thesis \
    --vqgan_checkpoint lapa_checkpoints/vqgan \
    --load_checkpoint params::lapa_checkpoints/params_sthv2 \
    --num_episodes 1 \
    --seed 7

Arguments:
    --dataset_dir: Local path to libero_spatial dataset directory (HuggingFace parquet format)
    --output_dir: Directory to save hidden states .npy files
    --vqgan_checkpoint: Path to VQGAN checkpoint
    --load_checkpoint: Path to LAPA model checkpoint
    --load_llama_config: LAPA config (default: 7b)
    --num_episodes: Maximum number of episodes to process (None = all)
    --seed: Random seed

Outputs:
    {output_dir}/lapa_hidden_states.npy containing:
    - hidden_states: array of shape [num_samples] with variable [seq_len, 4096] arrays
    - episode_indices: array of episode indices
    - frame_indices: array of frame indices (0, 12, 24, ...)
    - task_descriptions: array of task description strings
"""

import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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
    """Configuration for extracting LAPA hidden states."""
    # fmt: off

    # Dataset parameters (HuggingFace parquet format)
    dataset_dir: Union[str, Path] = "/path/to/libero_spatial"

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
    frame_stride: int = 12  # Process every 12 frames (0.6s at 20fps = ~0.05s per frame in Libero)

    # Output parameters
    output_dir: Union[str, Path] = "lapa_hidden_states"

    # fmt: on


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
    print("LAPA Hidden State Extraction from Libero Spatial Dataset")
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
    print(f"Frame stride: {cfg.frame_stride} (matches LAPA's ~0.6s training interval)")
    print(f"Expected format: HuggingFace parquet with task_index mapping\n")

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

    # Build episode index by reading parquet files in streaming mode
    print(f"\nBuilding episode index (streaming)...")
    episodes_metadata = {}  # episode_idx -> (parquet_file, row_indices, num_frames)
    total_records = 0

    for parquet_file in parquet_files:
        # Read only episode_index and task_index columns to build index
        df = pd.read_parquet(parquet_file, columns=['episode_index', 'task_index'])
        for idx, row in df.iterrows():
            episode_idx = int(row['episode_index'])
            task_idx = int(row['task_index'])
            if episode_idx not in episodes_metadata:
                episodes_metadata[episode_idx] = {
                    'parquet_file': parquet_file,
                    'indices': [],
                    'task_index': task_idx,
                    'num_frames': 0
                }
            episodes_metadata[episode_idx]['indices'].append(idx)
            episodes_metadata[episode_idx]['num_frames'] += 1
            total_records += 1
        del df  # Explicitly delete to free memory

    print(f"✓ Found {len(episodes_metadata)} episodes with {total_records} total records")

    # Process episodes with incremental saving every 10 episodes
    hidden_states_list = []
    episode_indices_list = []
    frame_indices_list = []
    task_descriptions_list = []
    seq_lengths_list = []
    extracted_count = 0
    batch_count = 0
    save_interval = 50  # Save every N episodes

    print(f"\nExtracting hidden states (sampling every {cfg.frame_stride} frames)...")
    print(f"(Auto-saving every {save_interval} episodes to manage RAM)")

    episode_list = sorted(episodes_metadata.items())
    if cfg.num_episodes is not None:
        episode_list = episode_list[:cfg.num_episodes]

    pbar = tqdm.tqdm(total=len(episode_list), desc="Episodes")

    for episode_count, (episode_idx, episode_meta) in enumerate(episode_list, 1):
        pbar.update(1)

        # Get task description from metadata (no record loading needed)
        task_index = episode_meta['task_index']
        num_frames = episode_meta['num_frames']

        if task_index < len(LIBERO_SPATIAL_TASKS):
            task_description = LIBERO_SPATIAL_TASKS[task_index]
        else:
            print(f"\n  ⚠️  Task index {task_index} out of range, using generic prompt")
            task_description = None

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

        # Process frames at stride: 0, 12, 24, ...
        frame_idx = 0
        while frame_idx < num_frames:
            # Extract frame from video at frame_idx using ffmpeg (supports AV1)
            import subprocess

            # Use ffmpeg to extract frame at index
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f'select=eq(n\\,{frame_idx})',
                '-vframes', '1',
                '-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-'
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=5)

            if result.returncode != 0:
                raise RuntimeError(
                    f"❌ FATAL: ffmpeg failed for episode {episode_idx} frame {frame_idx}. "
                    f"stderr: {result.stderr.decode()[:200]}"
                )

            # Convert bytes to PIL Image
            from io import BytesIO
            frame = Image.open(BytesIO(result.stdout)).convert('RGB')

            # Extract hidden states [seq_len, 4096]
            hidden_state = extractor.extract_hidden_state_from_image(frame, task_description)

            # Verify shape
            if hidden_state.ndim != 2 or hidden_state.shape[1] != 4096:
                raise RuntimeError(
                    f"❌ FATAL: Unexpected hidden state shape for episode {episode_idx} frame {frame_idx}. "
                    f"Got {hidden_state.shape}, expected [seq_len, 4096]"
                )

            hidden_states_list.append(hidden_state)
            seq_lengths_list.append(hidden_state.shape[0])
            episode_indices_list.append(episode_idx)
            frame_indices_list.append(frame_idx)
            task_descriptions_list.append(task_description or 'unknown')
            extracted_count += 1

            frame_idx += cfg.frame_stride

        # Incremental save every N episodes to manage RAM
        if episode_count % save_interval == 0 and hidden_states_list:
            batch_count += 1
            output_file = output_dir / "lapa_hidden_states.h5"

            try:
                import h5py
            except ImportError:
                print("❌ h5py not installed. Install with: pip install h5py")
                return

            # Append to HDF5 file (store 2D hidden states without copying)
            with h5py.File(output_file, 'a') as f:
                # Create datasets if they don't exist
                if 'hidden_states' not in f:
                    # Variable-length ragged array for 2D hidden states [seq_len, 4096]
                    vlen_float = h5py.vlen_dtype(np.float32)
                    f.create_dataset('hidden_states', (0,), maxshape=(None,), dtype=vlen_float, chunks=True)
                    f.create_dataset('episode_indices', (0,), maxshape=(None,), dtype=np.int32, chunks=True)
                    f.create_dataset('frame_indices', (0,), maxshape=(None,), dtype=np.int32, chunks=True)
                    f.create_dataset('seq_lengths', (0,), maxshape=(None,), dtype=np.int32, chunks=True)
                    f.create_dataset('task_descriptions', (0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'), chunks=True)

                # Append new data
                current_size = len(f['hidden_states'])
                f['hidden_states'].resize(current_size + len(hidden_states_list), axis=0)
                f['episode_indices'].resize(current_size + len(hidden_states_list), axis=0)
                f['frame_indices'].resize(current_size + len(hidden_states_list), axis=0)
                f['seq_lengths'].resize(current_size + len(hidden_states_list), axis=0)
                f['task_descriptions'].resize(current_size + len(hidden_states_list), axis=0)

                for i, hidden_state in enumerate(hidden_states_list):
                    # Flatten 2D array [seq_len, 4096] and store in variable-length dataset
                    f['hidden_states'][current_size + i] = hidden_state.astype(np.float32).flatten()
                    f['episode_indices'][current_size + i] = episode_indices_list[i]
                    f['frame_indices'][current_size + i] = frame_indices_list[i]
                    f['seq_lengths'][current_size + i] = seq_lengths_list[i]
                    f['task_descriptions'][current_size + i] = task_descriptions_list[i]

            print(f"\n✅ Batch {batch_count} saved: {current_size + len(hidden_states_list)} total samples to {output_file.name}")

            # Clear lists to free RAM
            hidden_states_list = []
            episode_indices_list = []
            frame_indices_list = []
            task_descriptions_list = []
            seq_lengths_list = []

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
                vlen_float = h5py.vlen_dtype(np.float32)
                f.create_dataset('hidden_states', (0,), maxshape=(None,), dtype=vlen_float, chunks=True)
                f.create_dataset('episode_indices', (0,), maxshape=(None,), dtype=np.int32, chunks=True)
                f.create_dataset('frame_indices', (0,), maxshape=(None,), dtype=np.int32, chunks=True)
                f.create_dataset('seq_lengths', (0,), maxshape=(None,), dtype=np.int32, chunks=True)
                f.create_dataset('task_descriptions', (0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'), chunks=True)

            # Append new data
            current_size = len(f['hidden_states'])
            f['hidden_states'].resize(current_size + len(hidden_states_list), axis=0)
            f['episode_indices'].resize(current_size + len(hidden_states_list), axis=0)
            f['frame_indices'].resize(current_size + len(hidden_states_list), axis=0)
            f['seq_lengths'].resize(current_size + len(hidden_states_list), axis=0)
            f['task_descriptions'].resize(current_size + len(hidden_states_list), axis=0)

            for i, hidden_state in enumerate(hidden_states_list):
                # Flatten 2D array [seq_len, 4096] and store in variable-length dataset
                f['hidden_states'][current_size + i] = hidden_state.astype(np.float32).flatten()
                f['episode_indices'][current_size + i] = episode_indices_list[i]
                f['frame_indices'][current_size + i] = frame_indices_list[i]
                f['seq_lengths'][current_size + i] = seq_lengths_list[i]
                f['task_descriptions'][current_size + i] = task_descriptions_list[i]

            final_size = len(f['hidden_states'])

        print(f"\n✅ Final batch saved: {final_size} total samples to {output_file.name}")

    print(f"\n{'='*70}")
    print(f"✅ Extraction Complete")
    print(f"{'='*70}")
    print(f"Successfully extracted: {extracted_count} hidden states")
    print(f"\nSaved to: {output_dir}/lapa_hidden_states.h5 (HDF5 format)")
    print(f"  - Contains: hidden_states [seq_len, 4096], episode_indices, frame_indices, seq_lengths, task_descriptions")
    print(f"  - Direct access: import h5py; f=h5py.File(...); hs=f['hidden_states'][i]; f.close()")


if __name__ == "__main__":
    extract_lapa_hidden_states()
