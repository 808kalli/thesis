import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import argparse
from datasets import load_dataset
from PIL import Image
import tempfile
from collections import defaultdict
import huggingface_hub


def extract_frames_from_video_bytes(video_bytes: bytes) -> List[np.ndarray]:
    """Extract all frames from video bytes using ffmpeg/imageio."""
    import imageio
    
    # Write video bytes to temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        tmp_file.write(video_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        frames = []
        
        # Use imageio with ffmpeg backend (handles AV1 codec)
        reader = imageio.get_reader(tmp_file_path, 'ffmpeg')
        
        for frame in reader:
            # imageio returns RGB frames directly
            frames.append(frame)
        
        reader.close()
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        return frames
    
    except Exception as e:
        print(f"Error extracting frames with imageio: {e}")
        print("Trying alternative method with cv2...")
        
        # Fallback to OpenCV
        try:
            cap = cv2.VideoCapture(tmp_file_path)
            frames = []
            
            if not cap.isOpened():
                raise ValueError("Could not open video with OpenCV")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            cap.release()
            
            if len(frames) == 0:
                raise ValueError("No frames extracted with OpenCV")
            
            return frames
            
        except Exception as cv2_error:
            raise ValueError(f"Failed to extract frames with both imageio and cv2: {e}, {cv2_error}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


def resize_frames(frames: List[np.ndarray], size: tuple = (256, 256)) -> np.ndarray:
    """Resize frames to target size and stack into array."""
    resized = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            pil_image = Image.fromarray(frame)
        else:
            pil_image = frame
        
        pil_image = pil_image.resize(size, Image.LANCZOS)
        resized.append(np.array(pil_image))
    
    return np.array(resized, dtype=np.uint8)


def load_task_metadata(dataset_path: str):
    """Load task metadata from the tasks.jsonl file."""
    
    tasks_file_path = Path(dataset_path) / "meta" / "tasks.jsonl"
    
    if not tasks_file_path.exists():
        print(f"Warning: tasks.jsonl not found at {tasks_file_path}")
        return {}
    
    tasks = {}
    with open(tasks_file_path, 'r') as f:
        for line in f:
            task_data = json.loads(line.strip())
            task_idx = task_data.get('task_index', task_data.get('index'))
            tasks[task_idx] = task_data.get('task', task_data.get('language_instruction', 'Unknown task'))
    
    print(f"Loaded {len(tasks)} task descriptions")
    return tasks


def load_parquet_data(dataset_path: str):
    """Load the dataset from local parquet files."""
    data_dir = Path(dataset_path) / "data" / "chunk-000"
    
    if not data_dir.exists():
        raise ValueError(f"Data directory not found at {data_dir}")
    
    # Find all parquet files
    parquet_files = sorted(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load dataset using datasets library
    from datasets import load_dataset
    dataset = load_dataset("parquet", data_files=str(data_dir / "*.parquet"), split="train")
    
    print(f"Loaded {len(dataset)} timesteps")
    return dataset


def group_timesteps_into_episodes(dataset):
    """Group individual timesteps into episodes based on episode_index."""
    
    print("Grouping timesteps into episodes...")
    episodes_data = defaultdict(list)
    
    # Group by episode_index
    for timestep in tqdm(dataset, desc="Grouping timesteps"):
        episode_idx = timestep['episode_index']
        episodes_data[episode_idx].append({
            'frame_index': timestep['frame_index'],
            'action': timestep['action'],
            'observation_state': timestep['observation.state'],
            'gripper_state': timestep['observation.states.gripper_state'],
            'task_index': timestep['task_index'],
            'timestamp': timestep.get('timestamp', 0)
        })
    
    # Sort timesteps within each episode by frame_index
    for episode_idx in episodes_data:
        episodes_data[episode_idx].sort(key=lambda x: x['frame_index'])
    
    print(f"Found {len(episodes_data)} episodes")
    return episodes_data


def load_video_for_episode(dataset_path: str, episode_idx: int, video_type: str) -> bytes:
    """Load a specific video file for an episode from local storage.
    
    Args:
        dataset_path: Path to local dataset
        episode_idx: Episode index
        video_type: Either 'wrist' or 'context'
    """
    if video_type == 'wrist':
        video_path = Path(dataset_path) / "videos" / "chunk-000" / "observation.images.wrist_image" / f"episode_{episode_idx:06d}.mp4"
    elif video_type == 'context':
        video_path = Path(dataset_path) / "videos" / "chunk-000" / "observation.images.image" / f"episode_{episode_idx:06d}.mp4"
    else:
        raise ValueError(f"Unknown video type: {video_type}")
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    with open(video_path, 'rb') as f:
        return f.read()


def process_episode(episode_idx: int, timesteps: List[Dict], dataset_path: str, tasks: Dict) -> Dict[str, Any]:
    """Process a single episode and create the numpy dictionary."""
    
    # Load videos
    wrist_video_bytes = load_video_for_episode(dataset_path, episode_idx, 'wrist')
    context_video_bytes = load_video_for_episode(dataset_path, episode_idx, 'context')
    
    # Extract frames
    wrist_frames = extract_frames_from_video_bytes(wrist_video_bytes)
    context_frames = extract_frames_from_video_bytes(context_video_bytes)
    
    # Resize frames to 256x256
    wrist_images = resize_frames(wrist_frames, (256, 256))
    context_images = resize_frames(context_frames, (256, 256))
    
    # Extract trajectory data
    actions = []
    states = []
    
    # Get task instruction
    task_index = timesteps[0]['task_index']
    language_instruction = tasks.get(task_index, f"Task {task_index}")
    
    for timestep in timesteps:
        actions.append(timestep['action'])
        states.append(timestep['observation_state'])
    
    # Convert to numpy arrays
    actions = np.array(actions, dtype=np.float32)
    states = np.array(states, dtype=np.float32)
    
    # Ensure all arrays have the same length
    num_timesteps = len(timesteps)
    num_wrist_frames = len(wrist_images)
    num_context_frames = len(context_images)
    
    if num_timesteps != num_wrist_frames or num_timesteps != num_context_frames:
        print(f"Warning: Episode {episode_idx} length mismatch - timesteps: {num_timesteps}, wrist: {num_wrist_frames}, context: {num_context_frames}")
        min_length = min(num_timesteps, num_wrist_frames, num_context_frames)
        actions = actions[:min_length]
        states = states[:min_length]
        wrist_images = wrist_images[:min_length]
        context_images = context_images[:min_length]
    
    # Create episode dictionary following OpenVLA format
    episode_dict = {
        'action': actions,                           # Shape: (T, 7) - robot actions
        'observation': {
            'image': context_images,                 # Shape: (T, 256, 256, 3) - main camera
            'wrist_image': wrist_images,            # Shape: (T, 256, 256, 3) - wrist camera
            'state': states,                        # Shape: (T, 8) - robot state
        },
        'language_instruction': language_instruction,  # String - task description
        'episode_metadata': {
            'episode_index': episode_idx,
            'task_index': task_index,
            'num_steps': len(actions)
        }
    }
    
    return episode_dict


def convert_dataset(
    dataset_path: str,
    output_folder: str,
    max_episodes: int = None,
    output_dataset_name: str = "openvla_libero_object"
):
    """Convert the local dataset to .npy episode format."""
    
    # Load dataset from local parquet files
    dataset = load_parquet_data(dataset_path)
    
    # Group timesteps into episodes
    episodes_data = group_timesteps_into_episodes(dataset)
    
    # Load task metadata
    tasks = load_task_metadata(dataset_path)
    
    # Get episode indices to process
    episode_indices = sorted(episodes_data.keys())
    if max_episodes is not None:
        episode_indices = episode_indices[:max_episodes]
    
    print(f"Will process {len(episode_indices)} episodes")
    
    # Create output directory: output_folder/output_dataset_name/train/
    output_path = Path(output_folder) / output_dataset_name / "train"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_path}")
    
    # Process each episode
    successful_episodes = 0
    
    for episode_idx in tqdm(episode_indices, desc="Processing episodes"):
        try:
            # Get timesteps for this episode
            timesteps = episodes_data[episode_idx]
            
            # Process episode
            episode_dict = process_episode(episode_idx, timesteps, dataset_path, tasks)
            
            # Save as .npy file
            output_file = output_path / f"episode_{episode_idx}.npy"
            np.save(output_file, episode_dict, allow_pickle=True)
            
            successful_episodes += 1
            
        except Exception as e:
            print(f"\nError processing episode {episode_idx}: {str(e)}")
            continue
    
    print(f"\nConversion complete! Created {successful_episodes} episode files in {output_path}")
    
    # Create a README with dataset info
    create_readme(output_path.parent, output_dataset_name, successful_episodes)


def create_readme(output_path: Path, dataset_name: str, num_episodes: int):
    """Create a README file describing the dataset format."""
    
    readme_content = f"""# {dataset_name}

Converted LIBERO Object dataset in .npy episode format for OpenVLA training.

## Dataset Structure

```
{dataset_name}/
└── train/
    ├── episode_0.npy
    ├── episode_1.npy
    ├── ...
    └── episode_N.npy
```

## Episode Format

Each `episode_X.npy` file contains a dictionary with the following structure:

```python
{{
    'action': np.ndarray,                    # Shape: (T, 7) - Robot EEF actions (6D pose + gripper)
    'observation': {{
        'image': np.ndarray,                 # Shape: (T, 256, 256, 3) - Main camera RGB images (uint8)
        'wrist_image': np.ndarray,           # Shape: (T, 256, 256, 3) - Wrist camera RGB images (uint8)
        'state': np.ndarray,                 # Shape: (T, 8) - Robot EEF state (6D pose + 2D gripper)
    }},
    'language_instruction': str,             # Natural language task description
    'episode_metadata': {{
        'episode_index': int,                # Episode index
        'task_index': int,                   # Task type index
        'num_steps': int                     # Number of timesteps in episode
    }}
}}
```

Where `T` is the number of timesteps in the episode (varies per episode).

## Dataset Statistics

- Total episodes: {num_episodes}
- Image resolution: 256x256
- Action dimension: 7 (6D EEF pose + 1D gripper)
- State dimension: 8 (6D EEF pose + 2D gripper)

## Loading an Episode

```python
import numpy as np

# Load episode
episode = np.load('train/episode_0.npy', allow_pickle=True).item()

# Access data
actions = episode['action']  # (T, 7)
images = episode['observation']['image']  # (T, 256, 256, 3)
wrist_images = episode['observation']['wrist_image']  # (T, 256, 256, 3)
states = episode['observation']['state']  # (T, 8)
instruction = episode['language_instruction']  # string

print(f"Instruction: {{instruction}}")
print(f"Number of steps: {{len(actions)}}")
```
"""
    
    readme_path = output_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"README saved to {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert local LIBERO Object dataset to .npy episode format")
    parser.add_argument("--dataset_path", type=str, 
                        default="/home/elias/LAMDA/raw_datasets/openvla_libero_object",
                        help="Path to local dataset directory")
    parser.add_argument("--output_folder", type=str, 
                        default="/home/elias/LAMDA/data",
                        help="Path where to save the converted .npy files")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Maximum number of episodes to process (for testing)")
    parser.add_argument("--output_dataset_name", type=str, default="openvla_libero_object",
                        help="Name for the converted dataset")
    
    args = parser.parse_args()
    
    print(f"Converting dataset from {args.dataset_path}")
    print(f"Output location: {args.output_folder}/{args.output_dataset_name}")
    if args.max_episodes:
        print(f"Max episodes: {args.max_episodes}")
    
    convert_dataset(
        args.dataset_path,
        args.output_folder,
        args.max_episodes,
        args.output_dataset_name
    )


if __name__ == "__main__":
    main()