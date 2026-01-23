"""
Generate few-shot episode lists for LIBERO Spatial.

This script creates a split with only 1 episode per task for training (10 total),
with all remaining episodes reserved for testing. This tests the model's ability
to learn from very few examples per task.

Usage:
    python scripts/libero_spatial_few_shot_split.py

Output:
    Creates libero_spatial_few_shot_split.json with 1 episode per task for training
"""

import json
from collections import defaultdict
from pathlib import Path

# Path to dataset metadata
DATASET_CACHE = Path.home() / ".cache/huggingface/lerobot/aopolin-lv/libero_spatial_no_noops_lerobot_v21/meta"

def load_task_episode_mapping():
    """Load episodes grouped by task."""
    episodes_by_task = defaultdict(list)

    with open(DATASET_CACHE / "episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            task_name = ep['tasks'][0]
            episodes_by_task[task_name].append(ep['episode_index'])

    # Load task indices
    task_to_index = {}
    with open(DATASET_CACHE / "tasks.jsonl") as f:
        for line in f:
            task = json.loads(line)
            task_to_index[task['task']] = task['task_index']

    # Create index to task name mapping
    index_to_task = {idx: name for name, idx in task_to_index.items()}

    # Reformat as task_index -> episode_list
    episodes_by_task_idx = {}
    for task_name, ep_list in episodes_by_task.items():
        task_idx = task_to_index[task_name]
        episodes_by_task_idx[task_idx] = sorted(ep_list)

    return episodes_by_task_idx, index_to_task


def create_few_shot_split(episodes_by_task_idx, episodes_per_task=1):
    """
    Create train/test split with only N episodes per task for training.

    Args:
        episodes_by_task_idx: Dict mapping task_idx -> list of episode IDs
        episodes_per_task: Number of episodes per task to use for training (default: 1)

    Returns:
        dict with 'train_episodes', 'test_episodes', 'episodes_per_task'
    """
    train_episodes = []
    test_episodes = []

    for task_idx in range(10):
        task_episodes = episodes_by_task_idx[task_idx]

        # Take first N episodes for training, rest for testing
        train_episodes.extend(task_episodes[:episodes_per_task])
        test_episodes.extend(task_episodes[episodes_per_task:])

    return {
        'train_episodes': sorted(train_episodes),
        'test_episodes': sorted(test_episodes),
        'num_train_episodes': len(train_episodes),
        'num_test_episodes': len(test_episodes),
        'episodes_per_task': episodes_per_task,
        'num_tasks': 10,
    }


def main():
    # Load data
    episodes_by_task_idx, index_to_task = load_task_episode_mapping()

    # Print task info
    print("=" * 80)
    print("LIBERO Spatial - Task Episode Counts")
    print("=" * 80)
    total_episodes = 0
    for task_idx in range(10):
        task_name = index_to_task[task_idx]
        num_episodes = len(episodes_by_task_idx[task_idx])
        total_episodes += num_episodes
        print(f"Task {task_idx}: {num_episodes} episodes")
        print(f"  {task_name}")
    print(f"\nTotal: {total_episodes} episodes across 10 tasks")
    print("=" * 80)
    print()

    # Generate multiple few-shot splits
    splits = {}
    episodes_configs = [1, 3, 5, 7, 9]

    for n_episodes in episodes_configs:
        split_name = f"{n_episodes}_per_task"
        print(f"\nGenerating split: {n_episodes} episode(s) per task...")

        split_data = create_few_shot_split(episodes_by_task_idx, episodes_per_task=n_episodes)
        split_data['description'] = f'Few-shot training: {n_episodes} episode(s) per task ({n_episodes * 10} total training episodes)'
        split_data['task_names'] = [index_to_task[idx] for idx in range(10)]

        splits[split_name] = split_data

        print(f"  → Training: {split_data['num_train_episodes']} episodes ({split_data['episodes_per_task']} per task)")
        print(f"  → Testing:  {split_data['num_test_episodes']} episodes")

    # Save to JSON file with compact list formatting
    output_file = Path(__file__).parent / "libero_spatial_few_shot_splits.json"

    # Custom JSON formatting: compact arrays, indented objects
    json_str = json.dumps(splits, indent=2)

    # Replace multi-line arrays with single-line arrays
    import re
    # Match arrays that span multiple lines and compress them
    def compress_array(match):
        content = match.group(1)
        # Remove whitespace and newlines, keep just the values
        values = re.findall(r'\d+', content)
        return '[' + ', '.join(values) + ']'

    json_str = re.sub(r'\[\s*((?:\d+,?\s*)+)\s*\]', compress_array, json_str, flags=re.DOTALL)

    with open(output_file, 'w') as f:
        f.write(json_str)

    print("\n" + "=" * 80)
    print(f"✓ Saved all few-shot splits to: {output_file}")
    print("=" * 80)

    # Print usage instructions
    print("\nUSAGE IN lerobot_dataset.py:")
    print("=" * 80)
    print("""
import json

# Load the few-shot splits
with open('scripts/libero_spatial_few_shot_splits.json') as f:
    splits = json.load(f)

# Choose which split to use
manual_ids = splits['1_per_task']['train_episodes']  # 10 episodes (1 per task)
# OR: manual_ids = splits['3_per_task']['train_episodes']  # 30 episodes (3 per task)
# OR: manual_ids = splits['5_per_task']['train_episodes']  # 50 episodes (5 per task)
# OR: manual_ids = splits['7_per_task']['train_episodes']  # 70 episodes (7 per task)
# OR: manual_ids = splits['9_per_task']['train_episodes']  # 90 episodes (9 per task)
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
