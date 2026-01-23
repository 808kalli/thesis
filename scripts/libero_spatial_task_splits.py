"""
Generate episode lists for LIBERO Spatial task-level holdout splits.

This script creates different train/test splits by holding out different numbers of tasks:
- Hold out 1 task (task 9): 9 training tasks, 1 test task
- Hold out 3 tasks (7,8,9): 7 training tasks, 3 test tasks
- Hold out 5 tasks (5,6,7,8,9): 5 training tasks, 5 test tasks
- Hold out 8 tasks (2-9): 2 training tasks, 8 test tasks

Usage:
    python scripts/libero_spatial_task_splits.py

Output:
    Creates libero_spatial_task_splits.json with episode lists for each configuration
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


def create_split(episodes_by_task_idx, held_out_tasks):
    """
    Create train/test split by holding out specified tasks.

    Args:
        episodes_by_task_idx: Dict mapping task_idx -> list of episode IDs
        held_out_tasks: List of task indices to hold out

    Returns:
        dict with 'train_episodes', 'test_episodes', 'train_tasks', 'test_tasks'
    """
    train_episodes = []
    test_episodes = []
    train_tasks = []
    test_tasks = []

    for task_idx in range(10):
        if task_idx in held_out_tasks:
            test_episodes.extend(episodes_by_task_idx[task_idx])
            test_tasks.append(task_idx)
        else:
            train_episodes.extend(episodes_by_task_idx[task_idx])
            train_tasks.append(task_idx)

    return {
        'train_episodes': sorted(train_episodes),
        'test_episodes': sorted(test_episodes),
        'train_tasks': train_tasks,
        'test_tasks': held_out_tasks,
        'num_train_episodes': len(train_episodes),
        'num_test_episodes': len(test_episodes),
        'num_train_tasks': len(train_tasks),
        'num_test_tasks': len(held_out_tasks),
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

    # Define split configurations - hold out last N tasks
    splits = {
        'holdout_1_task': {
            'description': 'Hold out 1 task (task 9) for zero-shot evaluation',
            'held_out_tasks': [9],
        },
        'holdout_3_tasks': {
            'description': 'Hold out 3 tasks (7,8,9) for zero-shot evaluation',
            'held_out_tasks': [7, 8, 9],
        },
        'holdout_5_tasks': {
            'description': 'Hold out 5 tasks (5,6,7,8,9) for zero-shot evaluation',
            'held_out_tasks': [5, 6, 7, 8, 9],
        },
        'holdout_7_tasks': {
            'description': 'Hold out 7 tasks (3,4,5,6,7,8,9) for zero-shot evaluation',
            'held_out_tasks': [3, 4, 5, 6, 7, 8, 9],
        },
        'holdout_9_tasks': {
            'description': 'Hold out 9 tasks (1-9) - extreme zero-shot (only 1 training task)',
            'held_out_tasks': list(range(1, 10)),
        },
    }

    # Generate all splits
    results = {}
    for split_name, config in splits.items():
        print(f"\nGenerating split: {split_name}")
        print(f"  Description: {config['description']}")
        print(f"  Held-out tasks: {config['held_out_tasks']}")

        split_data = create_split(episodes_by_task_idx, config['held_out_tasks'])
        split_data['description'] = config['description']
        split_data['held_out_task_names'] = [index_to_task[idx] for idx in config['held_out_tasks']]

        results[split_name] = split_data

        print(f"  → Training: {split_data['num_train_episodes']} episodes from {split_data['num_train_tasks']} tasks")
        print(f"  → Testing:  {split_data['num_test_episodes']} episodes from {split_data['num_test_tasks']} tasks")

    # Save to JSON file with compact list formatting
    output_file = Path(__file__).parent / "libero_spatial_task_splits.json"

    # Custom JSON formatting: compact arrays, indented objects
    import json
    json_str = json.dumps(results, indent=2)

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
    print(f"✓ Saved all splits to: {output_file}")
    print("=" * 80)

    # Print usage instructions
    print("\nUSAGE IN lerobot_dataset.py:")
    print("=" * 80)
    print("""
import json

# Load the desired split
with open('scripts/libero_spatial_task_splits.json') as f:
    splits = json.load(f)

# Choose which split to use
manual_ids = splits['holdout_1_task']['train_episodes']  # 9 training tasks
# OR: manual_ids = splits['holdout_3_tasks']['train_episodes']  # 7 training tasks
# OR: manual_ids = splits['holdout_5_tasks']['train_episodes']  # 5 training tasks
# OR: manual_ids = splits['holdout_7_tasks']['train_episodes']  # 3 training tasks
# OR: manual_ids = splits['holdout_9_tasks']['train_episodes']  # 1 training task
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
