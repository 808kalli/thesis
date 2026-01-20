"""
Find one episode per task for LIBERO-Object dataset.

This script scans the libero_object dataset to find one representative episode
for each of the 10 tasks (task_index 0-9).
"""

import pandas as pd
from pathlib import Path
import sys

DATASET_DIR = Path("/workspace/thesis/raw_datasets/libero_object_noops")

# LIBERO-Object tasks (from libero_suite_task_map.py)
LIBERO_OBJECT_TASKS = [
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",           # task 0
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",            # task 1
    "pick_up_the_salad_dressing_and_place_it_in_the_basket",          # task 2
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",               # task 3
    "pick_up_the_ketchup_and_place_it_in_the_basket",                 # task 4
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",            # task 5
    "pick_up_the_butter_and_place_it_in_the_basket",                  # task 6
    "pick_up_the_milk_and_place_it_in_the_basket",                    # task 7
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",       # task 8
    "pick_up_the_orange_juice_and_place_it_in_the_basket",            # task 9
]

def main():
    data_dir = DATASET_DIR / "data"

    if not data_dir.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        print(f"Please download the dataset first using:")
        print(f"huggingface-cli download aopolin-lv/libero_object_no_noops_lerobot_v21 \\")
        print(f"  --repo-type dataset \\")
        print(f"  --local-dir {DATASET_DIR} \\")
        print(f"  --local-dir-use-symlinks False")
        sys.exit(1)

    parquet_files = sorted(data_dir.glob("**/*.parquet"))

    if not parquet_files:
        print(f"❌ No parquet files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet files")
    print("\nScanning for one episode per task...\n")

    # Find first episode for each task
    task_to_episode = {}

    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=['episode_index', 'task_index'])

        for task_idx in range(10):
            if task_idx not in task_to_episode:
                # Find first episode with this task_index
                matching = df[df['task_index'] == task_idx]
                if not matching.empty:
                    episode_idx = int(matching.iloc[0]['episode_index'])
                    task_to_episode[task_idx] = episode_idx
                    print(f"Task {task_idx}: Episode {episode_idx:3d} - {LIBERO_OBJECT_TASKS[task_idx]}")

        # Stop if we found all tasks
        if len(task_to_episode) == 10:
            break

    if len(task_to_episode) < 10:
        print(f"\n⚠️  Warning: Only found {len(task_to_episode)}/10 tasks")
    else:
        print(f"\n✅ Found episodes for all 10 tasks")

    # Print summary
    print("\n" + "="*70)
    print("Episode list for extract_lapa_hidden_states.py:")
    print("="*70)
    episode_list = [task_to_episode[i] for i in range(10) if i in task_to_episode]
    print(f"Episodes: {episode_list}")
    print(f"\nUse with: --episode_list {' '.join(map(str, episode_list))}")
    print("="*70)

if __name__ == "__main__":
    main()
