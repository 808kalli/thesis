import os
import shutil
import numpy as np
import random

# === Paths ===
src = './rlds_dataset_builder/openvla_libero_spatial/data/train'  # current folder with episode_*.npy
dst = './rlds_dataset_builder/openvla_libero_spatial/data/train_100eps'
os.makedirs(dst, exist_ok=True)

# === Collect episodes by task ===
by_task = {}
for f in os.listdir(src):
    if not f.endswith('.npy'):
        continue
    path = os.path.join(src, f)
    data = np.load(path, allow_pickle=True).item()
    task = data.get('language_instruction', '').strip()
    if not task:
        continue
    by_task.setdefault(task, []).append(f)

# === Sample 10 episodes per task ===
sampled = []
for task, files in by_task.items():
    n = min(10, len(files))
    sampled.extend(random.sample(files, n))

# === Shuffle and rename ===
random.shuffle(sampled)
for i, f in enumerate(sampled[:100]):
    new_name = f"episode_{i}.npy"
    shutil.copy(os.path.join(src, f), os.path.join(dst, new_name))

print(f"✅ Collected {len(sampled[:100])} episodes from {len(by_task)} tasks into {dst}")
