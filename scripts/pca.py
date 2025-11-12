import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import torch

# -----------------------------
# Path to teacher latents
# -----------------------------
LATENT_DIR = Path("/home/elias/Thesis/lapa_latents")
OPENVLA_DIR = Path("/home/elias/Thesis/openvla_outputs")

# -----------------------------
# Load all teacher latents
# -----------------------------
print(f"Loading latents from {LATENT_DIR} ...")
all_latents = []

for f in sorted(LATENT_DIR.glob("*.npy")):
    data = np.load(f, allow_pickle=True).item()
    teacher_latent = np.array(data["teacher_latent"]).flatten()  # shape (4,)
    all_latents.append(teacher_latent)

latents_raw = np.vstack(all_latents)  # shape: (N, 4)
print(f"Loaded {len(latents_raw)} samples with latent dim {latents_raw.shape[1]}")

# -----------------------------
# Load all OpenVLA student latents (7D actions)
# -----------------------------
print(f"\nLoading OpenVLA latents from {OPENVLA_DIR} ...")
openvla_latents = []

for f in sorted(OPENVLA_DIR.glob("*.npy")):
    data = np.load(f, allow_pickle=True).item()
    student_latent = np.array(data["student_action"]).flatten()  # shape (7,)
    openvla_latents.append(student_latent)

openvla_raw = np.vstack(openvla_latents)  # shape: (N, 7)
print(f"Loaded {len(openvla_raw)} OpenVLA samples with latent dim {openvla_raw.shape[1]}")

# -----------------------------
# Scaled version (-1, 1)
# -----------------------------
latents_scaled = (latents_raw / 7.0) * 2.0 - 1.0

# -----------------------------
# LayerNorm (frozen / non-learnable) for teacher latents
# -----------------------------
latents_torch = torch.tensor(latents_scaled, dtype=torch.float32)
latents_ln = torch.nn.functional.layer_norm(
    latents_torch, normalized_shape=(latents_torch.shape[-1],),
    weight=None, bias=None, eps=1e-6
).numpy()

# -----------------------------
# LayerNorm for OpenVLA latents
# -----------------------------
openvla_torch = torch.tensor(openvla_raw, dtype=torch.float32)
openvla_ln = torch.nn.functional.layer_norm(
    openvla_torch, normalized_shape=(openvla_torch.shape[-1],),
    weight=None, bias=None, eps=1e-6
).numpy()

# -----------------------------
# Pad teacher latents to 7D (match OpenVLA dimension)
# -----------------------------
latents_raw_padded = np.pad(latents_raw, ((0, 0), (0, 3)), mode='constant')
latents_scaled_padded = np.pad(latents_scaled, ((0, 0), (0, 3)), mode='constant')
latents_ln_padded = np.pad(latents_ln, ((0, 0), (0, 3)), mode='constant')

# Combine all data for joint PCA fit
combined_raw = np.vstack([latents_raw_padded, openvla_raw])
combined_ln = np.vstack([latents_ln_padded, openvla_ln])

# Fit PCA on combined raw data
pca = PCA(n_components=2)
pca.fit(combined_raw)

print(f"Combined explained variance ratio: {pca.explained_variance_ratio_}")

# Transform all datasets
latent_raw_2d = pca.transform(latents_raw_padded)
latent_scaled_2d = pca.transform(latents_scaled_padded)
latent_ln_2d = pca.transform(latents_ln_padded)
openvla_raw_2d = pca.transform(openvla_raw)
openvla_ln_2d = pca.transform(openvla_ln)

# Extract teacher and OpenVLA portions
n_teacher = len(latents_raw)
latent_raw_2d = latent_raw_2d[:n_teacher]
latent_scaled_2d = latent_scaled_2d[:n_teacher]
latent_ln_2d = latent_ln_2d[:n_teacher]

# -----------------------------
# Plot all in same space
# -----------------------------
plt.figure(figsize=(12, 8))

# plt.scatter(latent_raw_2d[:, 0], latent_raw_2d[:, 1],
#             s=10, alpha=0.2, label="Teacher Raw ([0,7])", color="tab:blue")

# plt.scatter(latent_scaled_2d[:, 0], latent_scaled_2d[:, 1],
#             s=10, alpha=0.2, label="Teacher Scaled ([-1,1])", color="tab:orange")

# plt.scatter(openvla_raw_2d[:, 0], openvla_raw_2d[:, 1],
#             s=10, alpha=0.2, label="OpenVLA Raw", color="tab:red")

plt.scatter(openvla_ln_2d[:, 0], openvla_ln_2d[:, 1],
            s=10, alpha=0.2, label="OpenVLA After Frozen LayerNorm", color="tab:purple")

plt.scatter(latent_ln_2d[:, 0], latent_ln_2d[:, 1],
            s=10, alpha=0.3, label="Teacher After Frozen LayerNorm", color="tab:green")

plt.title("Latent Space Comparison (PCA 2D, Joint Fit)\nTeacher vs OpenVLA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
