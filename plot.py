import re
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# LOAD RAW LOG FILE
# ===============================

with open("distill_openvlap-distill-b24-lr0.0001-lora-r32.log", "r") as f:
    raw = f.read()

# ===============================
# HELPERS
# ===============================

def extract_matrix(name, raw_text):
    """Extract a numpy matrix from the log by name."""
    pattern = name + r".*?\[\[(.*?)\]\]"
    match = re.search(pattern, raw_text, re.S)
    if not match:
        raise ValueError(f"Could not find matrix '{name}'")

    text = match.group(1)

    # replace scientific notation like 1.23e-01 with float
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    # split rows
    rows = text.split("] [")
    parsed = []

    for row in rows:
        # clean brackets
        row = row.replace("[", "").replace("]", "")
        if len(row.strip()) == 0:
            continue
        nums = [float(x) for x in row.strip().split()]
        parsed.append(nums)

    return np.array(parsed, dtype=float)

# ===============================
# EXTRACT MATRICES
# ===============================

student_latents = extract_matrix("Student Latent Projected", raw)
teacher_latents = extract_matrix("Teacher Hidden", raw)
student_sim = extract_matrix("Student Similarity Matrix", raw)
teacher_sim = extract_matrix("Teacher Similarity Matrix", raw)
abs_diff = extract_matrix("Absolute Difference", raw)

# -------------------------------
# FIX TEACHER SHAPE AUTOMATICALLY
# -------------------------------
# teacher_latents may be either:
#  - (24, 1, 4)
#  - (24, 4)
#  - (24, 1, 1, 4) depending on log formatting

tl = teacher_latents

# flatten repeated dims
while tl.ndim > 2:
    # if middle dimension is 1, squeeze it
    if 1 in tl.shape:
        tl = tl.squeeze()
    else:
        break

# ensure final shape is (24, 4)
if tl.ndim == 1:
    tl = tl.reshape(-1, 4)

teacher_latents = tl
print("Fixed teacher latent shape:", teacher_latents.shape)


# ===============================
# COMPUTE MSE PER SAMPLE
# ===============================

mse = ((student_latents - teacher_latents) ** 2).mean(axis=1)

# ===============================
# PLOTTING
# ===============================

plt.figure(figsize=(22, 18))

# --- 1. Student similarity ---
plt.subplot(2, 3, 1)
plt.title("Student Similarity Matrix")
plt.imshow(student_sim, cmap="viridis")
plt.colorbar()

# --- 2. Teacher similarity ---
plt.subplot(2, 3, 2)
plt.title("Teacher Similarity Matrix")
plt.imshow(teacher_sim, cmap="viridis")
plt.colorbar()

# --- 3. Abs diff ---
plt.subplot(2, 3, 3)
plt.title("Absolute Difference |S - T|")
plt.imshow(abs_diff, cmap="inferno")
plt.colorbar()

# --- 4. Student vs Teacher latent comparison ---
plt.subplot(2, 3, 4)
plt.title("Student vs Teacher Latents")
plt.plot(student_latents.flatten(), label="Student", alpha=0.8)
plt.plot(teacher_latents.flatten(), label="Teacher", alpha=0.8)
plt.legend()

# --- 5. Per-sample MSE ---
plt.subplot(2, 3, 5)
plt.title("MSE per Sample")
plt.bar(np.arange(len(mse)), mse)
plt.xlabel("Sample index")
plt.ylabel("MSE")

# --- 6. Difference norm histogram ---
dif_norms = np.linalg.norm(student_latents - teacher_latents, axis=1)
plt.subplot(2, 3, 6)
plt.title("Norm of Differences")
plt.hist(dif_norms, bins=20)
plt.xlabel("||S - T||")

plt.tight_layout()
plt.show()
