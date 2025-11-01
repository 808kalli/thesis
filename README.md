# Vision-Language-Action Models are Efficient Learners

---

## 📖 Overview

This repository contains the code and experimental setup for the project **"Vision-Language-Action (VLA) Models are Efficient Learners"**.

This project benchmarks **OpenVLA (7B)** and **SmolVLA (2B)** on the **LIBERO-Spatial** simulation benchmark using a 7-DoF Franka Emika Panda robot. I explore how efficiently these models adapt to new robotic environments, how model size affects generalization, and what energy–performance trade-offs emerge during fine-tuning.

The focus is on:

1. How efficiently can VLA models adapt to **new embodiments, camera views, and environments**?
2. How does **model scale** (VLM backbone capacity) affect zero-shot and transfer learning?
3. What are the **energy–performance trade-offs** between large and compact architectures?

---

## ⚙️ Experimental Setup

### Simulation Environment
- **Benchmark:** LIBERO-Spatial (10 spatial reasoning tasks)
- **Robot:** Franka Emika Panda, 7-DoF arm  
- **Dataset:** 434 expert demonstrations (RLDS format via HuggingFace)

### Models Evaluated
| Model | Parameters | Backbone | Vision Encoder |
|:------|:------------|:----------|:----------------|
| **OpenVLA** | 7B | LLaMA-2 | DINOv2 + SigLIP |
| **SmolVLA** | 2B | Phi-3 | SigLIP |

---

## 🧪 Experiments

| Regime | Description |
|:--------|:-------------|
| **Zero-Shot** | Direct evaluation without fine-tuning |
| **Few-Shot (10 / 100)** | Adaptation with limited expert demonstrations |
| **Full Fine-Tuning** | Full dataset training |
| **Scratch (SmolVLA only)** | Train from scratch keeping vision & language frozen |

Both models were fine-tuned in simulated LIBERO environments, with **LoRA adapters** for efficient training.


### 📈 Training Curves
<img width="707" height="586" alt="Screenshot from 2025-10-17 13-00-22" src="https://github.com/user-attachments/assets/d369db42-3cc3-4245-9174-4f30bca3b240" />

---

## 📊 Results Summary

| Model | Zero-Shot | 10 Ep | 100 Ep | Full | Scratch |
|:------|:-----------|:------|:--------|:------|:----------|
| **OpenVLA** | 0.0 | 1.0 | 31.0 | 71.0 | – |
| **SmolVLA** | 0.0 | 9.4 | 59.6 | 79.4 | 82.6 |

- **SmolVLA** shows superior *sample efficiency* and *energy efficiency*, making it ideal for real-time deployment.  
- **OpenVLA** demonstrates stronger *transfer learning* and *semantic generalization* due to its larger VLM backbone.  
- Camera rotation caused total performance collapse in OpenVLA → lack of rotation-invariant features.  
- Action chunking reduced accuracy, highlighting the difficulty of multi-step prediction.

---

### 🎯 Task Completion Visualization
<img width="710" height="340" alt="Screenshot from 2025-10-17 12-59-57" src="https://github.com/user-attachments/assets/8371d1ea-116e-483a-a44b-8a4601ca060c" />

---

### 🔍 Zero-Shot Generalization

To evaluate generalization to unseen environments, both pretrained models were tested directly on **LIBERO-Spatial** and **LIBERO-Object** without any fine-tuning.

- In **pure zero-shot**, both models failed to complete tasks successfully due to the real-to-sim gap and differing camera configurations.  
  Their motion patterns appeared random, indicating no spatial understanding of the new environment.

- After fine-tuning on **LIBERO-Spatial**, **OpenVLA** demonstrated *emergent semantic understanding*: it began moving toward target objects and sometimes grasped them correctly on unseen **LIBERO-Object** tasks where it achieved partial success without retraining.

<img width="712" height="366" alt="Screenshot from 2025-10-17 12-59-06" src="https://github.com/user-attachments/assets/4cdf4ae0-c635-44fe-8b3e-79355f9f7b41" />

- **SmolVLA**, while faster and more efficient, lacked this emergent transfer behavior.  
  Its smaller 2 B backbone limited its ability to generalize to unseen object manipulation tasks.

- With only 10 episodes of fine-tuning in the **LIBERO-Object** benchmark (1 episode per task) we see that OpenVLA learns the new task much more effectively, utilizing its *emergent semantic understanding* capabilities.

| Model | Fine-tuning | Success Rate (%) on LIBERO-Object |
|:------|:-------------|:---------------------------------:|
| **OpenVLA** | Zero-shot | 0.0 |
| | + 10 Episodes | 5.5 |
| **SmolVLA** | Zero-shot | 0.0 |
| | + 10 Episodes | 0.0 |

These findings highlight that **large-scale VLAs** like OpenVLA retain richer visual-semantic priors, while **smaller VLAs** like SmolVLA require additional examples to transfer knowledge across tasks and embodiments.

---

## ⚡ Energy–Performance Trade-Offs

SmolVLA achieves ~80% task success at **half the energy cost** and training time of OpenVLA.  
Training OpenVLA for full fine-tuning required ~14.5 h on an A100 GPU; SmolVLA only ~7.7 h on an RTX 3060 12 GB.

<img width="556" height="312" alt="Screenshot from 2025-10-17 13-01-26" src="https://github.com/user-attachments/assets/93fb79b5-59e0-4b45-b95f-4be356f4d7d1" />

---

## 🧩 Key Findings

- VLAs can **learn efficient robotic behaviors** with very few demonstrations.  
- Compact architectures like **SmolVLA** provide a viable balance between performance and deployability.  
- Larger models like **OpenVLA** show **better transfer and semantic reasoning** but require more compute.  
- Fine-tuning on spatial benchmarks improves emergent reasoning across unseen manipulation tasks.

---

