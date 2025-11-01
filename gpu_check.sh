#!/bin/bash

# GPU Compatibility Check for All VLA Environments
echo "Checking GPU compatibility across all environments..."

# GPU Check Python script
GPU_CHECK_SCRIPT='
import torch
import sys

def print_section(title):
    print("\n" + "="*50)
    print(title)
    print("="*50)

def print_status(message, status):
    symbols = {"success": "✓", "warning": "⚠", "error": "✗", "info": "ℹ"}
    symbol = symbols.get(status, "•")
    print(f"{symbol} {message}")

print_section("SYSTEM COMPATIBILITY")
print_status(f"Python: {sys.version.split()[0]}", "info")

# PyTorch check
try:
    print_status(f"PyTorch: {torch.__version__}", "success")
    if torch.cuda.is_available():
        print_status("CUDA: Available", "success")
        print_status(f"CUDA Version: {torch.version.cuda}", "info")
        gpu_count = torch.cuda.device_count()
        print_status(f"GPU Count: {gpu_count}", "info")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print_status(f"GPU {i}: {props.name} ({memory_gb:.1f} GB)", "info")
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            print_status("GPU Operations: Working", "success")
            torch.cuda.empty_cache()
        except Exception as e:
            print_status(f"GPU Operations: Failed - {e}", "error")
    else:
        print_status("CUDA: Not Available", "error")
except ImportError:
    print_status("PyTorch: Not installed", "error")

# Check model-specific packages
model_checks = []
try:
    import lerobot
    model_checks.append(("LeRobot/SmolVLA", "success"))
except ImportError:
    model_checks.append(("LeRobot/SmolVLA", "error"))

try:
    from transformers import AutoModelForVision2Seq
    model_checks.append(("Transformers (OpenVLA)", "success"))
except ImportError:
    model_checks.append(("Transformers (OpenVLA)", "error"))

try:
    import stable_baselines3
    model_checks.append(("Stable-Baselines3", "success"))
except ImportError:
    model_checks.append(("Stable-Baselines3", "error"))

try:
    import pybullet
    model_checks.append(("PyBullet", "success"))
except ImportError:
    model_checks.append(("PyBullet", "error"))

if model_checks:
    print_section("MODEL FRAMEWORKS")
    for name, status in model_checks:
        availability = "Available" if status == "success" else "Not Available"
        print_status(f"{name}: {availability}", status)
'


# =================
# CHECK SMOLVLA ENV
# =================
echo ""
echo "========================================="
echo "CHECKING SMOLVLA ENVIRONMENT"
echo "========================================="

eval "$(conda shell.bash hook)"
if conda activate smolvla-env 2>/dev/null; then
    python -c "$GPU_CHECK_SCRIPT"
    conda deactivate
else
    echo "❌ SmolVLA environment not found"
fi

# =================
# CHECK OPENVLA ENV
# =================
echo ""
echo "========================================="
echo "CHECKING OPENVLA ENVIRONMENT" 
echo "========================================="

if conda activate openvla-env 2>/dev/null; then
    python -c "$GPU_CHECK_SCRIPT"
    conda deactivate
else
    echo "❌ OpenVLA environment not found"
fi

# =================
# CHECK RL BASELINE ENV
# =================
echo ""
echo "========================================="
echo "CHECKING RL BASELINE ENVIRONMENT"
echo "========================================="

if conda activate rl-baseline-env 2>/dev/null; then
    python -c "$GPU_CHECK_SCRIPT"
    conda deactivate
else
    echo "❌ RL Baseline environment not found"
fi

echo ""
echo "========================================="
echo "SUMMARY"
echo "========================================="
echo "GPU check complete for all environments."
echo ""
echo "If any environments show CUDA issues, you may need to reinstall PyTorch with CUDA:"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
