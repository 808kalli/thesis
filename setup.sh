#!/bin/bash

# VLA Framework - Separate Environment Setup
echo "Setting up separate environments for VLA models..."

# First, remove the old environment if it exists
echo "Cleaning up old environment..."
conda deactivate 2>/dev/null || true
conda env remove -n vla-framework -y 2>/dev/null || true

# =================
# SMOLVLA ENVIRONMENT
# =================
echo ""
echo "========================================="
echo "SETTING UP SMOLVLA ENVIRONMENT"
echo "========================================="

# Create minimal environment for SmolVLA
conda create -n smolvla-env python=3.10 -y

# Activate SmolVLA environment
eval "$(conda shell.bash hook)"
conda activate smolvla-env

# Clone and install LeRobot/SmolVLA
echo "Installing SmolVLA (LeRobot)..."
cd ~/LAMDA/models
if [ -d "lerobot" ]; then
    rm -rf lerobot
fi
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Let LeRobot install all its dependencies
pip install -e ".[smolvla]"

echo "SmolVLA environment setup complete."
echo "Testing SmolVLA installation..."
python -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
except:
    print('PyTorch not available')

try:
    import lerobot
    print(f'LeRobot: {lerobot.__version__}')
except Exception as e:
    print(f'LeRobot error: {e}')
"

# Test PyBullet availability
echo "Checking PyBullet in SmolVLA environment..."
python -c "
try:
    import pybullet as p
    print('PyBullet: Available')
except ImportError:
    print('PyBullet: Not installed - will install')
    import subprocess
    subprocess.run(['pip', 'install', 'pybullet'])
    import pybullet as p
    print('PyBullet: Installed and working')
"

conda deactivate

# =================
# OPENVLA ENVIRONMENT  
# =================
echo ""
echo "========================================="
echo "SETTING UP OPENVLA ENVIRONMENT"
echo "========================================="

# Create minimal environment for OpenVLA
conda create -n openvla-env python=3.10 -y

# Activate OpenVLA environment  
conda activate openvla-env

# Clone and install OpenVLA
echo "Installing OpenVLA..."
cd ~/LAMDA/models
if [ -d "openvla" ]; then
    rm -rf openvla
fi
git clone https://github.com/openvla/openvla.git
cd openvla

# Let OpenVLA install all its dependencies
pip install -e .

echo "OpenVLA environment setup complete."
echo "Testing OpenVLA installation..."
python -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
except:
    print('PyTorch not available')

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    print('OpenVLA: Transformers ready')
except Exception as e:
    print(f'OpenVLA error: {e}')

try:
    import tensorflow as tf
    print(f'TensorFlow: {tf.__version__}')
except Exception as e:
    print(f'TensorFlow error: {e}')
"

# Test PyBullet availability
echo "Checking PyBullet in OpenVLA environment..."
python -c "
try:
    import pybullet as p
    print('PyBullet: Available')
except ImportError:
    print('PyBullet: Not installed - will install')
    import subprocess
    subprocess.run(['pip', 'install', 'pybullet'])
    import pybullet as p
    print('PyBullet: Installed and working')
"

conda deactivate

# =================
# RL BASELINE ENVIRONMENT
# =================
echo ""
echo "========================================="
echo "SETTING UP RL BASELINE ENVIRONMENT"
echo "========================================="

# Create environment for RL baselines
conda create -n rl-baseline-env python=3.10 -y
conda activate rl-baseline-env

# Install RL frameworks
echo "Installing RL baseline frameworks..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3[extra]
pip install gymnasium[all]
pip install pybullet
pip install tensorboard
pip install opencv-python
pip install matplotlib seaborn
pip install numpy pandas scipy

echo "RL baseline environment setup complete."
echo "Testing RL environment..."
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
except Exception as e:
    print(f'PyTorch error: {e}')

try:
    import stable_baselines3 as sb3
    print(f'Stable-Baselines3: {sb3.__version__}')
except Exception as e:
    print(f'SB3 error: {e}')

try:
    import gymnasium as gym
    print('Gymnasium: Available')
except Exception as e:
    print(f'Gymnasium error: {e}')

try:
    import pybullet as p
    print('PyBullet: Available')
except Exception as e:
    print(f'PyBullet error: {e}')
"

conda deactivate

echo ""
echo "========================================="
echo "SETUP COMPLETE"
echo "========================================="
echo ""
echo "Three environments created:"
echo "1. smolvla-env    - SmolVLA and LeRobot"
echo "2. openvla-env    - OpenVLA and dependencies" 
echo "3. rl-baseline-env - RL algorithms for comparison"
echo ""
echo "To activate each environment:"
echo "  conda activate smolvla-env"
echo "  conda activate openvla-env"  
echo "  conda activate rl-baseline-env"
echo ""
echo "Next: Run GPU check in each environment to verify CUDA compatibility"