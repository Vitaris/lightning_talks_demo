#!/bin/bash
# Full environment setup for YOLO Can Detection Training
# Usage: bash setup_yolo.sh

set -e  # Exit on any error

echo "🚀 YOLO Can Detection - Full Environment Setup"
echo "================================================"

# 1. Check Python version
PYTHON_BIN=""
if command -v python3.12 &> /dev/null; then
    PYTHON_BIN="python3.12"
elif command -v python3.11 &> /dev/null; then
    PYTHON_BIN="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_BIN="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    echo "❌ Python 3 not found. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_BIN --version)
echo "🐍 Using: $PYTHON_VERSION"

# 2. Create virtual environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment '$VENV_DIR' already exists."
    read -p "   Recreate it? (y/n): " choice
    if [ "$choice" = "y" ]; then
        rm -rf "$VENV_DIR"
        echo "   Removed old environment."
    else
        echo "   Keeping existing environment."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_BIN -m venv "$VENV_DIR"
    echo "   ✅ Virtual environment created: $VENV_DIR"
fi

# 3. Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated"

# 4. Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 5. Install PyTorch with CUDA support
echo ""
echo "🔥 Installing PyTorch..."
echo "   Detecting GPU..."

if command -v nvidia-smi &> /dev/null; then
    echo "   ✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true

    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' 2>/dev/null || echo "")

    if [[ "$CUDA_VERSION" == 12.* ]]; then
        echo "   Installing PyTorch for CUDA 12.x..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    elif [[ "$CUDA_VERSION" == 11.* ]]; then
        echo "   Installing PyTorch for CUDA 11.x..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "   ⚠️  Unknown CUDA version ($CUDA_VERSION), installing default PyTorch with CUDA 12.4..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    fi
else
    echo "   ⚠️  No NVIDIA GPU detected, installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 6. Install Ultralytics (YOLO)
echo ""
echo "📦 Installing Ultralytics (YOLO)..."
pip install ultralytics

# 7. Install additional dependencies
echo ""
echo "📦 Installing additional dependencies..."
pip install opencv-python-headless
pip install matplotlib
pip install tensorboard

# 8. Create project directory structure
echo ""
echo "📁 Creating project directories..."
mkdir -p cans
mkdir -p dataset/images/train
mkdir -p dataset/images/val
mkdir -p dataset/labels/train
mkdir -p dataset/labels/val
echo "   ✅ Directories created"

# 9. Create cans.yaml if it doesn't exist
if [ ! -f "cans.yaml" ]; then
    echo ""
    echo "📝 Creating cans.yaml dataset config..."
    cat > cans.yaml << 'EOF'
path: ./dataset
train: images/train
val: images/val

nc: 3  # Adjust based on your classes
names: ['0_5L', '0_33L', '0_25L']  # Update with your actual class names
EOF
    echo "   ✅ cans.yaml created"
else
    echo "   ℹ️  cans.yaml already exists, skipping"
fi

# 10. Download base YOLO model if not present
echo ""
echo "📥 Downloading YOLO base model..."
if [ ! -f "yolo11n.pt" ]; then
    python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
    echo "   ✅ yolo11n.pt downloaded"
else
    echo "   ℹ️  yolo11n.pt already exists, skipping"
fi

# 11. Verify installation
echo ""
echo "🧪 Verifying installation..."
python << 'PYEOF'
import sys
print(f"  Python:       {sys.version.split()[0]}")

import torch
print(f"  PyTorch:      {torch.__version__}")
print(f"  CUDA:         {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

import ultralytics
print(f"  Ultralytics:  {ultralytics.__version__}")

import cv2
print(f"  OpenCV:       {cv2.__version__}")

print("\n  ✅ All core packages verified!")
PYEOF

# 12. Print usage instructions
echo ""
echo "================================================"
echo "✅ Setup complete!"
echo "================================================"
echo ""
echo "📋 Quick Start Guide:"
echo ""
echo "  1. Activate the environment:"
echo "     source $VENV_DIR/bin/activate"
echo ""
echo "  2. Put your images in the 'cans/' folder"
echo ""
echo "  3. Prepare the dataset:"
echo "     python prepare_dataset.py"
echo ""
echo "  4. Train the model:"
echo "     python train.py"
echo ""
echo "  5. Monitor training (in another terminal):"
echo "     source $VENV_DIR/bin/activate"
echo "     tensorboard --logdir runs/detect"
echo ""
echo "  6. Test the model:"
echo "     python test_model.py"
echo ""