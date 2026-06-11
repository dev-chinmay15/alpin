#!/bin/bash
# Setup script for RTX 5090 GPU machine
# Run this after SSH-ing into your Vast.ai instance

set -e

echo "========================================"
echo "Qwen3-TTS Megakernel Setup"
echo "========================================"

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Check CUDA
echo "Checking CUDA..."
nvcc --version || echo "nvcc not found - will use PyTorch's CUDA"
echo ""

# Install system dependencies
echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq libsndfile1 portaudio19-dev ffmpeg

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Clone megakernel if not present
if [ ! -d "../qwen_megakernel" ]; then
    echo "Cloning megakernel repository..."
    git clone https://github.com/AlpinDale/qwen_megakernel.git ../qwen_megakernel
fi

# Test megakernel compilation
echo ""
echo "Testing megakernel compilation..."
cd ../qwen_megakernel
python -c "from qwen_megakernel.build import get_extension; print('Megakernel compiled successfully!')" || echo "Compilation failed - check CUDA setup"
cd -

# Run quick benchmark
echo ""
echo "Running quick benchmark..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_properties(0).name}')
    print(f'CUDA version: {torch.version.cuda}')
"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Run options:"
echo ""
echo "  1. Gradio Web UI (recommended):"
echo "     python -m demo.gradio_app --share"
echo ""
echo "  2. Pipecat Voice Pipeline:"
echo "     python -m demo.pipecat_pipeline --mode text"
echo ""
echo "  3. Benchmark:"
echo "     python -m demo.benchmark"
echo ""
