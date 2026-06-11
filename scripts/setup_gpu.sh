#!/bin/bash
# GPU Setup Script for Qwen3-TTS Voice Agent
# Run this on your Vast.ai GPU instance

set -e

echo "========================================"
echo "Qwen3-TTS Voice Agent - GPU Setup"
echo "========================================"

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is CUDA installed?"
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Create conda environment
echo ""
echo "Creating conda environment..."
if ! command -v conda &> /dev/null; then
    echo "Installing miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    rm miniconda.sh
fi

conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# Install PyTorch
echo ""
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
echo ""
echo "Installing project dependencies..."
pip install -r requirements.txt

# Install Flash Attention
echo ""
echo "Installing Flash Attention..."
pip install -U flash-attn --no-build-isolation

# Clone and build megakernel
echo ""
echo "Setting up megakernel..."
if [ ! -d "../qwen_megakernel" ]; then
    cd ..
    git clone https://github.com/AlpinDale/qwen_megakernel.git
    cd qwen_megakernel
    pip install -e .
    cd ../qwen3-tts-pipecat
else
    echo "Megakernel already exists"
fi

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_properties(0).name}')

try:
    from qwen_tts import Qwen3TTSModel
    print('Qwen-TTS: OK')
except:
    print('Qwen-TTS: Not installed')

try:
    import anthropic
    print('Anthropic: OK')
except:
    print('Anthropic: Not installed')
"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Set your API key: export ANTHROPIC_API_KEY='your_key'"
echo "2. Run the voice agent: python demo/gradio_app.py --share"
echo "3. Run decode server: uvicorn qwen3_tts.server:app --host 0.0.0.0 --port 8000"
echo "4. Run benchmark: python scripts/benchmark.py"
