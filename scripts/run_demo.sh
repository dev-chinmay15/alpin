#!/bin/bash
# Run the voice agent demo

cd "$(dirname "$0")/.."

echo "Starting Qwen3-TTS Voice Agent Demo..."
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Copy .env.example to .env and add your API keys"
    exit 1
fi

# Run demo
python -m demo.voice_agent
