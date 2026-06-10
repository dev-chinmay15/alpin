# Qwen3-TTS Voice Agent with RTX 5090 Megakernel

A high-performance voice agent using **Qwen3-TTS** with **megakernel acceleration** on RTX 5090, achieving ~1000 tok/s decode speed.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Voice Agent Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   🎤 Voice Input                    💬 Text Input               │
│        │                                  │                      │
│        ▼                                  │                      │
│   ┌─────────┐                             │                      │
│   │ Whisper │ (Local STT, FREE)           │                      │
│   │  (base) │                             │                      │
│   └────┬────┘                             │                      │
│        │                                  │                      │
│        └──────────────┬───────────────────┘                      │
│                       ▼                                          │
│              ┌─────────────────┐                                 │
│              │  Claude Haiku   │ (Conversational LLM)            │
│              │   (Anthropic)   │                                 │
│              └────────┬────────┘                                 │
│                       │                                          │
│                       ▼                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Qwen3-TTS-0.6B-CustomVoice                 │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │         Megakernel-Accelerated Decoder          │    │   │
│   │  │         (~1000 tok/s on RTX 5090)              │    │   │
│   │  └─────────────────────────────────────────────────┘    │   │
│   └────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│                    🔊 Audio Output                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

| Component | Implementation | Notes |
|-----------|---------------|-------|
| **STT** | Whisper (base) | Local, free, ~150ms latency |
| **LLM** | Claude Haiku | Fast responses, low cost |
| **TTS** | Qwen3-TTS-0.6B | Megakernel-accelerated |
| **UI** | Gradio | Web interface with voice/text input |

## Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| **TTFC** | < 60ms | Time to first audio chunk |
| **RTF** | < 0.15 | Real-time factor (gen_time / audio_duration) |
| **Decode** | ~1000 tok/s | Megakernel token generation speed |

## Quick Start

### 1. Setup Environment (on GPU)

```bash
# Create conda environment
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention (for better performance)
pip install -U flash-attn --no-build-isolation
```

### 2. Set API Keys

```bash
# Claude LLM API key
export ANTHROPIC_API_KEY="your_key_here"

# HuggingFace token (for model download)
export HF_TOKEN="your_token_here"
```

### 3. Run the Voice Agent

```bash
# Run with Gradio UI
python demo/gradio_app.py --share

# Or run benchmark
python scripts/benchmark.py
```

## Project Structure

```
qwen3-tts-pipecat/
├── demo/
│   ├── gradio_app.py       # Main Gradio web UI
│   └── pipecat_pipeline.py # Pipecat voice pipeline
├── qwen3_tts/
│   ├── __init__.py
│   └── megakernel_tts.py   # Megakernel TTS integration
├── scripts/
│   ├── benchmark.py        # Performance benchmarking
│   └── setup_gpu.sh        # GPU setup script
├── requirements.txt
├── .env.example
└── README.md
```

## Megakernel Integration

The megakernel from [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) provides optimized CUDA kernels for Qwen3-0.6B decode:

- **Architecture**: 128 persistent thread blocks × 512 threads
- **Performance**: ~1000 tok/s decode (0.97 ms/step)
- **Memory**: 71% of theoretical GDDR7 bandwidth utilization

The `Qwen3-TTS-12Hz-0.6B-CustomVoice` model uses the same Qwen3-0.6B backbone, making it compatible with the megakernel.

## Qwen3-TTS Model

Using the official [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) from Hugging Face:

- **Model**: Qwen3-TTS-12Hz-0.6B-CustomVoice
- **Size**: 0.6B parameters (BF16)
- **Features**: 
  - 10 languages support
  - Multiple voice speakers
  - Instruction-based voice control
  - Streaming generation
  - 97ms end-to-end latency (official spec)

### Available Speakers

| Speaker | Voice Type | Language |
|---------|-----------|----------|
| Ryan | Dynamic male | English |
| Aiden | Sunny American male | English |
| Vivian | Bright young female | Chinese |
| Serena | Warm gentle female | Chinese |

## GPU Setup (Vast.ai)

1. Rent an RTX 5090 instance on [Vast.ai](https://vast.ai)
2. Select Ubuntu template with CUDA support
3. Clone this repository and the megakernel:

```bash
# Clone project
git clone https://github.com/your-repo/qwen3-tts-pipecat.git
cd qwen3-tts-pipecat

# Clone megakernel
git clone https://github.com/AlpinDale/qwen_megakernel.git

# Build megakernel
cd qwen_megakernel
pip install -e .
```

## Benchmarking

Run the benchmark suite to measure performance:

```bash
# Full benchmark
python scripts/benchmark.py

# Megakernel only
python scripts/benchmark.py --megakernel-only

# TTS only
python scripts/benchmark.py --tts-only --iterations 10
```

### Expected Results (RTX 5090)

| Test | TTFC | RTF | Notes |
|------|------|-----|-------|
| Short text | ~40ms | ~0.08 | "Hello, how are you?" |
| Medium text | ~80ms | ~0.10 | 100 characters |
| Long text | ~150ms | ~0.12 | 250+ characters |
| Megakernel | - | - | ~1000 tok/s |

## Development

### Running Tests

```bash
# Test TTS engine
python -c "from qwen3_tts import MegakernelTTSEngine; e = MegakernelTTSEngine(); e.initialize()"

# Test Gradio app locally (mock mode)
python demo/gradio_app.py
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `HF_TOKEN` | No | HuggingFace token for model download |
| `CUDA_VISIBLE_DEVICES` | No | GPU device selection |

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use 0.6B model instead of 1.7B
2. **Flash Attention error**: Rebuild with `pip install flash-attn --no-build-isolation`
3. **Model download fails**: Set `HF_TOKEN` environment variable

### Fallback Mode

If Qwen3-TTS is not available, the system falls back to:
1. **Edge TTS** (Microsoft, free, cloud-based)
2. **Mock audio** (sine wave for testing)

## References

- [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621)
- [AlpinDale's Megakernel Blog](https://blog.alpindale.net/posts/5090_decode_optimization/)
- [Megakernel Source](https://github.com/AlpinDale/qwen_megakernel)
- [Pipecat Documentation](https://docs.pipecat.ai)

## License

Apache 2.0 (same as Qwen3-TTS)
