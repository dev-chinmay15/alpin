# Qwen3-TTS with RTX 5090 Megakernel + Pipecat

High-performance Text-to-Speech using [AlpinDale's megakernel](https://github.com/AlpinDale/qwen_megakernel) for Qwen3-TTS, integrated into a [Pipecat](https://docs.pipecat.ai) voice pipeline.

## Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| **TTFC** | < 60 ms | Time to first audio chunk |
| **RTF** | < 0.15 | Real-time factor (generation time / audio duration) |
| **Talker** | ~1000 tok/s | Megakernel decode speed on RTX 5090 |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Voice Pipeline                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [User Speech] → [STT] → [LLM (Gemini)] → [TTS] → [Audio Output]    │
│                                             │                        │
│                                             ▼                        │
│                    ┌─────────────────────────────────────────┐      │
│                    │       Qwen3-TTS Engine                   │      │
│                    ├─────────────────────────────────────────┤      │
│                    │                                         │      │
│                    │  ┌─────────────────────────────────┐   │      │
│                    │  │ Talker (Megakernel) 🚀          │   │      │
│                    │  │ • 28 transformer layers         │   │      │
│                    │  │ • ~1000 tok/s on RTX 5090       │   │      │
│                    │  │ • Outputs: Codebook-0 tokens    │   │      │
│                    │  └─────────────────────────────────┘   │      │
│                    │              │                          │      │
│                    │              ▼                          │      │
│                    │  ┌─────────────────────────────────┐   │      │
│                    │  │ Code Predictor (PyTorch)        │   │      │
│                    │  │ • 5 layers × 15 passes          │   │      │
│                    │  │ • Outputs: Codebooks 1-15       │   │      │
│                    │  └─────────────────────────────────┘   │      │
│                    │              │                          │      │
│                    │              ▼                          │      │
│                    │  ┌─────────────────────────────────┐   │      │
│                    │  │ Speech Decoder (ConvNet)        │   │      │
│                    │  │ • Causal/streaming              │   │      │
│                    │  │ • Outputs: 24kHz audio          │   │      │
│                    │  └─────────────────────────────────┘   │      │
│                    │              │                          │      │
│                    │              ▼                          │      │
│                    │      Streaming Audio Chunks             │      │
│                    └─────────────────────────────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Requirements

- **GPU**: NVIDIA RTX 5090 (sm_120 / Blackwell)
- **CUDA**: 12.8+
- **Python**: 3.10+
- **OS**: Linux (Ubuntu 22.04+ recommended)

## Setup

### 1. Clone Repository

```bash
git clone <this-repo>
cd qwen3-tts-pipecat
```

### 2. Clone Megakernel

```bash
git clone https://github.com/AlpinDale/qwen_megakernel.git ../qwen_megakernel
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. Run Benchmark

```bash
python -m demo.benchmark
```

### 6. Run Voice Agent Demo

```bash
python -m demo.voice_agent
```

## Project Structure

```
qwen3-tts-pipecat/
├── qwen3_tts/
│   ├── __init__.py         # Package init
│   ├── config.py           # Configuration
│   ├── talker.py           # Megakernel-accelerated Talker
│   ├── code_predictor.py   # Code Predictor (codebooks 1-15)
│   ├── speech_decoder.py   # Audio decoder
│   ├── engine.py           # Main TTS engine
│   └── pipecat_service.py  # Pipecat integration
├── demo/
│   ├── voice_agent.py      # Voice agent demo
│   └── benchmark.py        # Performance benchmark
├── .env.example            # Environment template
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Kernel Modifications

The megakernel was originally designed for Qwen3-0.6B text generation. For Qwen3-TTS:

### Changes Made:

1. **Vocab Size**: The Talker outputs audio codebook tokens (vocab ~2048) instead of text tokens (vocab 151936). This significantly reduces LM head computation.

2. **Weight Loading**: Modified to load from `talker.model.layers.*` path structure in Qwen3-TTS instead of standard Qwen3 paths.

3. **Output Head**: Uses `codec_head` instead of `lm_head` for predicting audio tokens.

### Architecture Compatibility:

| Parameter | Qwen3-0.6B | Qwen3-TTS Talker |
|-----------|------------|------------------|
| Layers | 28 | 28 |
| Hidden Size | 1024 | 1024 |
| Q Heads | 16 | 16 |
| KV Heads | 8 | 8 |
| Head Dim | 128 | 128 |
| MLP | SwiGLU (3072) | SwiGLU (3072) |

The architectures are compatible, allowing direct use of the megakernel.

## Streaming Implementation

Audio is streamed frame-by-frame, not buffered:

```python
async for audio_chunk in engine.generate_streaming(text):
    # Each chunk is ~80ms of audio (1920 samples @ 24kHz)
    # Sent immediately to Pipecat output
    yield TTSAudioRawFrame(audio=audio_chunk, ...)
```

This ensures minimal latency - audio starts playing as soon as the first frame is ready.

## Performance Results

_(To be filled after GPU testing)_

| Metric | Measured | Target |
|--------|----------|--------|
| TTFC | TBD | < 60 ms |
| RTF | TBD | < 0.15 |
| Talker tok/s | TBD | ~1000 |
| E2E Latency | TBD | < 200 ms |

## Running the Demo

### Option 1: Gradio Web UI (Recommended)

```bash
# On GPU machine with RTX 5090
python -m demo.gradio_app --share

# Opens web UI with public URL
# Access from any browser!
```

### Option 2: Pipecat Voice Pipeline

```bash
# Full voice pipeline (requires mic)
python -m demo.pipecat_pipeline --mode voice

# Text-only mode (no mic needed)
python -m demo.pipecat_pipeline --mode text
```

### Option 3: Benchmark Only

```bash
python -m demo.benchmark
```

## Usage with Pipecat

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.google import GoogleLLMService
from pipecat.services.silero import SileroVADService
from qwen3_tts import Qwen3TTSService

# Create services
vad = SileroVADService()           # Voice detection (FREE)
llm = GoogleLLMService(...)        # LLM (Gemini)
tts = Qwen3TTSService(verbose=True) # TTS (Megakernel!)

# Build pipeline: STT → LLM → TTS
pipeline = Pipeline([
    transport.input(),
    vad,
    stt_service,                    # Whisper (FREE, local)
    llm,
    tts,                            # Our megakernel-powered TTS!
    transport.output(),
])
```

## Troubleshooting

### Kernel compilation fails
- Ensure CUDA 12.8+ is installed
- Verify RTX 5090 is detected: `nvidia-smi`
- Check `nvcc --version`

### Out of memory
- Reduce `MAX_SEQ_LEN` in `.env`
- Ensure no other GPU processes running

### Slow performance
- Verify running on RTX 5090 (not other GPU)
- Check power/thermal throttling
- Disable other GPU workloads

## Credits

- [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) - RTX 5090 megakernel
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - TTS model
- [Pipecat](https://docs.pipecat.ai) - Voice pipeline framework

## License

MIT
