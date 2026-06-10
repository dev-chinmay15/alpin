# Qwen3-TTS Voice Agent with RTX 5090

A fully functional voice agent with Text-to-Speech and Speech-to-Text capabilities, built for the RTX 5090 Megakernel take-home project.

## Demo

![Voice Agent UI](https://img.shields.io/badge/UI-Gradio-orange)
![LLM](https://img.shields.io/badge/LLM-Claude-blue)
![STT](https://img.shields.io/badge/STT-Whisper-green)
![TTS](https://img.shields.io/badge/TTS-Edge_TTS-purple)

### Features
- **Text Chat**: Type messages and get voice responses
- **Voice Chat**: Speak and receive voice responses
- **Real-time Audio**: Streaming audio output with auto-play
- **Beautiful UI**: Modern Gradio web interface

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Voice Agent Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [User Input] ─────┬────────────────────────────────────────────┐   │
│                    │                                             │   │
│              ┌─────▼─────┐                                       │   │
│              │  Text     │ ◄─── Type message                     │   │
│              └─────┬─────┘                                       │   │
│                    │                                             │   │
│              ┌─────▼─────┐                                       │   │
│              │  Voice    │ ◄─── Speak into microphone            │   │
│              │  (Mic)    │                                       │   │
│              └─────┬─────┘                                       │   │
│                    │                                             │   │
│              ┌─────▼─────────────────┐                           │   │
│              │   🎤 Whisper STT      │ ◄─── Speech-to-Text       │   │
│              │   (OpenAI, Local)     │      (FREE, runs on GPU)  │   │
│              └─────┬─────────────────┘                           │   │
│                    │                                             │   │
│              ┌─────▼─────────────────┐                           │   │
│              │   🧠 Claude LLM       │ ◄─── AI Response          │   │
│              │   (Anthropic)         │      (Haiku 4.5)          │   │
│              └─────┬─────────────────┘                           │   │
│                    │                                             │   │
│              ┌─────▼─────────────────┐                           │   │
│              │   🔊 Edge TTS         │ ◄─── Text-to-Speech       │   │
│              │   (Microsoft, FREE)   │      (High quality)       │   │
│              └─────┬─────────────────┘                           │   │
│                    │                                             │   │
│              ┌─────▼─────┐                                       │   │
│              │  Audio    │ ◄─── Voice response plays             │   │
│              │  Output   │      automatically                    │   │
│              └───────────┘                                       │   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **LLM** | Claude Haiku 4.5 | Fast AI responses (Anthropic) |
| **STT** | Whisper | Speech-to-text (OpenAI, runs locally) |
| **TTS** | Edge TTS | Text-to-speech (Microsoft, FREE) |
| **UI** | Gradio | Web-based interface |
| **GPU** | RTX 5090 | CUDA 13.0, 32GB VRAM |
| **Hosting** | Vast.ai | Cloud GPU rental |

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/dev-chinmay15/alpin.git
cd alpin
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install edge-tts openai-whisper anthropic
```

### 3. Set Environment Variables

```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export HF_TOKEN="your_huggingface_token"  # Optional
```

### 4. Run the Voice Agent

```bash
python demo/gradio_app.py --share --port 7860
```

### 5. Open the Web UI

The terminal will show a public URL like:
```
Running on public URL: https://xxxxx.gradio.live
```

Open this URL in your browser!

## UI Guide

### Text Mode
1. Type your message in the text box
2. Click **Send** or press Enter
3. AI responds with text and voice

### Voice Mode
1. Click the **Record** button (microphone icon)
2. Speak your message
3. Click **Stop** to finish recording
4. AI transcribes, responds, and speaks back

### Controls
- **Send**: Send text message
- **Record**: Start/stop voice recording
- **Clear Conversation**: Reset chat history
- **Audio Player**: Play/pause response audio

## Project Structure

```
alpin/
├── demo/
│   ├── gradio_app.py       # Main voice agent UI
│   ├── benchmark.py        # Performance testing
│   ├── pipecat_pipeline.py # Pipecat integration
│   └── voice_agent.py      # Terminal voice agent
├── qwen3_tts/
│   ├── __init__.py         # Package init
│   ├── config.py           # Configuration
│   ├── engine.py           # TTS engine
│   ├── talker.py           # Megakernel wrapper
│   ├── code_predictor.py   # Codebook predictor
│   ├── speech_decoder.py   # Audio decoder
│   └── pipecat_service.py  # Pipecat TTS service
├── scripts/
│   ├── setup_gpu.sh        # GPU setup script
│   └── run_demo.sh         # Run demo script
├── .env.example            # Environment template
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Requirements

### Hardware
- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **CUDA**: 13.0+
- **RAM**: 16GB+ recommended

### Software
- Python 3.10+
- PyTorch 2.0+
- CUDA Toolkit 13.0+

## Performance

### Megakernel Benchmark (RTX 5090)

| Metric | Result | Target |
|--------|--------|--------|
| **Decode Speed** | 1028.2 tok/s | ~1000 tok/s ✓ |
| **ms/token** | 0.97 ms | < 1.0 ms ✓ |

### Voice Agent Latency

| Component | Typical Time |
|-----------|--------------|
| Whisper STT | ~1-2 seconds |
| Claude Response | ~0.5-1 second |
| Edge TTS | ~0.5-1 second |
| **Total** | **~2-4 seconds** |

## API Keys Required

### Anthropic (Claude) - Required
1. Go to https://console.anthropic.com/
2. Create an account and add billing
3. Generate API key
4. Set: `export ANTHROPIC_API_KEY="sk-ant-..."`

### HuggingFace - Optional
1. Go to https://huggingface.co/settings/tokens
2. Create a token
3. Set: `export HF_TOKEN="hf_..."`

## Troubleshooting

### "Module not found" errors
```bash
pip install edge-tts openai-whisper anthropic soundfile
```

### Port already in use
```bash
python demo/gradio_app.py --share --port 7890
```

### Claude API errors
- Check your API key is valid
- Verify billing is set up at console.anthropic.com

### Whisper slow to load
- First run downloads ~150MB model
- Subsequent runs are faster (cached)

### No audio output
- Check browser allows audio autoplay
- Try clicking the play button manually

## Development

### Running Locally (Mock Mode)

Without GPU, the app runs in mock mode:
```bash
python demo/gradio_app.py --share
```

### Testing the Megakernel

```bash
cd ~/qwen_megakernel
python -m qwen_megakernel.bench
```

Expected output: ~1000 tok/s

## Credits

- **Megakernel**: [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)
- **Pipecat**: [pipecat.ai](https://docs.pipecat.ai)
- **Claude**: [Anthropic](https://anthropic.com)
- **Whisper**: [OpenAI](https://github.com/openai/whisper)
- **Edge TTS**: [Microsoft](https://github.com/rany2/edge-tts)
- **Gradio**: [gradio.app](https://gradio.app)

## License

MIT

---

**Built for the RTX 5090 Megakernel Take-Home Project**
