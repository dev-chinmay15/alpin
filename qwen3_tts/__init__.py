"""
Qwen3-TTS with Megakernel Acceleration

This package provides a high-performance TTS engine using:
- Megakernel-accelerated Talker model (~1000 tok/s on RTX 5090)
- Code Predictor for residual codebooks
- Streaming Speech Decoder
"""

__version__ = "0.1.0"

# Lazy imports to avoid requiring torch/pipecat for basic usage
def __getattr__(name):
    if name == "Qwen3TTSEngine":
        from .engine import Qwen3TTSEngine
        return Qwen3TTSEngine
    elif name == "Qwen3TTSService":
        from .pipecat_service import Qwen3TTSService
        return Qwen3TTSService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Qwen3TTSEngine", "Qwen3TTSService"]
