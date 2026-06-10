"""
Qwen3-TTS with Megakernel Acceleration

This package provides a high-performance TTS engine using:
- Qwen3-TTS (0.6B) for Text-to-Speech
- Megakernel-accelerated decode (~1000 tok/s on RTX 5090)
- Streaming audio output
"""

__version__ = "0.2.0"

# Lazy imports to avoid requiring torch/qwen-tts for basic usage
def __getattr__(name):
    if name == "MegakernelTTSEngine":
        from .megakernel_tts import MegakernelTTSEngine
        return MegakernelTTSEngine
    elif name == "create_tts_engine":
        from .megakernel_tts import create_tts_engine
        return create_tts_engine
    elif name == "Qwen3TTSEngine":
        # Alias for backward compatibility
        from .megakernel_tts import MegakernelTTSEngine
        return MegakernelTTSEngine
    elif name == "Qwen3TTSService":
        from .pipecat_service import Qwen3TTSService
        return Qwen3TTSService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MegakernelTTSEngine", "create_tts_engine", "Qwen3TTSEngine", "Qwen3TTSService"]
