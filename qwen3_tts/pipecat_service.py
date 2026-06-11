"""
Pipecat TTS Service for Qwen3-TTS with Megakernel.

Integrates the Qwen3-TTS engine into Pipecat's voice pipeline.
"""

import asyncio
from typing import AsyncGenerator

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import TTSService

from .engine import Qwen3TTSEngine
from .config import config


class Qwen3TTSService(TTSService):
    """
    Pipecat TTS service using Qwen3-TTS with megakernel acceleration.
    
    Features:
    - Streaming audio output (frame-by-frame)
    - ~1000 tok/s on RTX 5090
    - Low latency TTFC (<60ms target)
    - RTF < 0.15 target
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "cuda",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            sample_rate=config.sample_rate,
            **kwargs,
        )
        
        self.model_name = model_name or config.model_name
        self.device = device
        self.verbose = verbose
        
        # Initialize engine lazily
        self._engine = None
    
    @property
    def engine(self) -> Qwen3TTSEngine:
        """Lazy initialization of TTS engine."""
        if self._engine is None:
            self._engine = Qwen3TTSEngine(
                model_name=self.model_name,
                device=self.device,
                verbose=self.verbose,
                buffer_frames=1,  # Minimal buffering for streaming
            )
        return self._engine
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Generate speech from text, yielding audio frames.
        
        This is the main method called by Pipecat pipeline.
        
        Args:
            text: Input text to synthesize
            
        Yields:
            TTSAudioRawFrame with audio data
        """
        # Signal TTS started
        yield TTSStartedFrame()
        
        try:
            # Stream audio chunks
            async for audio_bytes in self.engine.generate_streaming(text):
                yield TTSAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=config.sample_rate,
                    num_channels=config.channels,
                )
        except Exception as e:
            print(f"[Qwen3TTSService] Error: {e}")
        
        # Signal TTS stopped
        yield TTSStoppedFrame()
    
    async def _generate_audio(self, text: str) -> bytes:
        """Generate complete audio (non-streaming)."""
        return self.engine.generate_sync(text)


class Qwen3TTSServiceMock(TTSService):
    """
    Mock TTS service for testing without GPU.
    
    Generates silence/sine waves for testing pipeline integration.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            sample_rate=config.sample_rate,
            **kwargs,
        )
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate mock audio for testing."""
        import numpy as np
        
        yield TTSStartedFrame()
        
        # Generate a short beep as placeholder
        duration_s = len(text) * 0.05  # ~50ms per character
        samples = int(config.sample_rate * duration_s)
        
        # Generate sine wave
        t = np.linspace(0, duration_s, samples)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
        
        # Convert to bytes
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Yield in chunks
        chunk_size = config.samples_per_frame * 4
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size].tobytes()
            yield TTSAudioRawFrame(
                audio=chunk,
                sample_rate=config.sample_rate,
                num_channels=config.channels,
            )
            await asyncio.sleep(0.01)  # Small delay to simulate processing
        
        yield TTSStoppedFrame()
