"""
Pipecat TTS Service for Qwen3-TTS.

Provides a Pipecat-compatible TTS service using Qwen3-TTS with megakernel acceleration.
"""

import asyncio
from typing import AsyncGenerator

# Lazy imports
PIPECAT_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    from pipecat.services.tts import TTSService
    from pipecat.frames.frames import AudioRawFrame, Frame
    PIPECAT_AVAILABLE = True
except ImportError:
    TTSService = object
    AudioRawFrame = None
    Frame = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass


class Qwen3TTSService(TTSService if PIPECAT_AVAILABLE else object):
    """
    Pipecat TTS service using Qwen3-TTS with megakernel acceleration.
    
    This service integrates with Pipecat pipelines for voice agents.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        speaker: str = "Ryan",
        language: str = "English",
        sample_rate: int = 24000,
        verbose: bool = False,
    ):
        if PIPECAT_AVAILABLE:
            super().__init__()
        
        self.model_name = model_name
        self.speaker = speaker
        self.language = language
        self.sample_rate = sample_rate
        self.verbose = verbose

        self.engine = None
        self.tts_model = None  # Non-streaming fallback path
        self._initialized = False

        if verbose:
            print(f"[Qwen3TTSService] Initialized with model: {model_name}")

    def _initialize(self):
        """Lazy initialization of streaming engine (preferred) and fallback model."""
        if self._initialized:
            return

        if not TORCH_AVAILABLE:
            if self.verbose:
                print("[Qwen3TTSService] PyTorch not available")
            return

        if not torch.cuda.is_available():
            if self.verbose:
                print("[Qwen3TTSService] CUDA not available")
            return

        try:
            # Prefer native streaming pipeline (Talker -> CodePredictor -> Decoder).
            from .engine import Qwen3TTSEngine

            self.engine = Qwen3TTSEngine(
                model_name=self.model_name,
                verbose=self.verbose,
                buffer_frames=1,  # minimize latency for TTFC
            )
            self._initialized = True
            if self.verbose:
                print("[Qwen3TTSService] Streaming engine initialized")
        except Exception as e:
            if self.verbose:
                print(f"[Qwen3TTSService] Streaming engine unavailable: {e}")

        if self._initialized:
            return

        # Fallback path for environments where full streaming stack is unavailable.
        try:
            from qwen_tts import Qwen3TTSModel
            self.tts_model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            self._initialized = True
            if self.verbose:
                print("[Qwen3TTSService] Loaded qwen-tts fallback")
        except Exception as e:
            if self.verbose:
                print(f"[Qwen3TTSService] Failed to load fallback model: {e}")

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech from text.

        Yields audio chunks as bytes (int16 PCM).
        """
        self._initialize()

        if self.engine is not None:
            async for chunk in self.engine.generate_streaming(text):
                yield chunk
            return

        if self.tts_model is not None:
            try:
                import numpy as np

                wavs, sr = self.tts_model.generate_custom_voice(
                    text=text,
                    language=self.language,
                    speaker=self.speaker,
                )

                audio = wavs[0]
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()

                if audio.dtype != np.int16:
                    audio = (audio * 32767).astype(np.int16)

                chunk_size = self.sample_rate // 10  # 100ms chunking in fallback
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    yield chunk.tobytes()
                    await asyncio.sleep(0)
                return
            except Exception as e:
                if self.verbose:
                    print(f"[Qwen3TTSService] Fallback synth error: {e}")

        if self.verbose:
            print("[Qwen3TTSService] Using mock synth")
        async for chunk in self._mock_synthesize(text):
            yield chunk

    async def run_tts(self, text: str):
        """
        Compatibility helper used by demo scripts.
        Produces `AudioRawFrame` items when Pipecat is available.
        """
        if not PIPECAT_AVAILABLE:
            async for chunk in self._mock_synthesize(text):
                yield chunk
            return

        async for audio_chunk in self.synthesize(text):
            yield AudioRawFrame(
                audio=audio_chunk,
                sample_rate=self.sample_rate,
                num_channels=1,
            )

    async def _mock_synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate mock audio (sine wave)."""
        import numpy as np

        duration = min(len(text) * 0.05, 5.0)
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        audio = np.sin(2 * np.pi * 440 * t) * 0.3
        audio = (audio * 32767).astype(np.int16)

        chunk_size = self.sample_rate // 10
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            yield chunk.tobytes()
            await asyncio.sleep(0.01)

    async def process_frame(self, frame: Frame, direction):
        """Process Pipecat frames."""
        if not PIPECAT_AVAILABLE:
            return

        await super().process_frame(frame, direction)

        from pipecat.frames.frames import TextFrame

        if isinstance(frame, TextFrame):
            text = frame.text

            async for audio_chunk in self.synthesize(text):
                audio_frame = AudioRawFrame(
                    audio=audio_chunk,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                await self.push_frame(audio_frame, direction)
        else:
            await self.push_frame(frame, direction)


class Qwen3TTSServiceMock(TTSService if PIPECAT_AVAILABLE else object):
    """
    Mock TTS service for testing without GPU.
    
    Generates simple sine wave audio.
    """
    
    def __init__(self, sample_rate: int = 24000):
        if PIPECAT_AVAILABLE:
            super().__init__()
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate mock audio."""
        import numpy as np

        duration = min(len(text) * 0.05, 3.0)
        t = np.linspace(0, duration, int(self.sample_rate * duration))

        frequencies = [440, 494, 523]
        audio = np.zeros_like(t)

        chunk_len = len(t) // len(frequencies)
        for i, freq in enumerate(frequencies):
            start = i * chunk_len
            end = min(start + chunk_len, len(t))
            audio[start:end] = np.sin(2 * np.pi * freq * t[start:end]) * 0.3

        audio = (audio * 32767).astype(np.int16)

        chunk_size = self.sample_rate // 10
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            yield chunk.tobytes()
            await asyncio.sleep(0.05)

    async def run_tts(self, text: str):
        if not PIPECAT_AVAILABLE:
            async for chunk in self.synthesize(text):
                yield chunk
            return
        async for audio_chunk in self.synthesize(text):
            yield AudioRawFrame(
                audio=audio_chunk,
                sample_rate=self.sample_rate,
                num_channels=1,
            )

    async def process_frame(self, frame: Frame, direction):
        """Process Pipecat frames."""
        if not PIPECAT_AVAILABLE:
            return

        await super().process_frame(frame, direction)

        from pipecat.frames.frames import TextFrame

        if isinstance(frame, TextFrame):
            async for audio_chunk in self.synthesize(frame.text):
                audio_frame = AudioRawFrame(
                    audio=audio_chunk,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                await self.push_frame(audio_frame, direction)
        else:
            await self.push_frame(frame, direction)
