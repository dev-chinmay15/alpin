"""
Full Pipecat Voice Pipeline with Qwen3-TTS.

Pipeline: Mic → VAD → STT (Whisper) → LLM (Gemini) → TTS (Megakernel) → Speaker

This is the complete Pipecat integration as required by the task.

Usage:
  python -m demo.pipecat_pipeline
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Pipecat imports
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    LLMMessagesFrame,
    EndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.google import GoogleLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioParams

# VAD for voice activity detection (free, local)
from pipecat.services.silero import SileroVADService

# Our TTS service
from qwen3_tts.pipecat_service import Qwen3TTSService, Qwen3TTSServiceMock


class TranscriptionLogger(FrameProcessor):
    """Log transcriptions for debugging."""
    
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            print(f"\n[User]: {frame.text}")
        
        await self.push_frame(frame, direction)


class ResponseLogger(FrameProcessor):
    """Log LLM responses for debugging."""
    
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            print(f"[Assistant]: {frame.text}")
        
        await self.push_frame(frame, direction)


class SimpleSTT(FrameProcessor):
    """
    Simple STT processor using local Whisper.
    
    For production, use pipecat's built-in WhisperSTTService.
    This is a simplified version for demo purposes.
    """
    
    def __init__(self):
        super().__init__()
        self.whisper_model = None
        self._load_whisper()
    
    def _load_whisper(self):
        """Load Whisper model."""
        try:
            import whisper
            print("[STT] Loading Whisper model (base)...")
            self.whisper_model = whisper.load_model("base")
            print("[STT] Whisper loaded!")
        except ImportError:
            print("[STT] Whisper not installed. Run: pip install openai-whisper")
            print("[STT] Using mock transcription")
        except Exception as e:
            print(f"[STT] Error loading Whisper: {e}")
    
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        
        # Pass through most frames
        if not hasattr(frame, 'audio'):
            await self.push_frame(frame, direction)
            return
        
        # Transcribe audio frames
        if self.whisper_model:
            try:
                import numpy as np
                import tempfile
                import soundfile as sf
                
                # Save audio to temp file
                audio_data = np.frombuffer(frame.audio, dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio_float, frame.sample_rate)
                    result = self.whisper_model.transcribe(f.name)
                    os.unlink(f.name)
                
                if result["text"].strip():
                    await self.push_frame(
                        TranscriptionFrame(text=result["text"].strip()),
                        direction
                    )
            except Exception as e:
                print(f"[STT] Transcription error: {e}")
        
        await self.push_frame(frame, direction)


async def run_voice_pipeline():
    """Run the full voice pipeline."""
    
    print("=" * 60)
    print("Qwen3-TTS Pipecat Voice Pipeline")
    print("=" * 60)
    
    # Check API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY not set in .env")
        return
    
    # Check for GPU
    use_megakernel = os.getenv("USE_MEGAKERNEL", "true").lower() == "true"
    gpu_available = False
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_properties(0).name
            print(f"[GPU] {gpu_name}")
            if "5090" in gpu_name:
                print("[GPU] RTX 5090 detected - Megakernel enabled!")
            else:
                print(f"[GPU] Note: Megakernel optimized for RTX 5090")
        else:
            print("[GPU] No CUDA GPU - using mock TTS")
            use_megakernel = False
    except ImportError:
        print("[GPU] PyTorch not available - using mock TTS")
        use_megakernel = False
    
    print()
    
    # Initialize services
    print("[Init] Setting up pipeline...")
    
    # Local audio transport (mic + speaker)
    transport = LocalAudioTransport(
        LocalAudioParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            sample_rate=24000,
        )
    )
    
    # Voice Activity Detection (free, local)
    vad = SileroVADService()
    print("[Init] ✓ Silero VAD (voice detection)")
    
    # STT (local Whisper - free)
    stt = SimpleSTT()
    print("[Init] ✓ Whisper STT (transcription)")
    
    # LLM (Google Gemini)
    llm = GoogleLLMService(
        api_key=google_api_key,
        model="gemini-1.5-flash",
    )
    print("[Init] ✓ Gemini LLM (conversation)")
    
    # TTS (our megakernel service!)
    if use_megakernel and gpu_available:
        tts = Qwen3TTSService(verbose=True)
        print("[Init] ✓ Qwen3-TTS Megakernel (speech synthesis)")
    else:
        tts = Qwen3TTSServiceMock()
        print("[Init] ✓ Mock TTS (no GPU)")
    
    # Logging processors
    transcription_logger = TranscriptionLogger()
    response_logger = ResponseLogger()
    
    # Build pipeline
    # STT → LLM → TTS (as per task requirement)
    pipeline = Pipeline([
        transport.input(),          # Mic input
        vad,                        # Voice activity detection
        stt,                        # Speech-to-text (Whisper)
        transcription_logger,       # Log user speech
        llm,                        # LLM response (Gemini)
        response_logger,            # Log assistant response
        tts,                        # Text-to-speech (Megakernel!)
        transport.output(),         # Speaker output
    ])
    
    print()
    print("=" * 60)
    print("Pipeline ready! Speak into your microphone.")
    print("Press Ctrl+C to stop.")
    print("=" * 60)
    print()
    
    # Run pipeline
    runner = PipelineRunner()
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )
    
    try:
        await runner.run(task)
    except KeyboardInterrupt:
        print("\n\nPipeline stopped.")
    
    # Print metrics
    if hasattr(task, 'metrics'):
        print("\n" + "=" * 60)
        print("Performance Metrics")
        print("=" * 60)
        print(task.metrics)


async def run_text_pipeline():
    """
    Run a text-only pipeline for testing without microphone.
    
    Useful for testing LLM + TTS without audio input.
    """
    print("=" * 60)
    print("Qwen3-TTS Text Pipeline (No Mic Required)")
    print("=" * 60)
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        return
    
    # Setup LLM
    try:
        import google.generativeai as genai
        genai.configure(api_key=google_api_key)
        llm = genai.GenerativeModel('gemini-1.5-flash')
        print("[Init] ✓ Gemini LLM")
    except Exception as e:
        print(f"[Init] LLM error: {e}")
        return
    
    # Setup TTS
    tts_engine = None
    try:
        from qwen3_tts import Qwen3TTSEngine
        tts_engine = Qwen3TTSEngine(verbose=False)
        print("[Init] ✓ Qwen3-TTS Engine")
    except Exception as e:
        print(f"[Init] TTS not available: {e}")
        print("[Init] Using mock TTS")
    
    print()
    print("Type your messages. Type 'quit' to exit.")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Get LLM response
            response = llm.generate_content(
                f"You are a helpful voice assistant. Keep responses brief.\n\nUser: {user_input}\nAssistant:"
            )
            assistant_text = response.text.strip()
            print(f"Assistant: {assistant_text}")
            
            # Generate and play audio
            if tts_engine:
                print("[Generating speech...]")
                audio_chunks = []
                async for chunk in tts_engine.generate_streaming(assistant_text):
                    audio_chunks.append(chunk)
                
                # Play audio
                try:
                    import sounddevice as sd
                    import numpy as np
                    
                    audio_bytes = b"".join(audio_chunks)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32) / 32767.0
                    
                    sd.play(audio_float, samplerate=24000)
                    sd.wait()
                    print("[Audio complete]\n")
                except ImportError:
                    print(f"[Generated {len(audio_chunks)} chunks - sounddevice not installed]\n")
            else:
                print("[Mock TTS - no audio]\n")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3-TTS Pipecat Pipeline")
    parser.add_argument(
        "--mode",
        choices=["voice", "text"],
        default="text",
        help="Pipeline mode: 'voice' for mic input, 'text' for keyboard input"
    )
    args = parser.parse_args()
    
    if args.mode == "voice":
        asyncio.run(run_voice_pipeline())
    else:
        asyncio.run(run_text_pipeline())


if __name__ == "__main__":
    main()
