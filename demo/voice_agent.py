"""
Voice Agent Demo using Qwen3-TTS with Pipecat.

Complete voice pipeline:
  User speaks -> STT -> LLM (Claude) -> TTS (Qwen3 Megakernel) -> Audio output

Usage:
  python -m demo.voice_agent
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.services.anthropic import AnthropicLLMService
from pipecat.transports.local.audio import LocalAudioTransport

# Import our TTS service
from qwen3_tts import Qwen3TTSService
from qwen3_tts.pipecat_service import Qwen3TTSServiceMock


async def main():
    """Run the voice agent demo."""
    
    print("=" * 60)
    print("Qwen3-TTS Voice Agent Demo")
    print("=" * 60)
    
    # Check for API key
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        return
    
    # Check if we have GPU for megakernel
    use_megakernel = os.getenv("USE_MEGAKERNEL", "true").lower() == "true"
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_properties(0).name
            print(f"GPU: {gpu_name}")
        else:
            print("GPU: Not available")
            use_megakernel = False
    except:
        gpu_available = False
        use_megakernel = False
        print("GPU: PyTorch CUDA not available")
    
    print(f"Megakernel: {'Enabled' if use_megakernel else 'Disabled (using mock)'}")
    print("=" * 60)
    
    # Initialize services
    
    # LLM - Anthropic Claude
    llm = AnthropicLLMService(
        api_key=anthropic_api_key,
        model="claude-haiku-4-5-20251001",
    )
    
    # TTS - Qwen3 with Megakernel (or mock if no GPU)
    if use_megakernel and gpu_available:
        tts = Qwen3TTSService(verbose=True)
    else:
        print("Using mock TTS service (no RTX 5090 detected)")
        tts = Qwen3TTSServiceMock()
    
    # System prompt
    messages = [
        {
            "role": "system",
            "content": """You are a helpful voice assistant. Keep your responses 
            concise and conversational. Respond in 1-2 sentences when possible.
            You are demonstrating a high-performance TTS system running at 
            1000 tokens per second."""
        }
    ]
    
    # For demo without microphone, use text input
    print("\nDemo Mode: Text Input -> LLM -> TTS -> Audio Output")
    print("Type your message and press Enter. Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Get LLM response
            print("Assistant: ", end="", flush=True)
            
            response = await llm.generate(
                messages=messages,
            )
            
            assistant_text = response.get("content", "")
            print(assistant_text)
            
            # Add to history
            messages.append({"role": "assistant", "content": assistant_text})
            
            # Generate speech
            print("\n[Generating speech...]")
            
            audio_chunks = []
            async for frame in tts.run_tts(assistant_text):
                if hasattr(frame, 'audio'):
                    audio_chunks.append(frame.audio)
            
            # Play audio (if sounddevice available)
            try:
                import sounddevice as sd
                import numpy as np
                
                audio_bytes = b"".join(audio_chunks)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32767.0
                
                sd.play(audio_float, samplerate=24000)
                sd.wait()
                print("[Audio playback complete]\n")
            except ImportError:
                print("[sounddevice not installed - skipping playback]")
                print(f"[Generated {len(audio_chunks)} audio chunks]\n")
            except Exception as e:
                print(f"[Playback error: {e}]\n")
                
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


async def run_full_pipeline():
    """
    Run full voice pipeline with microphone input.
    
    Requires:
    - Working microphone
    - sounddevice installed
    - STT service (Deepgram or similar)
    """
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioParams
    from pipecat.services.silero import SileroVADService
    
    print("Starting full voice pipeline...")
    print("Speak into your microphone. Press Ctrl+C to stop.")
    
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Transport for local audio I/O
    transport = LocalAudioTransport(
        LocalAudioParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            sample_rate=24000,
        )
    )
    
    # Voice Activity Detection
    vad = SileroVADService()
    
    # LLM
    llm = AnthropicLLMService(
        api_key=anthropic_api_key,
        model="claude-haiku-4-5-20251001",
    )
    
    # TTS
    tts = Qwen3TTSService(verbose=True)
    
    # Build pipeline
    pipeline = Pipeline([
        transport.input(),
        vad,
        # TODO: plug STT service if running full mic pipeline
        llm,
        tts,
        transport.output(),
    ])
    
    # Run
    runner = PipelineRunner()
    task = PipelineTask(pipeline, PipelineParams())
    
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
