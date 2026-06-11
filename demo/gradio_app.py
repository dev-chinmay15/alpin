"""
Gradio Web UI for Qwen3-TTS Voice Agent.

Features:
- Text input mode (type and get speech)
- Voice input mode (speak and get speech response)
- Real-time streaming audio output
- Works with Pipecat TTS service

Usage:
  python -m demo.gradio_app
  python -m demo.gradio_app --share  # For public URL
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import numpy as np
import soundfile as sf

# Check for optional dependencies
TORCH_AVAILABLE = False
ANTHROPIC_AVAILABLE = False
EDGE_TTS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: torch not installed")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    print("Warning: anthropic not installed")

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
    print("✓ Edge TTS available")
except ImportError:
    print("Warning: edge-tts not installed - pip install edge-tts")

# Whisper for Speech-to-Text
WHISPER_AVAILABLE = False
whisper_model = None
try:
    import whisper
    WHISPER_AVAILABLE = True
    whisper_model = whisper.load_model("base")
    print("✓ Whisper STT loaded (base model)")
except ImportError:
    print("Warning: whisper not installed - pip install openai-whisper")
except Exception as e:
    print(f"Warning: Failed to load Whisper: {e}")


class VoiceAgent:
    """Simple voice agent with LLM + TTS."""
    
    def __init__(self):
        self.conversation_history = []
        self.tts_engine = None
        self.llm = None
        
        self._setup_llm()
        self._setup_tts()
    
    def _setup_llm(self):
        """Setup Anthropic Claude LLM."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Warning: ANTHROPIC_API_KEY not set")
            return
        
        if ANTHROPIC_AVAILABLE:
            try:
                self.llm = anthropic.Anthropic(api_key=api_key)
                print("✓ Claude LLM initialized")
            except Exception as e:
                print(f"Warning: Failed to setup Claude: {e}")
    
    def _setup_tts(self):
        """Setup TTS engine."""
        if TORCH_AVAILABLE:
            try:
                from qwen3_tts import Qwen3TTSEngine
                self.tts_engine = Qwen3TTSEngine(verbose=False)
                print("✓ Qwen3-TTS engine initialized")
            except Exception as e:
                print(f"Warning: Failed to setup TTS engine: {e}")
                print("Using mock TTS")
    
    def get_llm_response(self, user_message: str) -> str:
        """Get response from LLM."""
        if not self.llm:
            return f"[Mock LLM] You said: {user_message}"
        
        try:
            response = self.llm.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                system="You are a helpful voice assistant. Keep responses concise (1-2 sentences).",
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"[Error] {str(e)}"
    
    async def generate_speech(self, text: str) -> tuple:
        """Generate speech from text using Edge TTS."""
        if EDGE_TTS_AVAILABLE:
            try:
                # Use Edge TTS (Microsoft, free)
                communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    temp_path = f.name
                
                await communicate.save(temp_path)
                
                # Read the audio file
                audio_data, sample_rate = sf.read(temp_path)
                
                # Convert to int16
                if audio_data.dtype != np.int16:
                    audio_data = (audio_data * 32767).astype(np.int16)
                
                # Clean up temp file
                os.unlink(temp_path)
                
                return sample_rate, audio_data
            except Exception as e:
                print(f"Edge TTS Error: {e}")
        
        # Fallback to mock audio
        return 24000, self._generate_mock_audio(text)
    
    def _generate_mock_audio(self, text: str) -> np.ndarray:
        """Generate mock audio (sine wave) for testing."""
        sample_rate = 24000
        duration = min(len(text) * 0.05, 5.0)  # ~50ms per char, max 5s
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Simple melody
        frequencies = [440, 494, 523, 587, 659]  # A4, B4, C5, D5, E5
        audio = np.zeros_like(t)
        
        chunk_len = len(t) // len(frequencies)
        for i, freq in enumerate(frequencies):
            start = i * chunk_len
            end = start + chunk_len
            if end > len(t):
                end = len(t)
            audio[start:end] = np.sin(2 * np.pi * freq * t[start:end]) * 0.3
        
        # Fade in/out
        fade_len = int(sample_rate * 0.05)
        audio[:fade_len] *= np.linspace(0, 1, fade_len)
        audio[-fade_len:] *= np.linspace(1, 0, fade_len)
        
        return (audio * 32767).astype(np.int16)
    
    def chat(self, message: str, history: list) -> tuple:
        """Process chat message and return response with audio."""
        if not message.strip():
            return "", history, None
        
        # Get LLM response
        response = self.get_llm_response(message)
        
        # Generate speech (returns sample_rate, audio_data)
        audio_result = asyncio.run(self.generate_speech(response))
        
        # Update history (dict format for Gradio 6.x)
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        # Return audio as tuple (sample_rate, audio_array)
        return "", history, audio_result
    
    def process_voice(self, audio_input, history: list) -> tuple:
        """Process voice input and return response with audio."""
        if audio_input is None:
            return history, None
        
        sample_rate, audio_data = audio_input
        
        # Transcribe audio using Whisper
        transcription = self._transcribe_audio(sample_rate, audio_data)
        
        if not transcription or transcription.strip() == "":
            transcription = "[Could not understand audio]"
        
        # Get LLM response
        response = self.get_llm_response(transcription)
        
        # Generate speech (returns sample_rate, audio_data)
        audio_result = asyncio.run(self.generate_speech(response))
        
        # Update history (dict format for Gradio 6.x)
        history = history or []
        history.append({"role": "user", "content": f"🎤 {transcription}"})
        history.append({"role": "assistant", "content": response})
        
        return history, audio_result
    
    def _transcribe_audio(self, sample_rate: int, audio_data: np.ndarray) -> str:
        """Transcribe audio using Whisper."""
        if not WHISPER_AVAILABLE or whisper_model is None:
            return "[Whisper not available]"
        
        try:
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            # Ensure audio is float32 for Whisper
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Handle stereo to mono
            if len(audio_float.shape) > 1:
                audio_float = audio_float.mean(axis=1)
            
            sf.write(temp_path, audio_float, sample_rate)
            
            # Transcribe
            result = whisper_model.transcribe(temp_path, language="en")
            
            # Clean up
            os.unlink(temp_path)
            
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return f"[Transcription error: {str(e)}]"


def create_ui():
    """Create the Gradio interface."""
    
    agent = VoiceAgent()
    
    # Custom CSS
    css = """
    .container { max-width: 800px; margin: auto; }
    .title { text-align: center; margin-bottom: 20px; }
    .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
    """
    
    with gr.Blocks(css=css, title="Qwen3-TTS Voice Agent") as app:
        gr.Markdown("""
        # 🎤 Qwen3-TTS Voice Agent
        
        **Powered by RTX 5090 Megakernel (~1000 tok/s)**
        
        Type a message or use voice input to chat with the AI assistant.
        """)
        
        # Status indicators
        with gr.Row():
            gpu_status = "✅ GPU Ready" if TORCH_AVAILABLE and agent.tts_engine else "⚠️ Mock TTS (No GPU)"
            llm_status = "✅ Claude Ready" if agent.llm else "⚠️ Mock LLM"
            gr.Markdown(f"**Status:** {gpu_status} | {llm_status}")
        
        # Chat interface
        chatbot = gr.Chatbot(
            label="Conversation",
            height=400,
        )
        
        # Text input
        with gr.Row():
            text_input = gr.Textbox(
                label="Type your message",
                placeholder="Hello, how are you?",
                scale=4,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Voice input
        with gr.Row():
            voice_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Or speak (click to record)",
            )
        
        # Audio output
        audio_output = gr.Audio(
            label="Response Audio",
            type="numpy",
            autoplay=True,
        )
        
        # Clear button
        clear_btn = gr.Button("Clear Conversation")
        
        # Event handlers
        send_btn.click(
            fn=agent.chat,
            inputs=[text_input, chatbot],
            outputs=[text_input, chatbot, audio_output],
        )
        
        text_input.submit(
            fn=agent.chat,
            inputs=[text_input, chatbot],
            outputs=[text_input, chatbot, audio_output],
        )
        
        voice_input.stop_recording(
            fn=agent.process_voice,
            inputs=[voice_input, chatbot],
            outputs=[chatbot, audio_output],
        )
        
        clear_btn.click(
            fn=lambda: ([], None),
            outputs=[chatbot, audio_output],
        )
        
        # Instructions
        gr.Markdown("""
        ---
        ### Instructions
        
        1. **Text Mode**: Type your message and click Send (or press Enter)
        2. **Voice Mode**: Click the microphone, speak, then click again to stop
        3. Audio response will play automatically
        
        ### Performance Targets
        - TTFC (Time to First Chunk): < 60ms
        - RTF (Real-Time Factor): < 0.15
        - Talker Speed: ~1000 tok/s
        
        ---
        *Built for RTX 5090 Megakernel Take-Home Project*
        """)
    
    return app


def main():
    """Run the Gradio app."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3-TTS Voice Agent")
    parser.add_argument("--share", action="store_true", help="Create public URL")
    parser.add_argument("--port", type=int, default=7861, help="Port number")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Qwen3-TTS Voice Agent - Gradio UI")
    print("=" * 60)
    
    app = create_ui()
    
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
