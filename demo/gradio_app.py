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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import numpy as np

# Check for optional dependencies
TORCH_AVAILABLE = False
ANTHROPIC_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: torch not installed - TTS will use mock")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    print("Warning: anthropic not installed")


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
    
    async def generate_speech(self, text: str) -> np.ndarray:
        """Generate speech from text."""
        if self.tts_engine:
            try:
                audio_chunks = []
                async for chunk in self.tts_engine.generate_streaming(text):
                    # Convert bytes to numpy
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    audio_chunks.append(audio_array)
                
                if audio_chunks:
                    return np.concatenate(audio_chunks)
            except Exception as e:
                print(f"TTS Error: {e}")
        
        # Mock audio - generate a simple tone
        return self._generate_mock_audio(text)
    
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
        
        # Generate speech
        audio = asyncio.run(self.generate_speech(response))
        
        # Update history (dict format for Gradio 6.x)
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        # Return audio as tuple (sample_rate, audio_array)
        return "", history, (24000, audio)
    
    def process_voice(self, audio_input, history: list) -> tuple:
        """Process voice input and return response with audio."""
        if audio_input is None:
            return history, None
        
        sample_rate, audio_data = audio_input
        
        # For now, we'll use a simple approach
        # In production, this would use Whisper or browser STT
        # The Gradio interface handles browser-based transcription
        
        # Mock transcription (in real app, audio would be transcribed)
        transcription = "[Voice input received - transcription requires Whisper]"
        
        # Get LLM response
        response = self.get_llm_response(transcription)
        
        # Generate speech
        audio = asyncio.run(self.generate_speech(response))
        
        # Update history (dict format for Gradio 6.x)
        history = history or []
        history.append({"role": "user", "content": transcription})
        history.append({"role": "assistant", "content": response})
        
        return history, (24000, audio)


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
