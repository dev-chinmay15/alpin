"""
Qwen3-TTS Engine - Main TTS pipeline combining all components.

Provides streaming text-to-speech using:
- Megakernel-accelerated Talker
- Code Predictor
- Streaming Speech Decoder
"""

import asyncio
import time
from typing import AsyncIterator, List, Optional
import numpy as np
import torch

from .config import config, TTSConfig
from .talker import MegakernelTalker
from .code_predictor import CodePredictor
from .speech_decoder import SpeechDecoder, StreamingSpeechDecoder


class Qwen3TTSEngine:
    """
    Main TTS engine for Qwen3-TTS with megakernel acceleration.
    
    Pipeline:
        Text -> Talker (Megakernel) -> Codebook 0
                    |
                    v (hidden states)
              Code Predictor -> Codebooks 1-15
                    |
                    v
              Speech Decoder -> Audio
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "cuda",
        verbose: bool = True,
        buffer_frames: int = 1,  # Minimal buffering for low latency
    ):
        self.model_name = model_name or config.model_name
        self.device = device
        self.verbose = verbose
        
        # Initialize components
        if self.verbose:
            print("=" * 50)
            print("Initializing Qwen3-TTS Engine")
            print("=" * 50)
        
        self.talker = MegakernelTalker(
            model_name=self.model_name,
            device=device,
            verbose=verbose,
        )
        
        self.code_predictor = CodePredictor(
            model_name=self.model_name,
            device=device,
            verbose=verbose,
        )
        
        self.speech_decoder = StreamingSpeechDecoder(
            model_name=self.model_name,
            device=device,
            buffer_frames=buffer_frames,
        )
        
        # Tokenizer for text
        self.tokenizer = None
        self._load_tokenizer()
        
        # Performance tracking
        self.metrics = {
            "talker_times": [],
            "code_predictor_times": [],
            "decoder_times": [],
        }
        
        if self.verbose:
            print("=" * 50)
            print("Qwen3-TTS Engine initialized!")
            print(f"  Megakernel: {'enabled' if self.talker.use_megakernel else 'disabled'}")
            print("=" * 50)
    
    def _load_tokenizer(self):
        """Load the text tokenizer."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            if self.verbose:
                print("[Engine] Tokenizer loaded")
        except Exception as e:
            print(f"[Engine] Failed to load tokenizer: {e}")
            self.tokenizer = None
    
    def reset(self):
        """Reset state for new generation."""
        self.talker.reset()
        self.speech_decoder.reset()
        self.metrics = {
            "talker_times": [],
            "code_predictor_times": [],
            "decoder_times": [],
        }
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize input text."""
        if self.tokenizer is None:
            # Mock tokenization
            return [ord(c) % 1000 for c in text]
        
        return self.tokenizer.encode(text, add_special_tokens=True)
    
    async def generate_streaming(
        self,
        text: str,
        speaker_id: int = 0,
    ) -> AsyncIterator[bytes]:
        """
        Generate speech from text with streaming output.
        
        Args:
            text: Input text to synthesize
            speaker_id: Speaker ID for voice selection
            
        Yields:
            Audio chunks as bytes (16-bit PCM)
        """
        self.reset()
        
        # Tokenize text
        input_ids = self._tokenize(text)
        
        if self.verbose:
            print(f"[Engine] Generating speech for: '{text[:50]}...'")
            print(f"[Engine] Input tokens: {len(input_ids)}")
        
        # Process all input tokens except last (prefill)
        for token_id in input_ids[:-1]:
            self.talker.step(token_id)
        
        # Start generation from last token
        current_token = input_ids[-1]
        generated_frames = 0
        eos_token = 2150  # Qwen3-TTS EOS token
        
        start_time = time.perf_counter()
        first_chunk_time = None
        
        while generated_frames < config.max_seq_len:
            # Step 1: Talker generates codebook 0 token
            t0 = time.perf_counter()
            codebook_0, hidden_states = self.talker.step(current_token)
            self.metrics["talker_times"].append(time.perf_counter() - t0)
            
            # Check for EOS
            if codebook_0 == eos_token:
                break
            
            # Step 2: Code Predictor generates codebooks 1-15
            t0 = time.perf_counter()
            codebooks_1_15 = self.code_predictor.predict(hidden_states, codebook_0)
            self.metrics["code_predictor_times"].append(time.perf_counter() - t0)
            
            # Combine all codebooks
            all_codebooks = [codebook_0] + codebooks_1_15
            
            # Step 3: Speech Decoder converts to audio
            t0 = time.perf_counter()
            audio_chunk = self.speech_decoder.decode_and_buffer(all_codebooks)
            self.metrics["decoder_times"].append(time.perf_counter() - t0)
            
            # Yield audio if ready
            if audio_chunk is not None:
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter() - start_time
                    if self.verbose:
                        print(f"[Engine] TTFC: {first_chunk_time * 1000:.1f} ms")
                
                # Convert to 16-bit PCM bytes
                audio_bytes = self._to_pcm_bytes(audio_chunk)
                yield audio_bytes
            
            # Update for next iteration
            current_token = codebook_0
            generated_frames += 1
            
            # Yield control to event loop
            await asyncio.sleep(0)
        
        # Flush remaining buffer
        remaining = self.speech_decoder.flush()
        if remaining is not None:
            yield self._to_pcm_bytes(remaining)
        
        # Log performance
        if self.verbose:
            self._log_performance(start_time, generated_frames)
    
    def generate_sync(self, text: str, speaker_id: int = 0) -> bytes:
        """
        Synchronous generation - returns complete audio.
        
        Note: For streaming use generate_streaming() instead.
        """
        async def collect():
            chunks = []
            async for chunk in self.generate_streaming(text, speaker_id):
                chunks.append(chunk)
            return b"".join(chunks)
        
        return asyncio.run(collect())
    
    def _to_pcm_bytes(self, audio: np.ndarray) -> bytes:
        """Convert float audio to 16-bit PCM bytes."""
        # Clip and convert to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()
    
    def _log_performance(self, start_time: float, num_frames: int):
        """Log performance metrics."""
        total_time = time.perf_counter() - start_time
        audio_duration = num_frames * (1.0 / config.frame_rate)  # seconds
        
        avg_talker = np.mean(self.metrics["talker_times"]) * 1000 if self.metrics["talker_times"] else 0
        avg_code_pred = np.mean(self.metrics["code_predictor_times"]) * 1000 if self.metrics["code_predictor_times"] else 0
        avg_decoder = np.mean(self.metrics["decoder_times"]) * 1000 if self.metrics["decoder_times"] else 0
        
        rtf = total_time / audio_duration if audio_duration > 0 else 0
        talker_toks = 1000 / avg_talker if avg_talker > 0 else 0
        
        print("\n" + "=" * 50)
        print("Performance Metrics")
        print("=" * 50)
        print(f"  Frames generated:     {num_frames}")
        print(f"  Audio duration:       {audio_duration:.2f} s")
        print(f"  Generation time:      {total_time:.2f} s")
        print(f"  RTF:                  {rtf:.3f}")
        print(f"  Talker tok/s:         {talker_toks:.1f}")
        print(f"  Avg Talker step:      {avg_talker:.2f} ms")
        print(f"  Avg Code Predictor:   {avg_code_pred:.2f} ms")
        print(f"  Avg Decoder:          {avg_decoder:.2f} ms")
        print("=" * 50)
    
    def get_metrics(self) -> dict:
        """Get performance metrics."""
        return {
            "talker_avg_ms": np.mean(self.metrics["talker_times"]) * 1000 if self.metrics["talker_times"] else 0,
            "code_predictor_avg_ms": np.mean(self.metrics["code_predictor_times"]) * 1000 if self.metrics["code_predictor_times"] else 0,
            "decoder_avg_ms": np.mean(self.metrics["decoder_times"]) * 1000 if self.metrics["decoder_times"] else 0,
        }
