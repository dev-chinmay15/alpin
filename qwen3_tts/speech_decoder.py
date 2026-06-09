"""
Speech Decoder for Qwen3-TTS.

Converts discrete codebook tokens into audio waveform.
Uses a causal ConvNet decoder for streaming capability.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Iterator

from .config import config


class SpeechDecoder:
    """
    Speech Tokenizer Decoder for Qwen3-TTS.
    
    Converts 16 codebook tokens per frame into audio samples.
    The decoder is causal, enabling streaming output.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS",
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        self.decoder = None
        
        self._load_decoder()
    
    def _load_decoder(self):
        """Load the speech tokenizer decoder."""
        try:
            from transformers import AutoModel
            
            if self.verbose:
                print("[SpeechDecoder] Loading decoder...")
            
            # Load full model and extract speech tokenizer decoder
            full_model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Decoder typically uses FP32
                device_map=self.device,
                trust_remote_code=True,
            )
            
            # Extract speech tokenizer decoder
            if hasattr(full_model, 'speech_tokenizer'):
                self.decoder = full_model.speech_tokenizer
                if self.verbose:
                    print("[SpeechDecoder] Loaded speech_tokenizer")
            elif hasattr(full_model, 'codec_decoder'):
                self.decoder = full_model.codec_decoder
                if self.verbose:
                    print("[SpeechDecoder] Loaded codec_decoder")
            else:
                self.decoder = full_model
                if self.verbose:
                    print("[SpeechDecoder] Using full model")
            
            if self.decoder is not None:
                self.decoder.eval()
            
        except Exception as e:
            print(f"[SpeechDecoder] Failed to load decoder: {e}")
            self.decoder = None
    
    @torch.no_grad()
    def decode_frame(self, codebook_tokens: List[int]) -> np.ndarray:
        """
        Decode a single frame (16 codebook tokens) to audio.
        
        Args:
            codebook_tokens: List of 16 tokens (one per codebook)
            
        Returns:
            Audio samples as numpy array [samples_per_frame]
        """
        if len(codebook_tokens) != 16:
            raise ValueError(f"Expected 16 codebook tokens, got {len(codebook_tokens)}")
        
        if self.decoder is None:
            return self._mock_decode_frame()
        
        try:
            # Convert tokens to tensor
            tokens = torch.tensor(codebook_tokens, dtype=torch.long, device=self.device)
            tokens = tokens.unsqueeze(0)  # [1, 16]
            
            # Decode
            if hasattr(self.decoder, 'decode'):
                audio = self.decoder.decode(tokens)
            elif hasattr(self.decoder, 'forward'):
                audio = self.decoder(tokens)
            else:
                return self._mock_decode_frame()
            
            # Convert to numpy
            if isinstance(audio, torch.Tensor):
                audio = audio.squeeze().cpu().numpy()
            
            return audio.astype(np.float32)
            
        except Exception as e:
            print(f"[SpeechDecoder] Decode error: {e}")
            return self._mock_decode_frame()
    
    def _mock_decode_frame(self) -> np.ndarray:
        """Generate mock audio for testing."""
        # Generate silence with tiny noise
        return np.random.randn(config.samples_per_frame).astype(np.float32) * 0.001
    
    def decode_streaming(
        self,
        token_stream: Iterator[List[int]],
    ) -> Iterator[np.ndarray]:
        """
        Streaming decode: yields audio chunks as tokens arrive.
        
        Args:
            token_stream: Iterator yielding lists of 16 codebook tokens
            
        Yields:
            Audio chunks as numpy arrays
        """
        for codebook_tokens in token_stream:
            audio_chunk = self.decode_frame(codebook_tokens)
            yield audio_chunk


class StreamingSpeechDecoder:
    """
    Optimized streaming decoder with buffering for smooth playback.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS",
        device: str = "cuda",
        buffer_frames: int = 4,  # Buffer 4 frames for smoother streaming
    ):
        self.decoder = SpeechDecoder(model_name, device, verbose=False)
        self.buffer_frames = buffer_frames
        self._buffer: List[np.ndarray] = []
    
    def reset(self):
        """Reset the buffer."""
        self._buffer = []
    
    def decode_and_buffer(self, codebook_tokens: List[int]) -> Optional[np.ndarray]:
        """
        Decode a frame and return buffered audio when ready.
        
        Returns audio chunk when buffer is full, None otherwise.
        """
        audio = self.decoder.decode_frame(codebook_tokens)
        self._buffer.append(audio)
        
        if len(self._buffer) >= self.buffer_frames:
            # Concatenate and return
            output = np.concatenate(self._buffer)
            self._buffer = []
            return output
        
        return None
    
    def flush(self) -> Optional[np.ndarray]:
        """Flush remaining buffer."""
        if self._buffer:
            output = np.concatenate(self._buffer)
            self._buffer = []
            return output
        return None
