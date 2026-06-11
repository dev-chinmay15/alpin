"""
Configuration for Qwen3-TTS Megakernel
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TTSConfig:
    """Configuration for the TTS engine."""
    
    # Model settings
    model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    use_megakernel: bool = True
    max_seq_len: int = 2048
    
    # Audio settings
    sample_rate: int = 24000
    channels: int = 1
    frame_rate: float = 12.5  # Qwen3-TTS runs at 12.5 Hz
    samples_per_frame: int = 1920  # 24000 / 12.5 = 1920
    
    # Talker model architecture (Qwen3-0.6B based)
    hidden_size: int = 1024
    num_layers: int = 28
    num_q_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    
    # Code Predictor
    code_predictor_layers: int = 5
    num_codebooks: int = 16
    codebook_size: int = 2048
    
    # Performance
    device: str = "cuda"
    dtype: str = "bfloat16"
    
    @classmethod
    def from_env(cls) -> "TTSConfig":
        """Load configuration from environment variables."""
        return cls(
            model_name=os.getenv("MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            use_megakernel=os.getenv("USE_MEGAKERNEL", "true").lower() == "true",
            max_seq_len=int(os.getenv("MAX_SEQ_LEN", "2048")),
            sample_rate=int(os.getenv("SAMPLE_RATE", "24000")),
            channels=int(os.getenv("CHANNELS", "1")),
        )


# Global config instance
config = TTSConfig.from_env()
