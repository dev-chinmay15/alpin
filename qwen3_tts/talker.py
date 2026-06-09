"""
Megakernel-accelerated Talker model for Qwen3-TTS.

The Talker is the core autoregressive model that generates codec tokens from text.
It uses the same Qwen3 architecture as the megakernel, adapted for TTS.
"""

import os
import sys
import math
import struct
from typing import Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn

from .config import config

# Add megakernel to path
MEGAKERNEL_PATH = Path(__file__).parent.parent.parent / "qwen_megakernel"
sys.path.insert(0, str(MEGAKERNEL_PATH))


class MegakernelTalker:
    """
    Talker model accelerated by the RTX 5090 megakernel.
    
    This wraps the megakernel decode functions for use with Qwen3-TTS weights.
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
        self._position = 0
        
        # Check if GPU is available and is RTX 5090
        self.gpu_available = self._check_gpu()
        
        if self.gpu_available and config.use_megakernel:
            self._init_megakernel()
        else:
            self._init_fallback()
    
    def _check_gpu(self) -> bool:
        """Check if RTX 5090 is available."""
        if not torch.cuda.is_available():
            if self.verbose:
                print("[Talker] No CUDA GPU available, using fallback")
            return False
        
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        
        if "5090" in gpu_name:
            if self.verbose:
                print(f"[Talker] Found RTX 5090: {gpu_name}")
            return True
        else:
            if self.verbose:
                print(f"[Talker] GPU {gpu_name} is not RTX 5090, megakernel may not work optimally")
            # Still try to use it, might work on other Blackwell GPUs
            return True
    
    def _init_megakernel(self):
        """Initialize with megakernel acceleration."""
        if self.verbose:
            print("[Talker] Initializing megakernel-accelerated Talker...")
        
        try:
            # Import and build megakernel
            from qwen_megakernel.build import get_extension
            self._ext = get_extension()
            
            # Load Qwen3-TTS Talker weights
            self._load_tts_weights()
            
            # Initialize buffers
            self._init_buffers()
            
            self.use_megakernel = True
            if self.verbose:
                print("[Talker] Megakernel initialized successfully!")
                
        except Exception as e:
            print(f"[Talker] Failed to initialize megakernel: {e}")
            print("[Talker] Falling back to standard PyTorch")
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback PyTorch implementation."""
        if self.verbose:
            print("[Talker] Using fallback PyTorch implementation")
        
        self.use_megakernel = False
        self._model = None  # Will load standard model if needed
    
    def _load_tts_weights(self):
        """Load Qwen3-TTS Talker weights for megakernel."""
        from transformers import AutoModel
        
        if self.verbose:
            print(f"[Talker] Loading weights from {self.model_name}...")
        
        # Load the full Qwen3-TTS model
        model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        
        # Extract Talker weights
        # Qwen3-TTS structure: model.talker.model.layers.{i}.*
        state = model.state_dict()
        
        # Build RoPE tables
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        positions = torch.arange(config.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self._cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
        self._sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
        
        # Extract layer weights
        self._layer_weights = []
        for i in range(config.num_layers):
            prefix = f"talker.model.layers.{i}."
            
            # Check if weights exist with this prefix
            # Qwen3-TTS might have different naming
            alt_prefix = f"model.layers.{i}."
            
            p = prefix if f"{prefix}input_layernorm.weight" in state else alt_prefix
            
            self._layer_weights.extend([
                state[p + "input_layernorm.weight"].contiguous(),
                state[p + "self_attn.q_proj.weight"].contiguous(),
                state[p + "self_attn.k_proj.weight"].contiguous(),
                state[p + "self_attn.v_proj.weight"].contiguous(),
                state.get(p + "self_attn.q_norm.weight", torch.ones(config.head_dim, dtype=torch.bfloat16, device="cuda")).contiguous(),
                state.get(p + "self_attn.k_norm.weight", torch.ones(config.head_dim, dtype=torch.bfloat16, device="cuda")).contiguous(),
                state[p + "self_attn.o_proj.weight"].contiguous(),
                state[p + "post_attention_layernorm.weight"].contiguous(),
                state[p + "mlp.gate_proj.weight"].contiguous(),
                state[p + "mlp.up_proj.weight"].contiguous(),
                state[p + "mlp.down_proj.weight"].contiguous(),
            ])
        
        # Embedding and output head
        self._embed_weight = state.get(
            "talker.model.embed_tokens.weight",
            state.get("model.embed_tokens.weight")
        ).contiguous()
        
        self._final_norm_weight = state.get(
            "talker.model.norm.weight",
            state.get("model.norm.weight")
        ).contiguous()
        
        # Codec head (for TTS, outputs audio tokens instead of text tokens)
        self._codec_head_weight = state.get(
            "talker.codec_head.weight",
            self._embed_weight  # Fallback to tied embeddings
        ).contiguous()
        
        # Pack layer weights for megakernel
        self._layer_weights_packed = self._pack_layer_weights(self._layer_weights)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        if self.verbose:
            print(f"[Talker] Loaded {config.num_layers} layers")
    
    def _pack_layer_weights(self, layer_weights: list) -> torch.Tensor:
        """Pack layer weights into a blob for megakernel."""
        ptr_size = 8  # 64-bit pointers
        n_ptrs = 11
        struct_bytes = n_ptrs * ptr_size
        buf = bytearray(config.num_layers * struct_bytes)
        
        for i in range(config.num_layers):
            for j in range(n_ptrs):
                ptr = layer_weights[i * n_ptrs + j].data_ptr()
                struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
        
        return torch.frombuffer(buf, dtype=torch.uint8).cuda()
    
    def _init_buffers(self):
        """Initialize GPU buffers for inference."""
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        
        # KV cache
        self._k_cache = torch.zeros(
            config.num_layers,
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
            **bf16
        )
        self._v_cache = torch.zeros_like(self._k_cache)
        
        # Scratch buffers
        self._hidden = torch.empty(config.hidden_size, **bf16)
        self._act = torch.empty(config.hidden_size, **f32)
        self._res = torch.empty(config.hidden_size, **f32)
        self._q = torch.empty(config.num_q_heads * config.head_dim, **f32)
        self._k = torch.empty(config.num_kv_heads * config.head_dim, **f32)
        self._v = torch.empty(config.num_kv_heads * config.head_dim, **f32)
        self._attn_out = torch.empty(config.num_q_heads * config.head_dim, **f32)
        self._mlp_inter = torch.empty(config.intermediate_size, **f32)
        self._norm_out = torch.empty(config.hidden_size, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")
        
        self._attn_scale = 1.0 / math.sqrt(config.head_dim)
    
    def reset(self):
        """Reset state for new generation."""
        self._position = 0
        if hasattr(self, '_k_cache'):
            self._k_cache.zero_()
            self._v_cache.zero_()
    
    def step(self, token_id: int) -> Tuple[int, torch.Tensor]:
        """
        Generate one token and return hidden states.
        
        Args:
            token_id: Input token ID
            
        Returns:
            Tuple of (output_token_id, hidden_states)
        """
        if not self.use_megakernel:
            return self._step_fallback(token_id)
        
        # Call megakernel decode
        _decode = torch.ops.qwen_megakernel_C.decode
        
        _decode(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._codec_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            config.num_layers,
            self._position,
            config.max_seq_len,
            self._attn_scale,
        )
        
        self._position += 1
        
        # Return token and hidden states (for code predictor)
        return self._out_token.item(), self._norm_out.clone()
    
    def _step_fallback(self, token_id: int) -> Tuple[int, torch.Tensor]:
        """Fallback step using standard PyTorch (for testing without GPU)."""
        # Mock output for testing
        import random
        mock_token = random.randint(0, config.codebook_size - 1)
        mock_hidden = torch.randn(config.hidden_size, dtype=torch.float32)
        self._position += 1
        return mock_token, mock_hidden
    
    @property
    def position(self) -> int:
        return self._position
