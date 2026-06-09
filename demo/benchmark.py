"""
Benchmark script for Qwen3-TTS Megakernel.

Measures:
- TTFC (Time to First Chunk)
- RTF (Real-Time Factor)
- Talker tok/s
- End-to-end latency

Usage:
  python -m demo.benchmark
"""

import asyncio
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from qwen3_tts import Qwen3TTSEngine
from qwen3_tts.config import config


# Test sentences of varying lengths
TEST_SENTENCES = [
    "Hello, how are you today?",
    "The weather is beautiful outside, perfect for a walk in the park.",
    "Artificial intelligence is transforming how we interact with technology, making voice assistants more natural and responsive than ever before.",
]

WARMUP_RUNS = 3
BENCHMARK_RUNS = 5


async def benchmark_single(engine: Qwen3TTSEngine, text: str) -> dict:
    """Benchmark a single text generation."""
    
    engine.reset()
    
    start_time = time.perf_counter()
    first_chunk_time = None
    chunks = []
    total_samples = 0
    
    async for chunk in engine.generate_streaming(text):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - start_time
        chunks.append(chunk)
        total_samples += len(chunk) // 2  # 16-bit samples
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Calculate metrics
    audio_duration = total_samples / config.sample_rate
    rtf = total_time / audio_duration if audio_duration > 0 else float('inf')
    
    metrics = engine.get_metrics()
    talker_toks = 1000 / metrics["talker_avg_ms"] if metrics["talker_avg_ms"] > 0 else 0
    
    return {
        "text_length": len(text),
        "audio_duration_s": audio_duration,
        "total_time_s": total_time,
        "ttfc_ms": first_chunk_time * 1000 if first_chunk_time else 0,
        "rtf": rtf,
        "talker_tok_s": talker_toks,
        "talker_avg_ms": metrics["talker_avg_ms"],
        "code_predictor_avg_ms": metrics["code_predictor_avg_ms"],
        "decoder_avg_ms": metrics["decoder_avg_ms"],
        "num_chunks": len(chunks),
    }


async def run_benchmark():
    """Run the full benchmark suite."""
    
    print("=" * 70)
    print("Qwen3-TTS Megakernel Benchmark")
    print("=" * 70)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_properties(0).name
            print(f"GPU: {gpu_name}")
        else:
            print("GPU: Not available (results will not be representative)")
    except:
        print("GPU: PyTorch not available")
    
    print()
    
    # Initialize engine
    print("Initializing engine...")
    engine = Qwen3TTSEngine(verbose=False)
    
    # Warmup
    print(f"\nWarming up ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        async for _ in engine.generate_streaming("Hello world"):
            pass
    
    # Benchmark
    print(f"\nRunning benchmarks ({BENCHMARK_RUNS} runs per sentence)...")
    print()
    
    all_results = []
    
    for i, text in enumerate(TEST_SENTENCES):
        print(f"Sentence {i+1}: \"{text[:50]}...\"")
        
        sentence_results = []
        for run in range(BENCHMARK_RUNS):
            result = await benchmark_single(engine, text)
            sentence_results.append(result)
        
        # Average results
        avg_result = {
            "text_length": sentence_results[0]["text_length"],
            "audio_duration_s": np.mean([r["audio_duration_s"] for r in sentence_results]),
            "total_time_s": np.mean([r["total_time_s"] for r in sentence_results]),
            "ttfc_ms": np.mean([r["ttfc_ms"] for r in sentence_results]),
            "rtf": np.mean([r["rtf"] for r in sentence_results]),
            "talker_tok_s": np.mean([r["talker_tok_s"] for r in sentence_results]),
            "talker_avg_ms": np.mean([r["talker_avg_ms"] for r in sentence_results]),
            "code_predictor_avg_ms": np.mean([r["code_predictor_avg_ms"] for r in sentence_results]),
            "decoder_avg_ms": np.mean([r["decoder_avg_ms"] for r in sentence_results]),
        }
        
        all_results.append(avg_result)
        
        print(f"  TTFC: {avg_result['ttfc_ms']:.1f} ms")
        print(f"  RTF:  {avg_result['rtf']:.3f}")
        print(f"  Talker: {avg_result['talker_tok_s']:.0f} tok/s")
        print()
    
    # Overall summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    overall_ttfc = np.mean([r["ttfc_ms"] for r in all_results])
    overall_rtf = np.mean([r["rtf"] for r in all_results])
    overall_talker = np.mean([r["talker_tok_s"] for r in all_results])
    
    print(f"Average TTFC:        {overall_ttfc:.1f} ms (target: <60 ms)")
    print(f"Average RTF:         {overall_rtf:.3f} (target: <0.15)")
    print(f"Average Talker:      {overall_talker:.0f} tok/s (target: ~1000)")
    print()
    
    # Check targets
    print("Target Check:")
    print(f"  TTFC < 60ms:  {'✓ PASS' if overall_ttfc < 60 else '✗ FAIL'}")
    print(f"  RTF < 0.15:   {'✓ PASS' if overall_rtf < 0.15 else '✗ FAIL'}")
    print(f"  Tok/s ~1000:  {'✓ PASS' if overall_talker > 900 else '✗ FAIL'}")
    print()
    
    # Per-component breakdown
    print("Per-Component Timing (avg):")
    print(f"  Talker step:      {np.mean([r['talker_avg_ms'] for r in all_results]):.2f} ms")
    print(f"  Code Predictor:   {np.mean([r['code_predictor_avg_ms'] for r in all_results]):.2f} ms")
    print(f"  Speech Decoder:   {np.mean([r['decoder_avg_ms'] for r in all_results]):.2f} ms")
    print()
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
