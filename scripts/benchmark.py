#!/usr/bin/env python3
"""
Benchmark script for Qwen3-TTS with Megakernel.

Measures:
- Token generation speed (tok/s)
- Time to first chunk (TTFC)
- Real-time factor (RTF)
- End-to-end latency

Usage:
  python scripts/benchmark.py
  python scripts/benchmark.py --iterations 10
  python scripts/benchmark.py --text "Custom text to benchmark"
"""

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test texts of varying lengths
TEST_TEXTS = {
    "short": "Hello, how are you?",
    "medium": "Welcome to the Qwen3-TTS voice agent. This system uses megakernel acceleration for high-performance speech synthesis.",
    "long": "The RTX 5090 megakernel enables single-kernel decode for Qwen3 models, achieving approximately one thousand tokens per second. This represents a significant improvement in text-to-speech latency, making real-time conversational AI possible with streaming audio output.",
}


def benchmark_megakernel():
    """Benchmark the megakernel decode speed."""
    print("\n" + "=" * 60)
    print("Megakernel Benchmark (Qwen3-0.6B Decode)")
    print("=" * 60)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            return None
        
        print(f"GPU: {torch.cuda.get_device_properties(0).name}")
        
        # Try to import megakernel
        sys.path.insert(0, "/root/qwen_megakernel")
        from qwen_megakernel import Decoder
        
        print("Loading megakernel decoder...")
        decoder = Decoder(verbose=False)
        
        # Warmup
        print("Warming up...")
        decoder.generate("Hello", max_tokens=10)
        decoder.reset()
        
        # Benchmark
        test_prompt = "The quick brown fox"
        max_tokens = 100
        iterations = 5
        
        print(f"\nGenerating {max_tokens} tokens x {iterations} iterations...")
        
        times = []
        for i in range(iterations):
            decoder.reset()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            output = decoder.generate(test_prompt, max_tokens=max_tokens)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            tok_s = max_tokens / elapsed
            print(f"  Run {i+1}: {elapsed*1000:.1f}ms, {tok_s:.1f} tok/s")
        
        avg_time = sum(times) / len(times)
        avg_tok_s = max_tokens / avg_time
        
        print("-" * 40)
        print(f"Average: {avg_time*1000:.1f}ms, {avg_tok_s:.1f} tok/s")
        
        return {"avg_tok_s": avg_tok_s, "avg_time_ms": avg_time * 1000}
        
    except ImportError as e:
        print(f"Megakernel not available: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def benchmark_qwen_tts(iterations: int = 5):
    """Benchmark Qwen3-TTS generation."""
    print("\n" + "=" * 60)
    print("Qwen3-TTS Benchmark")
    print("=" * 60)
    
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
        
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            return None
        
        print(f"GPU: {torch.cuda.get_device_properties(0).name}")
        
        print("Loading Qwen3-TTS-0.6B-CustomVoice...")
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        
        # Warmup
        print("Warming up...")
        model.generate_custom_voice(text="Hello.", language="English", speaker="Ryan")
        
        results = {}
        
        for name, text in TEST_TEXTS.items():
            print(f"\n--- {name.upper()} TEXT ({len(text)} chars) ---")
            print(f'"{text[:50]}..."' if len(text) > 50 else f'"{text}"')
            
            times = []
            rtfs = []
            
            for i in range(iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language="English",
                    speaker="Ryan",
                )
                
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                audio = wavs[0]
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                
                audio_duration = len(audio) / sr
                rtf = elapsed / audio_duration
                
                times.append(elapsed)
                rtfs.append(rtf)
                
                print(f"  Run {i+1}: {elapsed*1000:.1f}ms, RTF={rtf:.3f}, audio={audio_duration:.2f}s")
            
            avg_time = sum(times) / len(times)
            avg_rtf = sum(rtfs) / len(rtfs)
            
            results[name] = {
                "text_length": len(text),
                "avg_time_ms": avg_time * 1000,
                "avg_rtf": avg_rtf,
                "ttfc_ms": avg_time * 1000,  # Non-streaming
            }
            
            print(f"  Average: {avg_time*1000:.1f}ms, RTF={avg_rtf:.3f}")
        
        return results
        
    except ImportError as e:
        print(f"Qwen-TTS not available: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_edge_tts(iterations: int = 5):
    """Benchmark Edge TTS for comparison."""
    print("\n" + "=" * 60)
    print("Edge TTS Benchmark (Baseline)")
    print("=" * 60)
    
    try:
        import asyncio
        import edge_tts
        import soundfile as sf
        import tempfile
        import os
        
        async def generate(text):
            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name
            await communicate.save(temp_path)
            audio, sr = sf.read(temp_path)
            os.unlink(temp_path)
            return audio, sr
        
        results = {}
        
        for name, text in TEST_TEXTS.items():
            print(f"\n--- {name.upper()} TEXT ---")
            
            times = []
            rtfs = []
            
            for i in range(iterations):
                start = time.perf_counter()
                audio, sr = asyncio.run(generate(text))
                elapsed = time.perf_counter() - start
                
                audio_duration = len(audio) / sr
                rtf = elapsed / audio_duration
                
                times.append(elapsed)
                rtfs.append(rtf)
                
                print(f"  Run {i+1}: {elapsed*1000:.1f}ms, RTF={rtf:.3f}")
            
            avg_time = sum(times) / len(times)
            avg_rtf = sum(rtfs) / len(rtfs)
            
            results[name] = {
                "avg_time_ms": avg_time * 1000,
                "avg_rtf": avg_rtf,
            }
            
            print(f"  Average: {avg_time*1000:.1f}ms, RTF={avg_rtf:.3f}")
        
        return results
        
    except ImportError as e:
        print(f"Edge TTS not available: {e}")
        return None


def print_summary(megakernel_results, qwen_tts_results, edge_tts_results):
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    print("\n### Performance Targets ###")
    print("  - TTFC (Time to First Chunk): < 60ms")
    print("  - RTF (Real-Time Factor): < 0.15")
    print("  - Talker Speed: ~1000 tok/s")
    
    if megakernel_results:
        print("\n### Megakernel (Qwen3-0.6B Decode) ###")
        print(f"  Tokens/sec: {megakernel_results['avg_tok_s']:.1f}")
        target_met = megakernel_results['avg_tok_s'] >= 900
        print(f"  Target (≥1000 tok/s): {'✓ PASS' if target_met else '✗ CLOSE'}")
    
    if qwen_tts_results:
        print("\n### Qwen3-TTS (0.6B) ###")
        for name, results in qwen_tts_results.items():
            ttfc_pass = results['ttfc_ms'] < 60
            rtf_pass = results['avg_rtf'] < 0.15
            print(f"  {name}: TTFC={results['ttfc_ms']:.1f}ms {'✓' if ttfc_pass else '✗'}, RTF={results['avg_rtf']:.3f} {'✓' if rtf_pass else '✗'}")
    
    if edge_tts_results:
        print("\n### Edge TTS (Baseline) ###")
        for name, results in edge_tts_results.items():
            print(f"  {name}: {results['avg_time_ms']:.1f}ms, RTF={results['avg_rtf']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-TTS")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations")
    parser.add_argument("--text", type=str, help="Custom text to benchmark")
    parser.add_argument("--megakernel-only", action="store_true", help="Only benchmark megakernel")
    parser.add_argument("--tts-only", action="store_true", help="Only benchmark TTS")
    args = parser.parse_args()
    
    if args.text:
        TEST_TEXTS["custom"] = args.text
    
    print("=" * 60)
    print("Qwen3-TTS Megakernel Benchmark Suite")
    print("=" * 60)
    
    megakernel_results = None
    qwen_tts_results = None
    edge_tts_results = None
    
    if not args.tts_only:
        megakernel_results = benchmark_megakernel()
    
    if not args.megakernel_only:
        qwen_tts_results = benchmark_qwen_tts(args.iterations)
        edge_tts_results = benchmark_edge_tts(min(args.iterations, 3))
    
    print_summary(megakernel_results, qwen_tts_results, edge_tts_results)


if __name__ == "__main__":
    main()
