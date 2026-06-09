"""
Local test script - works without GPU.
Tests that the project structure and imports work correctly.

For full testing, install: pip install torch pipecat-ai
For lightweight local test, just needs: pip install python-dotenv numpy
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test imports - skip heavy dependencies if not installed."""
    print("Testing imports...")
    
    # Config should always work
    from qwen3_tts.config import config, TTSConfig
    print("  ✓ config module")
    
    # Check for optional heavy dependencies
    torch_available = False
    pipecat_available = False
    
    try:
        import torch
        torch_available = True
        print("  ✓ torch available")
    except ImportError:
        print("  ⚠ torch not installed (optional for local test)")
    
    try:
        import pipecat
        pipecat_available = True
        print("  ✓ pipecat available")
    except ImportError:
        print("  ⚠ pipecat not installed (optional for local test)")
    
    if torch_available:
        from qwen3_tts.talker import MegakernelTalker
        print("  ✓ talker module")
        
        from qwen3_tts.code_predictor import CodePredictor
        print("  ✓ code_predictor module")
        
        from qwen3_tts.speech_decoder import SpeechDecoder
        print("  ✓ speech_decoder module")
        
        from qwen3_tts.engine import Qwen3TTSEngine
        print("  ✓ engine module")
    else:
        print("  ⏭ Skipping torch-dependent modules")
    
    if pipecat_available:
        from qwen3_tts.pipecat_service import Qwen3TTSService, Qwen3TTSServiceMock
        print("  ✓ pipecat_service module")
    else:
        print("  ⏭ Skipping pipecat-dependent modules")
    
    print("\nImport test complete!")


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    from qwen3_tts.config import config
    
    print(f"  Model: {config.model_name}")
    print(f"  Sample rate: {config.sample_rate}")
    print(f"  Frame rate: {config.frame_rate} Hz")
    print(f"  Samples per frame: {config.samples_per_frame}")
    print(f"  Use megakernel: {config.use_megakernel}")
    
    print("\nConfiguration OK!")


def test_env():
    """Test environment variables."""
    print("\nTesting environment...")
    
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        print(f"  GOOGLE_API_KEY: {google_key[:10]}...")
    else:
        print("  GOOGLE_API_KEY: Not set")
    
    print("\nEnvironment OK!")


def test_mock_tts():
    """Test mock TTS generation."""
    print("\nTesting mock TTS...")
    
    try:
        import pipecat
    except ImportError:
        print("  ⏭ Skipping (pipecat not installed)")
        return
    
    import asyncio
    from qwen3_tts.pipecat_service import Qwen3TTSServiceMock
    
    async def run_test():
        tts = Qwen3TTSServiceMock()
        
        chunks = []
        async for frame in tts.run_tts("Hello, this is a test."):
            if hasattr(frame, 'audio'):
                chunks.append(frame.audio)
        
        total_bytes = sum(len(c) for c in chunks)
        print(f"  Generated {len(chunks)} chunks, {total_bytes} bytes")
    
    asyncio.run(run_test())
    print("\nMock TTS OK!")


def main():
    print("=" * 50)
    print("Qwen3-TTS Local Tests")
    print("=" * 50)
    
    try:
        test_imports()
        test_config()
        test_env()
        test_mock_tts()
        
        print("\n" + "=" * 50)
        print("Local tests passed! ✓")
        print("=" * 50)
        print("\nProject structure is ready.")
        print("\nFor full testing on GPU machine:")
        print("  1. Rent RTX 5090 on Vast.ai")
        print("  2. Clone repo to GPU machine")
        print("  3. Run: pip install -r requirements.txt")
        print("  4. Run: bash scripts/setup_gpu.sh")
        print("  5. Run: python -m demo.benchmark")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
