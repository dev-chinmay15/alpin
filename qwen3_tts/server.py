"""
Streaming decode server for Qwen3-TTS Talker.

Exposes a simple interface:
  prompt -> token stream
"""

import asyncio
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .config import config
from .talker import MegakernelTalker

try:
    import torch
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - runtime dependency check
    torch = None
    AutoTokenizer = None


app = FastAPI(title="Qwen3-TTS Decode Server", version="0.1.0")

_talker: Optional[MegakernelTalker] = None
_tokenizer = None


class DecodeRequest(BaseModel):
    prompt: str
    max_tokens: int = 128


def _lazy_init():
    global _talker, _tokenizer
    if _talker is None:
        _talker = MegakernelTalker(model_name=config.model_name, device=config.device, verbose=True)
    if _tokenizer is None and AutoTokenizer is not None:
        _tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)


@app.get("/health")
async def health():
    gpu = bool(torch and torch.cuda.is_available())
    return {"ok": True, "model": config.model_name, "cuda": gpu}


@app.post("/v1/decode")
async def decode_once(req: DecodeRequest):
    """
    Non-streaming decode endpoint for quick tests.
    """
    _lazy_init()
    assert _talker is not None

    input_ids = _tokenizer.encode(req.prompt, add_special_tokens=True) if _tokenizer else [1, 2, 3]
    _talker.reset()
    for tid in input_ids[:-1]:
        _talker.step(tid)

    current = input_ids[-1]
    out = []
    t0 = time.perf_counter()
    for _ in range(req.max_tokens):
        token, _ = _talker.step(current)
        out.append(int(token))
        current = token
    elapsed = (time.perf_counter() - t0) * 1000.0
    return {"tokens": out, "elapsed_ms": elapsed}


@app.websocket("/v1/decode/stream")
async def decode_stream(ws: WebSocket):
    """
    Streaming endpoint.

    Client sends:
      {"prompt": "...", "max_tokens": 128}

    Server emits:
      {"type":"token","token_id":123,"position":0}
      ...
      {"type":"done","elapsed_ms":...}
    """
    await ws.accept()
    try:
        payload = await ws.receive_json()
        prompt = payload.get("prompt", "")
        max_tokens = int(payload.get("max_tokens", 128))

        _lazy_init()
        assert _talker is not None

        input_ids = _tokenizer.encode(prompt, add_special_tokens=True) if _tokenizer else [1, 2, 3]
        _talker.reset()
        for tid in input_ids[:-1]:
            _talker.step(tid)

        current = input_ids[-1]
        t0 = time.perf_counter()
        for pos in range(max_tokens):
            token, _ = _talker.step(current)
            await ws.send_json({"type": "token", "token_id": int(token), "position": pos})
            current = token
            await asyncio.sleep(0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        await ws.send_json({"type": "done", "elapsed_ms": elapsed_ms})
    except WebSocketDisconnect:
        return
    except Exception as e:  # pragma: no cover
        await ws.send_json({"type": "error", "message": str(e)})
    finally:
        await ws.close()
