"""POST /api/v1/tts —— 文本转语音。"""

from __future__ import annotations

import asyncio
import base64
import logging
import time

import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from app.api.deps import get_manager, get_settings_dep, require_api_key
from app.audio.codec import CONTENT_TYPES, decode_any, encode
from app.config import Settings
from app.engines.base import TTSOptions, ZeroShotRef
from app.engines.manager import EngineManager
from app.errors import InvalidRequestError
from app.schemas import TTSJsonResponse, TTSRequest
from app.utils.text import split_text

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["tts"],
                   dependencies=[Depends(require_api_key)])

# 分块之间插入的静音时长（秒）
_CHUNK_GAP_S = 0.08


@router.post("/tts")
async def tts(
    req: TTSRequest,
    request: Request,
    manager: EngineManager = Depends(get_manager),
    settings: Settings = Depends(get_settings_dep),
):
    if len(req.text) > settings.max_text_chars:
        raise InvalidRequestError(
            f"text 超过上限 {settings.max_text_chars} 字符（当前 {len(req.text)}）"
        )
    if req.zero_shot and not settings.allow_zero_shot:
        raise InvalidRequestError("zero-shot 声音克隆未开放（ALLOW_ZERO_SHOT=false）")

    engine = await manager.get_ready("tts")

    # zero-shot 参考音频解码（CPU 段，不占 GPU 守卫）
    zero_shot = None
    if req.zero_shot:
        raw = base64.b64decode(req.zero_shot.prompt_audio_base64)
        ref = await asyncio.to_thread(decode_any, raw)
        zero_shot = ZeroShotRef(prompt_text=req.zero_shot.prompt_text,
                                prompt_audio=ref)

    chunks = split_text(req.text, settings.tts_chunk_chars)
    opts = TTSOptions(
        voice=req.voice or settings.default_voice or None,
        speed=req.speed,
        seed=req.seed,
        instruct=req.instruct,
        zero_shot=zero_shot,
    )

    t0 = time.perf_counter()

    def _work(eng):
        parts: list[np.ndarray] = []
        sr = eng.native_sample_rate
        gap = np.zeros(int(sr * _CHUNK_GAP_S), dtype=np.float32)
        for i, chunk in enumerate(chunks):
            pcm = eng.synthesize(chunk, opts)
            if i:
                parts.append(gap)
            parts.append(pcm.samples.astype(np.float32))
        return np.concatenate(parts), sr

    samples, sr = await manager.run("tts", _work)
    inference_ms = int((time.perf_counter() - t0) * 1000)

    from app.engines.base import PcmAudio

    pcm = PcmAudio(samples, sr)
    out_sr = req.sample_rate or sr
    audio_bytes = await asyncio.to_thread(encode, pcm, req.format, out_sr)

    duration_ms = pcm.duration_ms  # 重采样不改变时长
    rtf = round(inference_ms / max(duration_ms, 1), 3)
    headers = {
        "X-Duration-Ms": str(duration_ms),
        "X-Sample-Rate": str(out_sr),
        "X-Inference-Ms": str(inference_ms),
        "X-RTF": str(rtf),
        "X-Engine": engine.name,
        "X-Device": manager.device,
        "X-Chunks": str(len(chunks)),
        "Content-Disposition": f'inline; filename="tts.{req.format}"',
    }

    if req.response_format == "json":
        return TTSJsonResponse(
            audio_base64=base64.b64encode(audio_bytes).decode(),
            format=req.format,
            sample_rate=out_sr,
            duration_ms=duration_ms,
            inference_ms=inference_ms,
            engine=engine.name,
        )
    return Response(content=audio_bytes, media_type=CONTENT_TYPES[req.format],
                    headers=headers)
