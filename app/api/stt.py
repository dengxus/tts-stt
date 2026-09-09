"""POST /api/v1/stt —— 语音转文本（multipart 上传）。"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.api.deps import get_manager, get_settings_dep, require_api_key
from app.audio.codec import decode_any
from app.config import Settings
from app.engines.base import STTOptions
from app.engines.manager import EngineManager
from app.errors import InvalidRequestError, PayloadTooLargeError
from app.schemas import STTResponse, STTSegment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["stt"],
                   dependencies=[Depends(require_api_key)])

SUPPORTED_LANGUAGES = {"auto", "zh", "en", "ja", "ko", "yue"}


@router.post("/stt", response_model=STTResponse)
async def stt(
    request: Request,
    file: UploadFile = File(..., description="wav/mp3/m4a/ogg/opus/flac/webm"),
    language: str = Form("auto"),
    use_itn: bool = Form(True),
    timestamps: bool = Form(True),
    output_rich: bool = Form(False),
    manager: EngineManager = Depends(get_manager),
    settings: Settings = Depends(get_settings_dep),
):
    if language not in SUPPORTED_LANGUAGES:
        raise InvalidRequestError(
            f"language 必须是 {sorted(SUPPORTED_LANGUAGES)} 之一，得到 {language!r}"
        )
    data = await file.read()
    if not data:
        raise InvalidRequestError("上传文件为空")
    if len(data) > settings.max_audio_bytes:
        raise PayloadTooLargeError(
            f"文件超过上限 {settings.max_audio_bytes} 字节（当前 {len(data)}）"
        )

    engine = await manager.get_ready("stt")

    # 解码 + 重采样是 CPU 段，放线程但不占 GPU 守卫
    pcm = await asyncio.to_thread(decode_any, data, engine.required_sample_rate)
    if pcm.duration_ms / 1000 > settings.max_audio_seconds:
        raise PayloadTooLargeError(
            f"音频时长 {pcm.duration_ms / 1000:.0f}s 超过上限 "
            f"{settings.max_audio_seconds:.0f}s"
        )

    opts = STTOptions(language=language, use_itn=use_itn,
                      timestamps=timestamps, output_rich=output_rich)
    t0 = time.perf_counter()
    transcript = await manager.run("stt", lambda eng: eng.transcribe(pcm, opts))
    inference_ms = int((time.perf_counter() - t0) * 1000)

    segments = (
        [STTSegment(start_ms=s.start_ms, end_ms=s.end_ms, text=s.text)
         for s in transcript.segments]
        if timestamps else []
    )
    return STTResponse(
        text=transcript.text,
        language=transcript.language,
        duration_ms=pcm.duration_ms,
        segments=segments,
        emotion=transcript.emotion if output_rich else None,
        events=transcript.events if output_rich else [],
        engine=engine.name,
        itn=use_itn,
        inference_ms=inference_ms,
        rtf=round(inference_ms / max(pcm.duration_ms, 1), 3),
    )
