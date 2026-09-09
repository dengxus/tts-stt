"""API 请求/响应模型（与 docs/api.md、OpenAPI 文档一致）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

AudioFormat = Literal["wav", "mp3", "ogg", "flac"]
ALLOWED_SAMPLE_RATES = {8000, 16000, 22050, 24000, 44100, 48000}


# ---------- TTS ----------

class ZeroShotRefBody(BaseModel):
    prompt_text: str = Field(min_length=1)
    prompt_audio_base64: str = Field(min_length=1, description="≤15s 参考音频 base64")


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: str | None = None
    engine: str | None = None
    format: AudioFormat = "wav"
    sample_rate: int | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    instruct: str | None = None
    zero_shot: ZeroShotRefBody | None = None
    response_format: Literal["audio", "json"] = "audio"

    @field_validator("sample_rate")
    @classmethod
    def _check_sr(cls, v: int | None) -> int | None:
        if v is not None and v not in ALLOWED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate 必须是 {sorted(ALLOWED_SAMPLE_RATES)} 之一，得到 {v}"
            )
        return v

    @field_validator("text")
    @classmethod
    def _check_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text 不能为空白")
        return v


class TTSJsonResponse(BaseModel):
    audio_base64: str
    format: str
    sample_rate: int
    duration_ms: int
    inference_ms: int
    engine: str


# ---------- STT ----------

class STTSegment(BaseModel):
    start_ms: int
    end_ms: int
    text: str


class STTResponse(BaseModel):
    text: str
    language: str
    duration_ms: int
    segments: list[STTSegment] = Field(default_factory=list)
    emotion: str | None = None
    events: list[str] = Field(default_factory=list)
    engine: str
    itn: bool = True
    inference_ms: int = 0
    rtf: float = 0.0


# ---------- 系统 ----------

class VoiceInfoBody(BaseModel):
    id: str
    name: str
    engine: str
    mode: str
    languages: list[str]
    gender: str = ""
    sample_rate: int = 0
    tags: list[str] = Field(default_factory=list)
    default_instruct: str = ""


class VoicesResponse(BaseModel):
    voices: list[VoiceInfoBody]
