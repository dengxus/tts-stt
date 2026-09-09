"""全局配置，由 .env 或环境变量驱动（pydantic-settings）。"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- 运行时 ---
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    fp16: bool = True                      # 仅 CUDA 生效
    log_level: str = "INFO"
    api_key: str = ""                      # 空 = 不鉴权
    cors_origins: str = "*"

    # --- 引擎 ---
    tts_engine: str = "cosyvoice2"
    stt_engine: str = "sensevoice"
    preload_engines: bool = True
    load_timeout_s: float = 300.0
    default_voice: str = ""

    # --- 模型 ---
    model_dir: Path = Path("models")
    model_source: Literal["modelscope", "hf"] = "modelscope"
    hf_endpoint: str = ""

    # --- 限额 ---
    max_text_chars: int = 5000
    tts_chunk_chars: int = 800
    max_audio_seconds: float = 600.0
    max_audio_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    allow_zero_shot: bool = False

    # --- 并发 ---
    max_concurrent_infer: int = Field(default=1, ge=1)
    max_queue: int = Field(default=8, ge=1)
    request_timeout_s: float = Field(default=120.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
