"""FastAPI 入口。

- `create_app(engines=...)` 工厂支持测试注入假引擎（不装 torch 也能跑 api 测试）。
- lifespan 中按配置后台预加载引擎，加载中 /health 如实上报、业务端点返回 503。
- 生产启动：`uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1`
  （必须单 worker：模型驻留进程内，多 worker 会双份显存）。
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from app import __version__
from app.api import stt as stt_api
from app.api import system as system_api
from app.api import tts as tts_api
from app.config import Settings, get_settings
from app.engines.base import STTEngine, TTSEngine
from app.engines.manager import EngineManager
from app.errors import register_error_handlers

logger = logging.getLogger("app")

# macOS 开发：CosyVoice 个别算子未移植 MPS，自动回落 CPU，保证跑通
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_ENGINE_REGISTRY = {
    "sensevoice": ("app.engines.stt_sensevoice", "SenseVoiceEngine", "stt"),
    "cosyvoice2": ("app.engines.tts_cosyvoice2", "CosyVoice2Engine", "tts"),
}


def _register_engines(
    manager: EngineManager,
    settings: Settings,
    engines: dict[str, TTSEngine | STTEngine] | None,
) -> None:
    if engines:
        for kind, engine in engines.items():
            manager.register(kind, engine)
        return

    import importlib

    for kind, name in (("stt", settings.stt_engine), ("tts", settings.tts_engine)):
        entry = _ENGINE_REGISTRY.get(name)
        if entry is None:
            logger.warning("未知 %s 引擎 %r，跳过注册", kind, name)
            continue
        module_name, class_name, _kind = entry
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
            manager.register(kind, cls(settings))
        except ImportError as e:
            logger.warning("引擎 %s 依赖未安装（%s），该能力返回 503", name, e)


def create_app(
    settings: Settings | None = None,
    engines: dict[str, TTSEngine | STTEngine] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = EngineManager(settings)
        app.state.manager = manager
        _register_engines(manager, settings, engines)
        if settings.preload_engines:
            manager.start_background_load()
        logger.info("tts-stt v%s 已启动（引擎后台加载中）", __version__)
        yield
        await manager.shutdown()

    app = FastAPI(
        title="tts-stt 基础服务",
        description="开源文本转语音 (CosyVoice2) / 语音转文本 (SenseVoice) REST API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    register_error_handlers(app)
    app.include_router(system_api.router)
    app.include_router(tts_api.router)
    app.include_router(stt_api.router)

    # 浏览器测试台（合成试听 + 上传/麦克风识别）；Swagger 在 /docs
    web_dir = Path(__file__).resolve().parent.parent / "web"

    @app.get("/playground", include_in_schema=False)
    async def playground():
        return FileResponse(web_dir / "playground.html")

    @app.get("/", include_in_schema=False)
    async def index():
        return RedirectResponse("/playground")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)
