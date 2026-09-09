"""系统端点：/health*、/api/v1/voices、/api/v1/engines。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.deps import get_manager, require_api_key
from app.engines.base import EngineState, TTSEngine
from app.engines.manager import EngineManager
from app.schemas import VoiceInfoBody, VoicesResponse

router = APIRouter()


@router.get("/health", tags=["system"])
async def health(manager: EngineManager = Depends(get_manager)) -> dict:
    """运维详情：设备、各引擎状态、队列深度。"""
    return manager.status()


@router.get("/health/live", tags=["system"])
async def live():
    """进程活着即 200（模型状态不影响存活探针）。"""
    return {"status": "alive"}


@router.get("/health/ready", tags=["system"])
async def ready(manager: EngineManager = Depends(get_manager)):
    """预加载引擎全部 ready 才 200，供服务管理器/负载均衡探活。"""
    if manager.ready_for_readiness():
        return {"status": "ready"}
    return JSONResponse(
        status_code=503,
        content={"error": {"code": "model_loading",
                           "message": "引擎尚未全部就绪",
                           "engines": {k: v["state"]
                                       for k, v in manager.status()["engines"].items()}}},
        headers={"Retry-After": "5"},
    )


@router.get("/api/v1/voices", response_model=VoicesResponse, tags=["system"],
            dependencies=[Depends(require_api_key)])
async def voices(manager: EngineManager = Depends(get_manager)):
    engine = await manager.get_ready("tts")
    assert isinstance(engine, TTSEngine)
    voices_ = [
        VoiceInfoBody(
            id=v.id, name=v.name, engine=v.engine, mode=v.mode,
            languages=v.languages, gender=v.gender,
            sample_rate=v.sample_rate or engine.native_sample_rate,
            tags=v.tags, default_instruct=v.default_instruct,
        )
        for v in engine.list_voices()
    ]
    return VoicesResponse(voices=voices_)


@router.get("/api/v1/engines", tags=["system"],
            dependencies=[Depends(require_api_key)])
async def engines(manager: EngineManager = Depends(get_manager)):
    status = manager.status()
    return {
        "device": status["device"],
        "tts": _engine_caps(status["engines"].get("tts")),
        "stt": _engine_caps(status["engines"].get("stt")),
    }


def _engine_caps(info: dict | None) -> dict:
    if info is None:
        return {"name": None, "state": EngineState.UNLOADED.value}
    return info
