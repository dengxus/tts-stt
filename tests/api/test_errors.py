"""错误路径：401/415/422/429/504/OOM-503。"""

from __future__ import annotations

import asyncio

import pytest

from tests.conftest import make_app, make_client
from tests.fake_engines import (
    BrokenTTSEngine,
    FakeSTTEngine,
    HangingTTSEngine,
)

pytestmark = pytest.mark.api


async def test_validation_422():
    async with make_client(make_app()) as client:
        r = await client.post("/api/v1/tts", json={"text": "", "speed": 5})
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["code"] == "validation_error"
        assert "request_id" in err


async def test_unsupported_audio_415():
    async with make_client(make_app()) as client:
        r = await client.post(
            "/api/v1/stt",
            files={"file": ("x.wav", b"not audio, definitely not" * 10)},
        )
        assert r.status_code == 415
        assert r.json()["error"]["code"] == "unsupported_format"


async def test_queue_full_429():
    """GPU 信号量=1 + hang 引擎占住执行权，MAX_QUEUE=2 时第 4 个请求被拒。"""
    app = make_app(
        {"tts": HangingTTSEngine(), "stt": FakeSTTEngine()},
        max_queue=2, request_timeout_s=10,
    )
    async with make_client(app) as client:
        tasks = [asyncio.create_task(
            client.post("/api/v1/tts", json={"text": "卡住"})) for _ in range(4)]
        await asyncio.sleep(0.3)
        for t in tasks:
            t.cancel()
        codes = []
        for t in tasks:
            try:
                codes.append((await t).status_code)
            except asyncio.CancelledError:
                codes.append("cancelled")
        assert 429 in codes
        assert codes.count(200) <= 2  # 最多 1 个成功执行 + 排队者


async def test_inference_timeout_504():
    app = make_app({"tts": HangingTTSEngine(), "stt": FakeSTTEngine()},
                   request_timeout_s=0.3)
    async with make_client(app) as client:
        r = await client.post("/api/v1/tts", json={"text": "超时"})
        assert r.status_code == 504
        assert r.json()["error"]["code"] == "inference_timeout"


async def test_oom_maps_to_503():
    app = make_app({"tts": BrokenTTSEngine(), "stt": FakeSTTEngine()})
    async with make_client(app) as client:
        r = await client.post("/api/v1/tts", json={"text": "oom"})
        assert r.status_code == 503
        assert "OOM" in r.json()["error"]["message"]


async def test_auth_required():
    app = make_app(api_key="secret123")
    async with make_client(app) as client:
        r = await client.post("/api/v1/tts", json={"text": "你好"})
        assert r.status_code in (401, 403)
        r = await client.post("/api/v1/tts", json={"text": "你好"},
                              headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
        r = await client.post("/api/v1/tts", json={"text": "你好"},
                              headers={"Authorization": "Bearer secret123"})
        assert r.status_code == 200
        # 健康探针不鉴权
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/health/live")).status_code == 200


async def test_request_id_echo():
    async with make_client(make_app()) as client:
        r = await client.get("/health", headers={"X-Request-ID": "trace-me-42"})
        assert r.headers["x-request-id"] == "trace-me-42"
