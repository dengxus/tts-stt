"""健康与系统端点测试（mock 引擎，无需模型）。"""

from __future__ import annotations

import asyncio
import io

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import make_app, make_client
from tests.fake_engines import (
    FailingLoadTTSEngine,
    FakeSTTEngine,
    FakeTTSEngine,
    SlowLoadSTTEngine,
)

pytestmark = pytest.mark.api


def silent_wav(seconds: float = 0.1, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * rate), dtype=np.float32), rate, format="WAV")
    return buf.getvalue()


async def test_health_with_fakes():
    async with make_client(make_app()) as client:
        r = await client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["engines"]["tts"]["state"] == "ready"
        assert body["engines"]["stt"]["state"] == "ready"
        assert body["queue"]["max_queue"] == 8
        assert "x-request-id" in {k.lower() for k in r.headers}


async def test_live_always_ok():
    async with make_client(make_app()) as client:
        assert (await client.get("/health/live")).status_code == 200


async def test_ready_503_while_loading_then_ok():
    app = make_app({"tts": FakeTTSEngine(), "stt": SlowLoadSTTEngine()})
    async with make_client(app) as client:
        r = await client.get("/health/ready")
        assert r.status_code == 503 and r.headers.get("Retry-After") == "5"
        # stt 业务请求在加载中也 503（load_timeout 已含等待）
        # tts 不受影响
        r = await client.post("/api/v1/tts", json={"text": "你好"})
        assert r.status_code == 200
        await asyncio.sleep(2.2)
        r = await client.get("/health/ready")
        assert r.status_code == 200


async def test_load_failure_state_error():
    app = make_app({"tts": FailingLoadTTSEngine(), "stt": FakeSTTEngine()})
    async with make_client(app) as client:
        await asyncio.sleep(0.3)
        r = await client.post("/api/v1/tts", json={"text": "你好"})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "engine_error"
        assert "权重文件缺失" in r.json()["error"]["message"]
        h = (await client.get("/health")).json()
        assert h["engines"]["tts"]["state"] == "error"
        # 进程不崩：其他端点仍工作
        assert (await client.get("/health/live")).status_code == 200


async def test_voices():
    async with make_client(make_app()) as client:
        r = await client.get("/api/v1/voices")
        assert r.status_code == 200
        voices = r.json()["voices"]
        assert voices[0]["id"] == "fake"
        assert voices[0]["sample_rate"] == 24000


async def test_engines_endpoint():
    async with make_client(make_app()) as client:
        r = await client.get("/api/v1/engines")
        assert r.status_code == 200
        assert r.json()["tts"]["name"] == "fake-tts"


async def test_playground_page():
    async with make_client(make_app()) as client:
        r = await client.get("/playground")
        assert r.status_code == 200
        assert "tts-stt 测试台" in r.text
        r = await client.get("/", follow_redirects=False)
        assert r.status_code in (307, 302) and "/playground" in r.headers["location"]


async def test_engine_not_registered_503():
    app = make_app({"tts": FakeTTSEngine()})  # 无 stt
    async with make_client(app) as client:
        r = await client.post(
            "/api/v1/stt", files={"file": ("a.wav", silent_wav())}
        )
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "engine_error"
