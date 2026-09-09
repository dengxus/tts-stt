from __future__ import annotations

import base64

import numpy as np
import pytest

from app.audio.codec import decode_any
from tests.conftest import make_app, make_client
from tests.fake_engines import FakeSTTEngine, FakeTTSEngine

pytestmark = pytest.mark.api


async def test_tts_binary_wav():
    fake = FakeTTSEngine()
    async with make_client(make_app({"tts": fake, "stt": FakeSTTEngine()})) as client:
        r = await client.post("/api/v1/tts", json={"text": "你好世界"})
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/wav"
        assert int(r.headers["X-Chunks"]) == 1
        pcm = decode_any(r.content)
        assert pcm.sample_rate == 24000
        assert pcm.duration_ms == pytest.approx(800, abs=30)  # 4字*0.2s
        rms = float(np.sqrt(np.mean(pcm.samples**2)))
        assert rms > 0.1  # 非静音


@pytest.mark.parametrize("fmt", ["mp3", "ogg", "flac"])
async def test_tts_output_formats(fmt):
    async with make_client(make_app()) as client:
        r = await client.post(
            "/api/v1/tts", json={"text": "测试", "format": fmt}
        )
        assert r.status_code == 200
        pcm = decode_any(r.content)
        assert pcm.duration_ms > 100


async def test_tts_custom_sample_rate():
    async with make_client(make_app()) as client:
        r = await client.post(
            "/api/v1/tts", json={"text": "测试", "sample_rate": 16000}
        )
        assert r.status_code == 200
        assert r.headers["x-sample-rate"] == "16000"
        pcm = decode_any(r.content)
        assert pcm.sample_rate == 16000


async def test_tts_speed_param():
    fake = FakeTTSEngine()
    async with make_client(make_app({"tts": fake, "stt": FakeSTTEngine()})) as client:
        r1 = await client.post("/api/v1/tts", json={"text": "同一句话", "speed": 1.0})
        r2 = await client.post("/api/v1/tts", json={"text": "同一句话", "speed": 2.0})
        d1 = int(r1.headers["X-Duration-Ms"])
        d2 = int(r2.headers["X-Duration-Ms"])
        assert d2 < d1  # 语速 2x 时长约减半


async def test_tts_json_mode():
    async with make_client(make_app()) as client:
        r = await client.post(
            "/api/v1/tts", json={"text": "你好", "response_format": "json"}
        )
        assert r.status_code == 200
        body = r.json()
        audio = base64.b64decode(body["audio_base64"])
        assert decode_any(audio).duration_ms > 100
        assert body["sample_rate"] == 24000


async def test_tts_long_text_chunking():
    """长文本分块：settings.tts_chunk_chars=20，每块 <=20 字。"""
    fake = FakeTTSEngine()
    text = "句。" * 60  # 120 字符，超过 max_text_chars=200? 不，120<200 通过
    async with make_client(
        make_app({"tts": fake, "stt": FakeSTTEngine()})
    ) as client:
        r = await client.post("/api/v1/tts", json={"text": text})
        assert r.status_code == 200
        n_chunks = int(r.headers["X-Chunks"])
        assert n_chunks > 1
        assert all(len(c) <= 20 for c in fake.calls)
        assert "".join(fake.calls).replace(" ", "") == text


async def test_tts_zero_shot_gated_off():
    async with make_client(make_app()) as client:
        r = await client.post("/api/v1/tts", json={
            "text": "你好",
            "zero_shot": {"prompt_text": "参考", "prompt_audio_base64": "AAAA"},
        })
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_request"


async def test_tts_over_max_text_chars():
    async with make_client(make_app()) as client:
        r = await client.post("/api/v1/tts", json={"text": "字" * 201})
        assert r.status_code == 400


async def test_tts_passes_options_to_engine():
    fake = FakeTTSEngine()
    async with make_client(make_app({"tts": fake, "stt": FakeSTTEngine()})) as client:
        r = await client.post("/api/v1/tts", json={
            "text": "你好", "voice": "zh_female_default", "speed": 1.2,
            "instruct": "用开心的语气说",
        })
        assert r.status_code == 200
        opts = fake.opts_seen[0]
        assert opts.voice == "zh_female_default"
        assert opts.speed == 1.2
        assert opts.instruct == "用开心的语气说"
