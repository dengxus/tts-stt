from __future__ import annotations

import numpy as np
import pytest

from app.audio.codec import encode
from app.engines.base import PcmAudio
from tests.conftest import make_app, make_client
from tests.fake_engines import FakeSTTEngine, FakeTTSEngine

pytestmark = pytest.mark.api


def sine_wav(seconds: float = 1.0, rate: int = 24000) -> bytes:
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return encode(PcmAudio((np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32),
                           rate), "wav")


def upload(file_bytes: bytes, name: str = "a.wav", **form):
    return {"file": (name, file_bytes, "application/octet-stream"),
            **{k: (None, str(v)) for k, v in form.items()}}


async def test_stt_basic():
    fake = FakeSTTEngine()
    async with make_client(make_app({"tts": FakeTTSEngine(), "stt": fake})) as client:
        r = await client.post("/api/v1/stt", files=upload(sine_wav()))
        assert r.status_code == 200
        body = r.json()
        assert body["text"] == "你好世界"
        assert body["language"] == "zh"
        assert body["duration_ms"] == pytest.approx(1000, abs=30)
        assert body["segments"][0]["text"] == "你好世界"
        assert body["engine"] == "fake-stt"
        # 引擎收到的是 16k 重采样后的数据
        sr, n, lang, itn = fake.last_opts
        assert sr == 16000
        assert abs(n - 16000) < 50
        assert lang == "auto" and itn is True


async def test_stt_mp3_input():
    async with make_client(make_app()) as client:
        mp3 = encode(
            PcmAudio(np.zeros(24000, dtype=np.float32), 24000), "mp3")
        r = await client.post(
            "/api/v1/stt", files=upload(mp3, "song.mp3", language="en")
        )
        assert r.status_code == 200
        assert r.json()["language"] == "zh"  # fake 固定


async def test_stt_params_passthrough():
    fake = FakeSTTEngine()
    async with make_client(make_app({"tts": FakeTTSEngine(), "stt": fake})) as client:
        r = await client.post("/api/v1/stt", files=upload(
            sine_wav(), language="zh", use_itn="false",
            timestamps="false", output_rich="true"))
        assert r.status_code == 200
        body = r.json()
        assert body["segments"] == []       # timestamps=false
        assert body["emotion"] == "HAPPY"   # output_rich=true
        assert body["itn"] is False


async def test_stt_rich_off_hides_emotion():
    async with make_client(make_app()) as client:
        r = await client.post("/api/v1/stt", files=upload(sine_wav()))
        assert r.json()["emotion"] is None


async def test_stt_bad_language():
    async with make_client(make_app()) as client:
        r = await client.post(
            "/api/v1/stt", files=upload(sine_wav(), language="fr"))
        assert r.status_code == 400


async def test_stt_too_long_audio():
    async with make_client(make_app(max_audio_seconds=0.5)) as client:
        r = await client.post("/api/v1/stt", files=upload(sine_wav(2.0)))
        assert r.status_code == 413


async def test_stt_file_too_large():
    async with make_client(make_app(max_audio_bytes=100)) as client:
        r = await client.post("/api/v1/stt", files=upload(sine_wav(1.0)))
        assert r.status_code == 413
