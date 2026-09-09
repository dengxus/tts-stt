from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.fake_engines import FakeSTTEngine, FakeTTSEngine  # noqa: E402


def make_settings(**overrides) -> Settings:
    """构造隔离配置的 Settings（不读 .env，避免本机环境变量干扰测试）。"""
    base = dict(
        _env_file=None,
        preload_engines=True,
        max_queue=8,
        max_concurrent_infer=1,
        request_timeout_s=10.0,
        max_text_chars=200,
        tts_chunk_chars=20,
    )
    base.update(overrides)
    return Settings(**base)


def make_app(engines: dict | None = None, **settings_overrides):
    if engines is None:
        engines = {"tts": FakeTTSEngine(), "stt": FakeSTTEngine()}
    settings = make_settings(**settings_overrides)
    return create_app(settings=settings, engines=engines)


@pytest.fixture
def app_with_fakes():
    return make_app()


@pytest.fixture
def tts_engine():
    return FakeTTSEngine()


@pytest.fixture
def stt_engine():
    return FakeSTTEngine()


@asynccontextmanager
async def make_client(app):
    """进入 lifespan（触发预加载）并返回 httpx AsyncClient。

    用法: async with make_client(app) as client: ...
    """
    from asgi_lifespan import LifespanManager
    from httpx import ASGITransport, AsyncClient

    stack = LifespanManager(app)
    await stack.__aenter__()
    try:
        client = AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test")
        try:
            yield client
        finally:
            await client.__aexit__(None, None, None)
    finally:
        await stack.__aexit__(None, None, None)
