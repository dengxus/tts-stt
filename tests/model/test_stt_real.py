"""真实 SenseVoice 权重测试（无权重自动 skip）。

使用 SenseVoiceSmall 仓库自带的 example 音频（下载模型时一并获取）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.audio.codec import decode_any
from app.config import Settings
from app.engines.base import PcmAudio, STTOptions
from app.models_spec import STT_SPECS, is_downloaded

pytestmark = pytest.mark.model

_SETTINGS = Settings(_env_file=None)
_MODEL_DIR = Path(_SETTINGS.model_dir)
_EXAMPLE_DIR = _MODEL_DIR / "SenseVoiceSmall" / "example"

pytest.importorskip("funasr", reason="funasr 未安装")

if not is_downloaded(_MODEL_DIR, STT_SPECS[0]):
    pytest.skip("SenseVoice 权重未下载（运行 scripts/download_models.py --only stt）",
                allow_module_level=True)


@pytest.fixture(scope="module")
def engine():
    from app.engines.stt_sensevoice import SenseVoiceEngine

    eng = SenseVoiceEngine(_SETTINGS)
    import asyncio

    asyncio.run(eng.load("cpu", False))
    return eng


def _example(name: str) -> Path | None:
    p = _EXAMPLE_DIR / name
    return p if p.exists() else None


@pytest.mark.parametrize("sample,expect_lang", [
    ("zh.mp3", "zh"),
    ("en.mp3", "en"),
])
def test_transcribe_examples(engine, sample, expect_lang):
    path = _example(sample)
    if path is None:
        pytest.skip(f"example 音频不存在: {path}")
    audio = decode_any(path.read_bytes(), target_rate=16000)
    t = engine.transcribe(audio, STTOptions())
    assert len(t.text) >= 3
    assert t.language == expect_lang, f"实际文本: {t.text}"
    # 时间戳递增且合法
    prev_end = 0
    for seg in t.segments:
        assert seg.start_ms <= seg.end_ms
        assert seg.start_ms >= prev_end - 1
        assert seg.text
        prev_end = seg.end_ms


def test_silence_returns_empty(engine):
    audio = PcmAudio(np.zeros(16000 * 8, dtype=np.float32), 16000)  # 8s 触发 VAD 路径
    t = engine.transcribe(audio, STTOptions())
    assert t.text == ""
    assert t.segments == []


def test_force_language_hint(engine):
    path = _example("zh.mp3") or _example("en.mp3")
    if path is None:
        pytest.skip("无 example 音频")
    audio = decode_any(path.read_bytes(), target_rate=16000)
    t = engine.transcribe(audio, STTOptions(language="zh", output_rich=True))
    assert t.language == "zh"
    # rich 字段类型正确
    assert t.emotion is None or isinstance(t.emotion, str)
    assert isinstance(t.events, list)
