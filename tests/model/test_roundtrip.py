"""质量门禁：TTS → STT 回环一致性（两侧任一退化都会被抓到）。

需 TTS+STT 权重齐备；自动 skip。语料 10 句（中 7 + 英 3）。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from app.audio.codec import resample
from app.config import Settings
from app.engines.base import PcmAudio, STTOptions, TTSOptions
from app.models_spec import ALL_SPECS, is_downloaded

pytestmark = pytest.mark.model

_SETTINGS = Settings(_env_file=None)
_ROOT = Path(_SETTINGS.model_dir)

pytest.importorskip("funasr")
if not all(is_downloaded(_ROOT, s) for s in ALL_SPECS):
    pytest.skip("TTS/STT 权重未齐备", allow_module_level=True)

ZH_CORPUS = [
    "今天天气不错，我们去公园走走吧。",
    "请输入正确的手机号码。",
    "他的生日是二零二四年三月十五日。",
    "欢迎使用语音合成服务。",
    "请帮我预订明天上午十点去上海的机票。",
    "这道菜的味道非常鲜美。",
    "学而不思则罔，思而不学则殆。",
]
EN_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, how are you doing today?",
    "Please set an alarm for seven thirty in the morning.",
]


@pytest.fixture(scope="module")
def engines():
    from app.engines.stt_sensevoice import SenseVoiceEngine
    from app.engines.tts_cosyvoice2 import CosyVoice2Engine

    tts = CosyVoice2Engine(_SETTINGS)
    stt = SenseVoiceEngine(_SETTINGS)
    asyncio.run(tts.load("cpu", False))
    asyncio.run(stt.load("cpu", False))
    return tts, stt


_PUNCT = re.compile(r"[。，、；：！？“”‘’‘’（）《》,.!?;:'\"()\[\]]+")


def _norm_zh(s: str) -> str:
    """去空白与标点（TTS/STT 回环对标点位置不敏感）。"""
    return re.sub(r"\s+", "", _PUNCT.sub("", s))


def _cer(ref: str, hyp: str) -> float:
    from jiwer import cer

    return cer(_norm_zh(ref), _norm_zh(hyp))


def _wer(ref: str, hyp: str) -> float:
    from jiwer import wer

    return wer(_PUNCT.sub(" ", ref).lower(), _PUNCT.sub(" ", hyp).lower())


# 阈值说明：中文同音字（预订/预定等）是 ASR 固有混淆，CER 阈值按业内惯例放宽到 0.12；
# 回环比较统一 use_itn=False，避免 ITN 把"二零二四"变"2024"造成口径错位。
ZH_CER_LIMIT = 0.12
EN_WER_LIMIT = 0.15


@pytest.mark.parametrize("text", ZH_CORPUS)
def test_roundtrip_zh(engines, text):
    tts, stt = engines
    pcm = tts.synthesize(text, TTSOptions())
    assert isinstance(pcm, PcmAudio) and pcm.duration_ms > 300
    back = stt.transcribe(
        resample(pcm, 16000), STTOptions(language="zh", use_itn=False))
    score = _cer(text, back.text)
    assert score < ZH_CER_LIMIT, f"CER={score:.2%} 合成->{back.text}"


@pytest.mark.parametrize("text", EN_CORPUS)
def test_roundtrip_en(engines, text):
    tts, stt = engines
    pcm = tts.synthesize(text, TTSOptions())
    back = stt.transcribe(
        resample(pcm, 16000), STTOptions(language="en", use_itn=False))
    score = _wer(text, back.text)
    assert score < EN_WER_LIMIT, f"WER={score:.2%} 合成->{back.text}"
