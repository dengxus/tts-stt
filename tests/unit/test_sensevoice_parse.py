"""SenseVoice 输出解析纯函数单测（不需要模型/funasr）。"""

from __future__ import annotations

import pytest

from app.engines.stt_sensevoice import (
    SenseVoiceEngine,
    parse_sensevoice_output,
    tags_to_fields,
)

pytestmark = pytest.mark.unit


def test_parse_full_tags():
    text, tags = parse_sensevoice_output(
        "<|zh|><|HAPPY|><|Speech|><|withitn|>今天天气不错，我们去公园吧。")
    assert text == "今天天气不错，我们去公园吧。"
    assert tags == ["zh", "HAPPY", "Speech", "withitn"]
    f = tags_to_fields(tags)
    assert f["language"] == "zh"
    assert f["emotion"] == "HAPPY"
    assert f["events"] == ["Speech"]


def test_parse_no_tags():
    text, tags = parse_sensevoice_output("纯文本没有标签")
    assert text == "纯文本没有标签"
    assert tags == []


def test_parse_auto_lang_and_woitn():
    text, tags = parse_sensevoice_output(
        "<|aut|><|NEUTRAL|><|BGM|><|woitn|>歌词内容")
    f = tags_to_fields(tags)
    assert f["language"] == ""      # aut 视为未指定
    assert f["emotion"] == "NEUTRAL"
    assert f["events"] == ["BGM"]
    assert text == "歌词内容"


def test_parse_mixed_content_with_angle_brackets():
    """正文中的 <...> 不应被误当标签。"""
    text, tags = parse_sensevoice_output("<|en|><|NEUTRAL|>a < b and c")
    assert tags == ["en", "NEUTRAL"]
    assert text == "a < b and c"


def test_cap_long_segments():
    ranges = SenseVoiceEngine._cap([(0, 75_000)])
    assert ranges == [(0, 30_000), (30_000, 60_000), (60_000, 75_000)]


def test_cap_passthrough_short():
    assert SenseVoiceEngine._cap([(100, 2000)]) == [(100, 2000)]
