from __future__ import annotations

import pytest

from app.utils.text import split_text

pytestmark = pytest.mark.unit


def test_short_text_single_chunk():
    assert split_text("你好，世界。", 800) == ["你好，世界。"]


def test_empty():
    assert split_text("   ", 800) == []


def test_split_by_sentence_boundary():
    text = "第一句话。" * 30  # 180 字符
    chunks = split_text(text, max_chars=50)
    assert all(len(c) <= 50 for c in chunks)
    assert "".join(chunks) == text


def test_long_sentence_without_punctuation_hardcut():
    text = "啊" * 200
    chunks = split_text(text, max_chars=30)
    assert all(len(c) <= 30 for c in chunks)
    assert "".join(chunks) == text


def test_mixed_cn_en():
    text = "Hello world! 你好世界。This is a test. 这是测试。" + "结束。" * 40
    chunks = split_text(text, max_chars=40)
    assert all(len(c) <= 40 for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_greedy_merge_keeps_sentences_together():
    text = "短句一。短句二。短句三。"
    chunks = split_text(text, max_chars=800)
    assert chunks == [text]
