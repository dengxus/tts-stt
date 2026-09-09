from __future__ import annotations

import pytest

from app.utils.textnorm import normalize

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("num,expect", [
    ("123456", "十二万三千四百五十六"),
    ("10001", "一万零一"),
    ("100", "一百"),
    ("10500", "一万零五百"),
    ("3", "三"),
    ("10.5", "十点五"),
])
def test_int_reading(num, expect):
    assert normalize(f"共有{num}人。") == f"共有{expect}人。"


def test_percent():
    assert normalize("增长了10.5%。") == "增长了百分之十点五。"
    assert normalize("合格率100%") == "合格率百分之一百"


def test_year():
    assert normalize("2024年是龙年。") == "二零二四年是龙年。"


def test_phone_long_digits_spelled():
    assert normalize("拨打13800138000联系") == "拨打一三八零零一三八零零零联系"


def test_pure_english_untouched():
    assert normalize("There were 1,234 users in 2024.") == \
        "There were 1,234 users in 2024."


def test_mixed_cn_en_keeps_version_like():
    """数字紧跟英文/单位不误伤（如 v2.1、GPT4 场景保守跳过）。"""
    assert normalize("模型是 v2.1。") == "模型是 v2.1。"


def test_no_chinese_short_circuit():
    assert normalize("hello 123") == "hello 123"
