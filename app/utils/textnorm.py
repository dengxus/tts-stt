"""轻量中文文本正则化（TN）兜底。

仅在 wetext（CosyVoice 前端 TN）缺失时启用（见 engines/tts_cosyvoice2._prenormalize）。
覆盖高频场景：整数/小数读法、百分比、年份、手机号/长数字串。
不追求完备——完备方案是装 wetext（pip install wetext，win 亦可）。
"""

from __future__ import annotations

import re

_DIGITS = "零一二三四五六七八九"
_UNITS = ["", "十", "百", "千"]
_GROUPS = ["", "万", "亿", "万亿"]


def _int_to_zh(n: int) -> str:
    """0 <= n < 10**12 的中文读法（口语习惯：十X 省略前导一，低位组不足四位补零）。"""
    if n == 0:
        return "零"
    parts: list[tuple[int, int]] = []
    gi = 0
    while n > 0:
        parts.append((gi, n % 10000))
        n //= 10000
        gi += 1
    parts.reverse()
    out = ""
    for idx, (group_i, rem) in enumerate(parts):
        if rem == 0:
            if out and idx < len(parts) - 1 and not out.endswith("零"):
                out += "零"
            continue
        s = _four_to_zh(rem)
        if idx > 0 and rem < 1000 and not out.endswith("零"):
            s = "零" + s  # 一万零三百 / 一万零一
        out += s + _GROUPS[group_i]
    if out.startswith("一十"):
        out = out[1:]
    return out


def _four_to_zh(x: int) -> str:
    s = ""
    zero_pending = False
    for i in range(3, -1, -1):
        d = x // 10**i % 10
        if d == 0:
            if s:
                zero_pending = True
        else:
            if zero_pending:
                s += "零"
                zero_pending = False
            s += _DIGITS[d] + _UNITS[i]
    return s


def _decimals_to_zh(s: str) -> str:
    return "".join(_DIGITS[int(c)] for c in s)


def _num_to_zh(num_str: str) -> str:
    num_str = num_str.replace(",", "").replace("，", "")
    if "." in num_str:
        ip, dp = num_str.split(".", 1)
        return f"{_int_to_zh(int(ip))}点{_decimals_to_zh(dp)}"
    try:
        v = int(num_str)
    except ValueError:
        return num_str
    if len(num_str) >= 7:  # 长数字串（手机号/编号）按位读
        return _decimals_to_zh(num_str)
    if 10**12 <= v:
        return _decimals_to_zh(num_str)
    return _int_to_zh(v)


_PERCENT = re.compile(r"(\d[\d,，]*(?:\.\d+)?)\s*[%％]")
_YEAR = re.compile(r"(?<!\d)((?:1[89]|20)\d{2})(?=年)")
# 前后紧跟字母/数字/小数点/逗号的片段（v2.1、GPT4、1,2,3 场景）不处理，宁漏勿错
_NUMBER = re.compile(r"(?<![A-Za-z\d.,%])(\d[\d,，]*(?:\.\d+)?)(?![\d%.A-Za-z])")


def normalize(text: str) -> str:
    """保守替换：只在含中文的文本上做数字化学读法。"""
    if not any("一" <= c <= "鿿" for c in text):
        return text
    text = _PERCENT.sub(lambda m: f"百分之{_num_to_zh(m.group(1))}", text)
    text = _YEAR.sub(lambda m: _decimals_to_zh(m.group(1)), text)
    text = _NUMBER.sub(lambda m: _num_to_zh(m.group(1)), text)
    return text
