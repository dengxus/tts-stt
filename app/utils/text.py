"""长文本按标点切句（TTS 分块合成用），中英混排安全。"""

from __future__ import annotations

import re

# 句末标点（含中文全角与换行），保留分隔符在句尾
_SENT_END = re.compile(r"(?<=[。！？；!?;\n])")
# 次级分隔：逗号/顿号/空格
_SOFT_SPLIT = re.compile(r"(?<=[，,、；;：: ])")


def split_text(text: str, max_chars: int = 800) -> list[str]:
    """把文本切成的每块长度 <= max_chars。

    策略：先按句末标点切句 → 贪心合并至接近 max_chars；
    单句仍超长 → 按逗号/空格再切 → 仍超长则硬切（兜底）。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [s for s in _SENT_END.split(text) if s and s.strip()]

    chunks: list[str] = []
    cur = ""
    for sent in sentences:
        while len(sent) > max_chars:  # 超长单句先细分
            piece = _cut_one(sent, max_chars)
            sent = sent[len(piece):]
            if cur and len(cur) + len(piece) <= max_chars:
                cur += piece
            else:
                if cur:
                    chunks.append(cur)
                cur = piece
        if cur and len(cur) + len(sent) > max_chars:
            chunks.append(cur)
            cur = sent
        else:
            cur += sent
    if cur.strip():
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


def _cut_one(s: str, max_chars: int) -> str:
    """从 s 头部切出 <=max_chars 的一段，优先在次级标点处断开。"""
    head = s[: max_chars + 1]
    parts = _SOFT_SPLIT.split(head)
    if len(parts) > 1 and len("".join(parts[:-1])) > max_chars * 0.4:
        return "".join(parts[:-1])
    return s[:max_chars]
