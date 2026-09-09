#!/usr/bin/env python
"""生成测试音频 fixtures：

1. tests/fixtures/sine_*.{wav,mp3,ogg,flac,m4a} —— 合成正弦波（编解码矩阵/接口测试）
2. tests/fixtures/real_*.{wav,mp3} —— 从已下载的 SenseVoice example 复制真实语音
   （模型测试用；未下载模型时跳过）
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.audio.codec import encode  # noqa: E402
from app.engines.base import PcmAudio  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


def sine(seconds: float, sr: int, freq: float = 440.0) -> PcmAudio:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return PcmAudio((np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32), sr)


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    n = 0
    for fmt in ("wav", "mp3", "ogg", "flac"):
        p = FIXTURES / f"sine_1s.{fmt}"
        p.write_bytes(encode(sine(1.0, 24000), fmt))
        n += 1
    p = FIXTURES / "sine_1s_16k.wav"
    p.write_bytes(encode(sine(1.0, 16000), "wav"))
    n += 1

    examples = ROOT / "models" / "SenseVoiceSmall" / "example"
    if examples.is_dir():
        for name, lang in (("zh.mp3", "real_zh"), ("en.mp3", "real_en")):
            src = examples / name
            if src.exists():
                shutil.copy(src, FIXTURES / f"{lang}.mp3")
                n += 1
        print("[ok] 含真实语音 example 音频")
    else:
        print("[skip] 未找到 SenseVoice example（先运行 download_models.py --only stt）")

    print(f"[PASS] 生成 {n} 个 fixture -> {FIXTURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
