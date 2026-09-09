#!/usr/bin/env python
"""双平台环境自检：Python/torch/设备/音频库/模型目录。

Windows 生产机验收要点：输出必须包含
    capability=(12, 0)  且  arch_list 含 sm_120  且  torch.version.cuda=12.8
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, object]] = []


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    ok = True
    _section("系统")
    print(f"platform      : {platform.platform()}")
    print(f"python        : {sys.version.split()[0]} ({sys.executable})")
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        print("[WARN] 建议 Python 3.10~3.12（生产机统一 3.10）")

    _section("numpy / 音频库")
    try:
        import numpy as np

        print(f"numpy         : {np.__version__}")
        if not (1, 26) <= tuple(map(int, np.__version__.split(".")[:2])) < (2, 1):
            print("[WARN] numpy 需 >=1.26,<2.1（numba/whisper ABI 兼容区间）")
            ok = False
    except ImportError as e:
        print(f"[FAIL] numpy: {e}")
        ok = False
    for mod, name in (("av", "PyAV"), ("soundfile", "soundfile"), ("soxr", "soxr")):
        try:
            m = __import__(mod)
            print(f"{name:<13}: {getattr(m, '__version__', '?')}")
        except ImportError as e:
            print(f"[FAIL] {name}: {e}")
            ok = False

    _section("torch / GPU")
    try:
        import torch
    except ImportError as e:
        print(f"[FAIL] torch 未安装: {e}")
        return 1
    print(f"torch         : {torch.__version__}")
    print(f"cuda (torch)  : {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device name   : {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"capability    : {cap}")
        print(f"arch_list     : {torch.cuda.get_arch_list()}")
        archs = torch.cuda.get_arch_list()
        if cap >= (12, 0):
            if not any("120" in a for a in archs):
                print("[FAIL] Blackwell GPU 但 arch_list 无 sm_120 —— 必须装 cu128 wheel")
                ok = False
            if (torch.version.cuda or "0") < "12.8":
                print("[FAIL] torch CUDA 版本 < 12.8，Blackwell 无法运行")
                ok = False
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM          : {vram:.1f} GB")
        if vram < 7.5:
            print("[WARN] 显存 <7.5GB，双模型常驻可能吃紧（见部署文档降级策略）")
    mps = getattr(torch.backends, "mps", None)
    print(f"mps available : {bool(mps and mps.is_available())}")

    _section("推理引擎依赖（未安装只提示，不判失败）")
    for mod, name, phase in (
        ("funasr", "funasr/STT", "Phase 2"),
        ("modelscope", "modelscope", "Phase 2"),
        ("hyperpyyaml", "hyperpyyaml/CosyVoice", "Phase 3"),
    ):
        try:
            __import__(mod)
            print(f"OK   {name}")
        except ImportError:
            print(f"--   {name} 未安装（{phase} 前安装即可）")

    _section("设备选择结果（应用层逻辑）")
    from app.utils.device import detect_device, resolve_fp16

    dev = detect_device("auto")
    print(f"detect_device : {dev}")
    print(f"fp16          : {resolve_fp16(True, dev)}（仅 cuda 为 True）")

    _section("模型目录")
    models_dir = ROOT / "models"
    if models_dir.exists():
        for p in sorted(models_dir.iterdir()):
            if p.is_dir():
                n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                print(f"{p.name:<45}: {n / 1024**2:.0f} MB")
    else:
        print("models/ 不存在（用 scripts/download_models.py 下载）")

    print("\n" + ("[PASS] 环境检查通过" if ok else "[WARN] 存在问题，见上文"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
