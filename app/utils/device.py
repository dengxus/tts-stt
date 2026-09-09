"""设备探测与精度策略：cuda > mps > cpu，fp16 仅 CUDA。

torch 采用惰性导入：Phase 0 骨架与 api 测试（mock 引擎）无需安装 torch。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _import_torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def torch_version() -> str | None:
    torch = _import_torch()
    return torch.__version__ if torch else None


def detect_device(preference: str = "auto") -> str:
    """返回 "cuda" | "mps" | "cpu"。preference 可强制指定。"""
    torch = _import_torch()
    if torch is None:
        if preference == "auto":
            logger.warning("torch 未安装，设备回退 cpu（仅接口验证可用）")
            return "cpu"
        return "cpu"

    if preference != "auto":
        if preference == "cuda" and not torch.cuda.is_available():
            logger.warning("配置 DEVICE=cuda 但 CUDA 不可用，自动降级探测")
        elif preference == "mps" and not torch.backends.mps.is_available():
            logger.warning("配置 DEVICE=mps 但 MPS 不可用，自动降级探测")
        else:
            return preference

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_blackwell_compat(device: str) -> list[str]:
    """返回兼容性警告（如 sm_120 + 旧 CUDA）。"""
    warnings: list[str] = []
    if device != "cuda":
        return warnings
    torch = _import_torch()
    if torch is None:
        return warnings
    try:
        cap = torch.cuda.get_device_capability(0)
        cuda_ver = tuple(int(x) for x in torch.version.cuda.split(".")[:2])
        if cap >= (12, 0) and cuda_ver < (12, 8):
            warnings.append(
                f"GPU 为 Blackwell (sm_{cap[0]}{cap[1]})，torch 编译于 CUDA "
                f"{torch.version.cuda} <12.8，需 cu128 wheel，"
                "否则将报 no kernel image is available"
            )
    except Exception as e:  # pragma: no cover
        warnings.append(f"CUDA 兼容性检查失败: {e}")
    return warnings


def resolve_fp16(requested: bool, device: str) -> bool:
    """fp16 仅 CUDA；MPS fp16 部分算子不稳，CPU 不支持，均强制 fp32。"""
    enabled = requested and device == "cuda"
    if requested and not enabled:
        logger.info("DEVICE=%s，fp16 自动关闭（使用 fp32）", device)
    return enabled
