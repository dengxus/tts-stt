from __future__ import annotations

import types

import pytest

from app.utils import device as dev_mod
from app.utils.device import detect_device, resolve_fp16

pytestmark = pytest.mark.unit


def fake_torch(cuda: bool, mps: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        __version__="2.7.1",
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda,
            get_device_capability=lambda i: (12, 0),
            get_device_name=lambda i: "test",
        ),
        version=types.SimpleNamespace(cuda="12.8"),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)
        ),
    )


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [(True, True, "cuda"), (True, False, "cuda"),
     (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_detection(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(dev_mod, "_import_torch", lambda: fake_torch(cuda, mps))
    assert detect_device("auto") == expected


def test_no_torch(monkeypatch):
    monkeypatch.setattr(dev_mod, "_import_torch", lambda: None)
    assert detect_device("auto") == "cpu"
    assert dev_mod.torch_version() is None


def test_preference_fallback(monkeypatch):
    monkeypatch.setattr(dev_mod, "_import_torch", lambda: fake_torch(False, False))
    assert detect_device("cuda") == "cpu"  # 不可用时降级


def test_preference_respected(monkeypatch):
    monkeypatch.setattr(dev_mod, "_import_torch", lambda: fake_torch(True, True))
    assert detect_device("cpu") == "cpu"
    assert detect_device("cuda") == "cuda"


@pytest.mark.parametrize(
    "device,expected", [("cuda", True), ("mps", False), ("cpu", False)]
)
def test_fp16_only_cuda(device, expected):
    assert resolve_fp16(True, device) is expected
    assert resolve_fp16(False, "cuda") is False


def test_blackwell_compat_warning(monkeypatch):
    t = fake_torch(True, False)
    t.version = types.SimpleNamespace(cuda="12.6")
    monkeypatch.setattr(dev_mod, "_import_torch", lambda: t)
    warns = dev_mod.check_blackwell_compat("cuda")
    assert warns and "cu128" in warns[0]


def test_blackwell_compat_ok(monkeypatch):
    monkeypatch.setattr(dev_mod, "_import_torch", lambda: fake_torch(True, False))
    assert dev_mod.check_blackwell_compat("cuda") == []
