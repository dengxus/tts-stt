"""音频编解码单测：解码矩阵 / 编码 round-trip / 嗅探 / 重采样。"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.audio.codec import (
    UnsupportedFormatError,
    decode_any,
    encode,
    resample,
    sniff_format,
)
from app.engines.base import PcmAudio

pytestmark = pytest.mark.unit


def sine_pcm(duration_s: float = 1.0, sr: int = 24000, freq: float = 440.0) -> PcmAudio:
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / sr
    return PcmAudio((np.sin(2 * np.pi * freq * t) * 0.6).astype(np.float32), sr)


def snr_db(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    err = np.mean((a - b) ** 2)
    if err <= 0:
        return 100.0
    return float(10 * np.log10(np.mean(a**2) / err))


# ---------- 解码矩阵 ----------

@pytest.mark.parametrize("fmt", ["wav", "flac", "ogg"])
def test_decode_soundfile_formats(fmt):
    data = encode(sine_pcm(1.0, 16000), fmt)
    pcm = decode_any(data)
    assert pcm.sample_rate == 16000
    assert abs(pcm.duration_ms - 1000) < 20


def test_decode_mp3():
    data = encode(sine_pcm(1.0, 24000), "mp3")
    pcm = decode_any(data)
    assert pcm.sample_rate == 24000
    # mp3 有编解码延迟（encoder delay ~576 样本），允许 60ms
    assert abs(pcm.duration_ms - 1000) < 60


def test_decode_m4a_pyav():
    """m4a/aac 走 PyAV 解码路径（soundfile 不支持 ftyp）。"""
    import av

    pcm_in = sine_pcm(1.0, 16000)
    buf = io.BytesIO()
    container = av.open(buf, "w", format="mp4")
    stream = container.add_stream("aac", rate=16000)
    stream.layout = "mono"
    ints = (np.clip(pcm_in.samples, -1, 1) * 32767).astype(np.int16)
    fsz = 1024
    for i in range(0, len(ints) - fsz + 1, fsz):
        frame = av.AudioFrame.from_ndarray(
            ints[i : i + fsz].reshape(1, -1), format="s16p", layout="mono")
        frame.sample_rate = 16000
        for p in stream.encode(frame):
            container.mux(p)
    for p in stream.encode(None):
        container.mux(p)
    container.close()

    pcm = decode_any(buf.getvalue())
    assert pcm.samples.dtype == np.float32
    assert abs(pcm.duration_ms - 1000) < 80  # aac priming 延迟


def test_decode_rejects_garbage():
    with pytest.raises(UnsupportedFormatError):
        decode_any(b"this is not audio at all, just text bytes")


def test_decode_too_short():
    with pytest.raises(UnsupportedFormatError):
        decode_any(b"RIFF")


def test_decode_fake_extension():
    """扩展名撒谎（实际是文本），靠魔数/解码失败拒绝。"""
    with pytest.raises(UnsupportedFormatError):
        decode_any(b"print('hello')\n" * 10)


# ---------- 嗅探 ----------

def test_sniff():
    assert sniff_format(encode(sine_pcm(0.2), "wav")) == "wav"
    assert sniff_format(encode(sine_pcm(0.2), "ogg")) == "ogg"
    assert sniff_format(encode(sine_pcm(0.2), "flac")) == "flac"
    assert sniff_format(encode(sine_pcm(0.2), "mp3")) == "mp3"
    assert sniff_format(b"nonsense") is None


# ---------- round-trip ----------

@pytest.mark.parametrize("fmt,expected_type", [
    ("wav", "audio/wav"), ("mp3", None), ("ogg", None), ("flac", None),
])
def test_encode_roundtrip_duration(fmt, expected_type):
    src = sine_pcm(2.0, 24000)
    data = encode(src, fmt)
    back = decode_any(data)
    assert abs(back.duration_ms - 2000) < (80 if fmt == "mp3" else 25)


def test_encode_roundtrip_snr():
    """wav 无损 round-trip：SNR 应很高。"""
    src = sine_pcm(1.0, 24000)
    back = decode_any(encode(src, "wav"))
    assert snr_db(src.samples, back.samples) > 40


@pytest.mark.parametrize("rate", [8000, 16000, 44100])
def test_encode_resample_on_output(rate):
    src = sine_pcm(1.0, 24000)
    data = encode(src, "wav", sample_rate=rate)
    back = decode_any(data)
    assert back.sample_rate == rate


# ---------- 重采样 ----------

def test_resample_length():
    src = sine_pcm(1.0, 24000)
    out = resample(src, 16000)
    assert out.sample_rate == 16000
    assert abs(len(out.samples) - 16000) <= 1


def test_resample_noop_same_rate():
    src = sine_pcm(0.5, 16000)
    assert resample(src, 16000) is src


def test_decode_with_target_rate():
    data = encode(sine_pcm(1.0, 24000), "wav")
    pcm = decode_any(data, target_rate=16000)
    assert pcm.sample_rate == 16000


def test_stereo_input_mixdown():
    """双声道输入 → 均值混单声道。"""
    t = np.arange(16000, dtype=np.float32) / 16000
    stereo = np.stack([np.sin(t * 100), np.cos(t * 100)], axis=1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, stereo, 16000, format="WAV")
    pcm = decode_any(buf.getvalue())
    assert pcm.samples.ndim == 1
    assert len(pcm.samples) == 16000
