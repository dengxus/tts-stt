"""音频编解码：任意字节流 <-> PcmAudio(float32 mono)。

选型结论（见方案 §核心机制）：
- 解码主干：soundfile（wav/flac/ogg/mp3，libsndfile 内置）→ 失败回落 PyAV
  （m4a/aac/opus/webm 等一切格式；PyAV wheel 自带 FFmpeg，**无需系统 ffmpeg**）
- 重采样：soxr（VHQ），不引入 librosa 全家桶
- 编码：wav/flac/ogg 走 soundfile；mp3 走 PyAV(libmp3lame)
- 不用 pydub（ffmpeg 子进程封装，Windows 需系统 ffmpeg）；不用 librosa（依赖重）
"""

from __future__ import annotations

import io
import logging

import numpy as np
import soundfile as sf
import soxr

from app.engines.base import PcmAudio
from app.errors import InvalidRequestError, UnsupportedFormatError

logger = logging.getLogger(__name__)

CONTENT_TYPES = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "mp3": "audio/mpeg",
}


def sniff_format(data: bytes) -> str | None:
    """魔数嗅探（防扩展名撒谎），仅用于报错提示与解码路由。"""
    if len(data) < 12:
        return None
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    if data[4:8] == b"ftyp":
        return "mp4"  # m4a/aac
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if data[:4] == b"\x30\x26\xb2\x75":
        return "asf"
    return None


def decode_any(data: bytes, target_rate: int | None = None) -> PcmAudio:
    """解码任意支持的音频格式 → float32 单声道，可选重采样到 target_rate。

    抛 UnsupportedFormatError（415）。
    """
    if len(data) < 12:
        raise UnsupportedFormatError("音频数据过短，不是有效文件")

    pcm = _decode_soundfile(data) or _decode_pyav(data)
    if pcm is None or pcm.samples.size == 0:
        raise UnsupportedFormatError(
            f"无法解码音频（嗅探格式={sniff_format(data) or '未知'}），"
            "支持 wav/mp3/m4a/ogg/opus/flac/webm"
        )
    if not np.isfinite(pcm.samples).all():  # pragma: no cover
        raise UnsupportedFormatError("解码结果包含非法样本")
    if target_rate and target_rate != pcm.sample_rate:
        pcm = resample(pcm, target_rate)
    return pcm


def _decode_soundfile(data: bytes) -> PcmAudio | None:
    try:
        arr, sr = sf.read(io.BytesIO(data), always_2d=True, dtype="float32")
    except Exception:  # noqa: BLE001 —— soundfile 对未知格式抛异常，属预期分流
        return None
    if arr.shape[0] == 0:
        return None
    mono = arr.mean(axis=1).astype(np.float32)
    return PcmAudio(mono, sr)


def _decode_pyav(data: bytes) -> PcmAudio | None:
    try:
        import av

        container = av.open(io.BytesIO(data))
        try:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                return None
            rate = stream.rate or 16000
            # 统一重排为 mono fltp（float32 planar），顺带完成混单声道
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for rf in resampler.resample(frame):
                    chunks.append(rf.to_ndarray().reshape(-1))
            try:  # flush 尾帧（部分 PyAV 版本不接受 None，失败不影响已解码数据）
                for rf in resampler.resample(None):
                    chunks.append(rf.to_ndarray().reshape(-1))
            except Exception:  # noqa: BLE001
                pass
            if not chunks:
                return None
            return PcmAudio(np.concatenate(chunks).astype(np.float32), rate)
        finally:
            container.close()
    except Exception as e:  # noqa: BLE001
        logger.debug("PyAV 解码失败: %s", e)
        return None


def resample(pcm: PcmAudio, target_rate: int) -> PcmAudio:
    if pcm.sample_rate == target_rate:
        return pcm
    out = soxr.resample(pcm.samples, pcm.sample_rate, target_rate, quality="VHQ")
    return PcmAudio(np.ascontiguousarray(out, dtype=np.float32), target_rate)


# ---------- 编码 ----------

def encode(pcm: PcmAudio, fmt: str = "wav", sample_rate: int | None = None) -> bytes:
    fmt = fmt.lower().strip()
    if fmt not in CONTENT_TYPES:
        raise InvalidRequestError(
            f"不支持的输出格式 {fmt!r}，可选: {sorted(CONTENT_TYPES)}"
        )
    sr = sample_rate or pcm.sample_rate
    data = resample(pcm, sr).samples if sr != pcm.sample_rate else pcm.samples
    # libsndfile/编码器要求 [-1, 1]
    data = np.clip(data, -1.0, 1.0)

    buf = io.BytesIO()
    if fmt == "wav":
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    elif fmt == "flac":
        sf.write(buf, data, sr, format="FLAC")
    elif fmt == "ogg":
        sf.write(buf, data, sr, format="OGG", subtype="VORBIS")
    elif fmt == "mp3":
        _encode_mp3(buf, data, sr)
    return buf.getvalue()


def _encode_mp3(buf: io.BytesIO, data: np.ndarray, sr: int) -> None:
    import av

    container = av.open(buf, "w", format="mp3")
    try:
        stream = container.add_stream("libmp3lame", rate=sr)
        stream.bit_rate = 192_000
        stream.layout = "mono"
        int_data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
        frame_size = 1152 * 8  # mp3 frame=1152，取整倍数分帧
        for i in range(0, len(int_data), frame_size):
            frame = av.AudioFrame.from_ndarray(
                int_data[i : i + frame_size].reshape(1, -1),
                format="s16p", layout="mono",
            )
            frame.sample_rate = sr
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
