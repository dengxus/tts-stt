#!/usr/bin/env python
"""简易压测：串行 + 并发 RTF / p50 / p95 / 显存峰值。

用法：
    python scripts/bench.py --tts-n 10 --stt-n 10 --concurrency 4
"""

from __future__ import annotations

import argparse
import io
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import soundfile as sf


def pctl(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(int(len(xs) * p), len(xs) - 1)] if xs else 0.0


def vram_peak_mb() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return info.used / 1024 / 1024
    except Exception:
        return None


def make_wav(seconds: float = 3.0, rate: int = 16000) -> bytes:
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    data = (np.sin(2 * np.pi * 300 * t) * 0.3).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, rate, format="WAV")
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--tts-n", type=int, default=5)
    ap.add_argument("--stt-n", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    client = httpx.Client(base_url=args.base_url, headers=headers, timeout=300)
    wav = make_wav()
    text = "这是一段用于压测的中文文本，服务会分块合成并统计时延。" * 2

    def one_tts():
        t0 = time.perf_counter()
        r = client.post("/api/v1/tts", json={"text": text, "format": "wav"})
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            raise RuntimeError(f"TTS {r.status_code}")
        dur = int(r.headers["X-Duration-Ms"]) / 1000
        return dt, dur

    def one_stt():
        t0 = time.perf_counter()
        r = client.post("/api/v1/stt", files={"file": ("b.wav", wav)})
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            raise RuntimeError(f"STT {r.status_code}")
        return dt, 3.0

    for label, fn, n in (("TTS", one_tts, args.tts_n), ("STT", one_stt, args.stt_n)):
        results = []
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(fn) for _ in range(n)]
            for f in futures:
                results.append(f.result())
        wall = time.perf_counter() - t0
        lat = [x[0] for x in results]
        rtf = [x[0] / max(x[1], 1e-6) for x in results]
        vram = vram_peak_mb()
        print(f"{label} n={n} conc={args.concurrency}: "
              f"p50={pctl(lat, .5):.2f}s p95={pctl(lat, .95):.2f}s "
              f"RTF均值={statistics.mean(rtf):.3f} 吞吐={n / wall:.2f} req/s"
              + (f" 显存占用={vram:.0f}MB" if vram else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
