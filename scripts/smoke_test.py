#!/usr/bin/env python
"""端到端冒烟：对运行中的服务执行 TTS → STT 回环一致性检查。

用法：
    python scripts/smoke_test.py --base-url http://localhost:8000 [--zh|--en]
退出码 0=全部通过。TTS 合成中文句 → 保存 → 回灌 STT → 比对（CER 阈值）。
"""

from __future__ import annotations

import argparse
import difflib
import sys

import httpx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--cer-threshold", type=float, default=0.15)
    args = ap.parse_args()

    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    client = httpx.Client(base_url=args.base_url, headers=headers, timeout=180)

    ok = True
    h = client.get("/health").json()
    print(f"device={h['device']} tts={h['engines'].get('tts', {}).get('state')} "
          f"stt={h['engines'].get('stt', {}).get('state')}")
    for kind in ("tts", "stt"):
        if h["engines"].get(kind, {}).get("state") != "ready":
            print(f"[FAIL] 引擎 {kind} 未就绪")
            return 1

    voices = client.get("/api/v1/voices").json()["voices"]
    named = [v for v in voices if v["mode"] in ("sft", "zero_shot")]
    voice = "default" if any(v["id"] == "default" for v in named) else (
        named[0]["id"] if named else None)
    print(f"音色: {voice}（共 {len(voices)} 个: {[v['id'] for v in voices]}）")

    samples = [
        ("中文女声冒烟测试，服务运行正常。", voice),
        ("The quick brown fox jumps over the lazy dog.", voice),
        ("数字测试：2024年增长10.5%，共1234人。", voice),
    ]
    for text, v in samples:
        r = client.post("/api/v1/tts", json={"text": text, "voice": v, "format": "wav"})
        if r.status_code != 200:
            print(f"[FAIL] TTS {r.status_code}: {r.text[:120]}")
            ok = False
            continue
        ms = r.headers.get("X-Duration-Ms")
        rtf = r.headers.get("X-RTF")
        st = client.post("/api/v1/stt",
                         files={"file": ("smoke.wav", r.content)},
                         data={"language": "auto"})
        if st.status_code != 200:
            print(f"[FAIL] STT {st.status_code}: {st.text[:120]}")
            ok = False
            continue
        got = st.json()["text"].strip()
        sim = difflib.SequenceMatcher(None, text.replace(" ", ""),
                                      got.replace(" ", "")).ratio()
        passed = sim >= 1 - args.cer_threshold
        ok &= passed
        print(f"[{'ok ' if passed else 'FAIL'}] 回环相似度 {sim:.2f} "
              f"(tts {ms}ms rtf={rtf})  {text[:18]}… -> {got[:24]}")

    instruct = client.post("/api/v1/tts", json={
        "text": "今天天气真好！", "instruct": "用开心的语气说", "format": "mp3"})
    print(f"[*] instruct 模式: {instruct.status_code}")
    ok &= instruct.status_code == 200

    print("\n" + ("[PASS] 冒烟全部通过" if ok else "[FAIL] 存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
