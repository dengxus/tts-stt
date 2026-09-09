#!/usr/bin/env python
"""模型权重下载：modelscope 主通道 / hf 备通道，带重试与完整性校验。

用法：
    python scripts/download_models.py --only stt            # 下载 SenseVoice + VAD (~1GB)
    python scripts/download_models.py --only tts            # 下载 CosyVoice2-0.5B (~2GB)
    python scripts/download_models.py --source hf --hf-endpoint https://hf-mirror.com
离线部署：手工下载后把目录拷入 MODEL_DIR（目录名见 app/models_spec.py）即可。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models_spec import ALL_SPECS, ModelSpec, is_downloaded  # noqa: E402


def _download(spec: ModelSpec, source: str, target: Path, hf_endpoint: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source == "modelscope":
        from modelscope import snapshot_download

        snapshot_download(spec.modelscope_id, local_dir=str(target))
    else:
        import os

        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        from huggingface_hub import snapshot_download as hf_download

        hf_download(spec.hf_id, local_dir=str(target))


def fetch(spec: ModelSpec, model_dir: Path, source: str, hf_endpoint: str,
          retries: int = 3) -> bool:
    target = model_dir / spec.dir_name
    if is_downloaded(model_dir, spec):
        print(f"[skip] {spec.dir_name} 已存在且校验通过")
        return True
    for attempt in range(1, retries + 1):
        try:
            print(f"[down] {spec.key}: {spec.modelscope_id if source=='modelscope' else spec.hf_id}"
                  f" -> {target}（第 {attempt}/{retries} 次）")
            t0 = time.monotonic()
            _download(spec, source, target, hf_endpoint)
            if is_downloaded(model_dir, spec):
                print(f"[ ok ] {spec.dir_name} 完成 ({time.monotonic() - t0:.0f}s)")
                return True
            print(f"[fail] {spec.dir_name} 缺少必需文件 {spec.required_files}")
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {spec.dir_name} 下载异常: {e}")
            time.sleep(min(2 ** attempt, 10))
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=str(ROOT / "models"))
    ap.add_argument("--source", choices=["modelscope", "hf"], default="modelscope")
    ap.add_argument("--hf-endpoint", default="", help="例如 https://hf-mirror.com")
    ap.add_argument("--only", choices=["stt", "tts", "all"], default="all")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    specs = {"stt": [s for s in ALL_SPECS if s.kind == "stt"],
             "tts": [s for s in ALL_SPECS if s.kind == "tts"],
             "all": list(ALL_SPECS)}[args.only]

    failed = [s.key for s in specs if not fetch(s, model_dir, args.source,
                                                args.hf_endpoint)]
    if failed:
        print(f"\n[FAIL] 下载失败: {failed}（换 --source 或 --hf-endpoint 重试，或手工放置模型）")
        return 1
    print("\n[PASS] 全部模型就绪于", model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
