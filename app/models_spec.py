"""模型组件清单：下载脚本与引擎共用，避免目录名漂移。

下载渠道：modelscope 主通道（国内快、免代理），hf 备通道（配合 HF_ENDPOINT 镜像）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str                 # 引擎侧引用名
    dir_name: str            # MODEL_DIR 下的落地目录名
    modelscope_id: str
    hf_id: str
    required_files: list[str] = field(default_factory=list)
    kind: str = ""           # stt | tts


STT_SPECS: list[ModelSpec] = [
    ModelSpec(
        key="sensevoice",
        dir_name="SenseVoiceSmall",
        modelscope_id="iic/SenseVoiceSmall",
        hf_id="FunAudioLLM/SenseVoiceSmall",
        required_files=["model.pt"],
        kind="stt",
    ),
    ModelSpec(
        key="vad",
        dir_name="speech_fsmn_vad_zh-cn-16k-common-pytorch",
        modelscope_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        hf_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        required_files=["model.pt"],
        kind="stt",
    ),
]

TTS_SPECS: list[ModelSpec] = [
    ModelSpec(
        key="cosyvoice2",
        dir_name="CosyVoice2-0.5B",
        modelscope_id="iic/CosyVoice2-0.5B",
        hf_id="FunAudioLLM/CosyVoice2-0.5B",
        required_files=["cosyvoice2.yaml", "llm.pt", "flow.pt", "hift.pt"],
        kind="tts",
    ),
]

ALL_SPECS = STT_SPECS + TTS_SPECS


def spec_paths(model_dir: Path) -> dict[str, Path]:
    return {s.key: Path(model_dir) / s.dir_name for s in ALL_SPECS}


def is_downloaded(model_dir: Path, spec: ModelSpec) -> bool:
    p = Path(model_dir) / spec.dir_name
    return p.is_dir() and all((p / f).exists() for f in spec.required_files)
