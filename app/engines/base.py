"""引擎抽象层。

关键决策：
- 引擎的推理方法是**同步阻塞**的（GPU 推理天然阻塞），async 只出现在生命周期上；
  并发适配统一放在 manager.InferenceGuard（to_thread + 信号量），
  引擎实现不碰 asyncio —— 便于被脚本/测试直接同步调用。
- 引擎间统一中间表示为 PcmAudio（float32 单声道 [-1,1] + 采样率），
  格式编解码全部收敛在 app/audio/codec.py，新引擎即插即用。
- synthesize_stream() 为流式（SSE/chunked）预留扩展点，本期不实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class EngineState(str, Enum):  # 3.10 无 StrEnum
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"  # 如：OOM 后降级到 CPU 继续服务
    ERROR = "error"


@dataclass
class PcmAudio:
    samples: np.ndarray  # float32, mono, [-1, 1]
    sample_rate: int

    @property
    def duration_ms(self) -> int:
        return int(round(len(self.samples) / self.sample_rate * 1000))


@dataclass
class ZeroShotRef:
    """zero-shot 声音克隆参考。"""

    prompt_text: str
    prompt_audio: PcmAudio


@dataclass
class TTSOptions:
    voice: str | None = None
    speed: float = 1.0
    instruct: str | None = None            # 触发 instruct2 模式
    zero_shot: ZeroShotRef | None = None   # 触发 zero-shot 模式
    seed: int | None = None


@dataclass
class STTOptions:
    language: str = "auto"   # auto|zh|en|ja|ko|yue
    use_itn: bool = True
    timestamps: bool = True
    output_rich: bool = False  # 保留情绪/事件标签


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    text: str


@dataclass
class Transcript:
    text: str
    language: str = "auto"
    segments: list[Segment] = field(default_factory=list)
    emotion: str | None = None
    events: list[str] = field(default_factory=list)


@dataclass
class VoiceInfo:
    id: str
    name: str
    engine: str
    mode: str                    # sft | instruct2 | zero_shot
    languages: list[str]
    gender: str = ""
    sample_rate: int = 0
    tags: list[str] = field(default_factory=list)
    default_instruct: str = ""


class TTSEngine(ABC):
    name: str = "tts-base"
    native_sample_rate: int = 24000

    state: EngineState = EngineState.UNLOADED

    @abstractmethod
    async def load(self, device: str, fp16: bool) -> None: ...

    async def unload(self) -> None:
        self.state = EngineState.UNLOADED

    def warmup(self) -> None:  # noqa: B027 —— 刻意为可选钩子，非抽象
        """可选：加载后跑一次哑推理，预分配 cuDNN workspace，消除首请求长尾。"""

    @abstractmethod
    def synthesize(self, text: str, opts: TTSOptions) -> PcmAudio:
        """同步阻塞实现，由 InferenceGuard 放入线程池执行。"""

    def synthesize_stream(self, text: str, opts: TTSOptions) -> Iterator[np.ndarray]:
        """流式预留接口，本期不实现。"""
        raise NotImplementedError(f"{self.name} 暂不支持流式合成")

    @abstractmethod
    def list_voices(self) -> list[VoiceInfo]: ...

    def extra_status(self) -> dict:
        return {}


class STTEngine(ABC):
    name: str = "stt-base"
    required_sample_rate: int = 16000

    state: EngineState = EngineState.UNLOADED

    @abstractmethod
    async def load(self, device: str, fp16: bool) -> None: ...

    async def unload(self) -> None:
        self.state = EngineState.UNLOADED

    def warmup(self) -> None:  # noqa: B027 —— 刻意为可选钩子，非抽象
        """可选：见 TTSEngine.warmup。"""

    @abstractmethod
    def transcribe(self, audio: PcmAudio, opts: STTOptions) -> Transcript:
        """同步阻塞实现，由 InferenceGuard 放入线程池执行。audio 已重采样到
        required_sample_rate。"""

    def extra_status(self) -> dict:
        return {}
