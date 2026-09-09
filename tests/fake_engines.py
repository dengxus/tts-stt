"""假引擎：不装真模型也能测全 API 栈（CI 用）。"""

from __future__ import annotations

import asyncio

import numpy as np

from app.engines.base import (
    EngineState,
    PcmAudio,
    Segment,
    STTEngine,
    Transcript,
    TTSEngine,
    TTSOptions,
    VoiceInfo,
)


class FakeTTSEngine(TTSEngine):
    """返回 440Hz 正弦波；时长 = 文本长度 × 0.2s × 1/speed。"""

    name = "fake-tts"
    native_sample_rate = 24000

    def __init__(self, settings=None):
        self.loaded_with: tuple[str, bool] | None = None
        self.calls: list[str] = []
        self.opts_seen: list[TTSOptions] = []

    async def load(self, device: str, fp16: bool) -> None:
        self.loaded_with = (device, fp16)
        self.state = EngineState.READY

    async def unload(self) -> None:
        self.state = EngineState.UNLOADED

    def synthesize(self, text: str, opts: TTSOptions) -> PcmAudio:
        self.calls.append(text)
        self.opts_seen.append(opts)
        dur_s = max(len(text) * 0.2 / max(opts.speed, 0.1), 0.1)
        n = int(dur_s * self.native_sample_rate)
        t = np.arange(n, dtype=np.float32) / self.native_sample_rate
        return PcmAudio((np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32),
                        self.native_sample_rate)

    def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id="fake", name="假音色", engine=self.name, mode="sft",
                          languages=["zh", "en"], gender="female",
                          sample_rate=self.native_sample_rate)]


class FakeSTTEngine(STTEngine):
    """固定返回文本，并回显音频信息以便断言解码链路。"""

    name = "fake-stt"
    required_sample_rate = 16000

    def __init__(self, settings=None):
        self.loaded_with: tuple[str, bool] | None = None
        self.last_opts: tuple | None = None

    async def load(self, device: str, fp16: bool) -> None:
        self.loaded_with = (device, fp16)
        self.state = EngineState.READY

    async def unload(self) -> None:
        self.state = EngineState.UNLOADED

    def transcribe(self, audio: PcmAudio, opts) -> Transcript:
        self.last_opts = (audio.sample_rate, audio.samples.size, opts.language,
                          opts.use_itn)
        return Transcript(
            text="你好世界",
            language="zh",
            segments=[Segment(0, audio.duration_ms, "你好世界")],
            emotion="HAPPY",
            events=["Speech"],
        )


class SlowLoadSTTEngine(FakeSTTEngine):
    """load 耗时 2s，测加载中 503。"""

    name = "slow-stt"

    async def load(self, device: str, fp16: bool) -> None:
        await asyncio.sleep(2.0)
        await super().load(device, fp16)


class HangingTTSEngine(FakeTTSEngine):
    """synthesize 阻塞 2s，配合 request_timeout_s<2 测 504。"""

    name = "hanging-tts"

    def synthesize(self, text: str, opts: TTSOptions) -> PcmAudio:
        import time

        time.sleep(2.0)
        return super().synthesize(text, opts)


class BrokenTTSEngine(FakeTTSEngine):
    """synthesize 抛 OOM 异常，测 503 inference_failed 的 OOM 映射。"""

    name = "broken-tts"

    def synthesize(self, text: str, opts: TTSOptions) -> PcmAudio:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


class FailingLoadTTSEngine(FakeTTSEngine):
    async def load(self, device: str, fp16: bool) -> None:
        raise RuntimeError("权重文件缺失")
