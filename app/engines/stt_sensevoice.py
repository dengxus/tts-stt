"""SenseVoice-Small STT 引擎（FunASR 生态，纯 PyTorch）。

设计决策：
- FSMN-VAD **独立加载、手动分段**，而不是把 vad_model 传给 SenseVoice 的 AutoModel：
  funasr 内部 vad-merge 的输出格式随版本漂移，手动切片后每段独立转写，
  段级时间戳由 VAD 边界直接给出，确定性且有单测可验证。
- fp16 不启用：SenseVoice 仅 234M 参数（fp32 总占用 ~1.3GB），
  省显存收益小，规避 funasr 各版本对 half 权重处理的不一致。
- Mac 开发（mps）强制 cpu：SenseVoice 在 CPU 上 RTF 已 <0.1，规避 mps 算子坑。
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import numpy as np

from app.config import Settings
from app.engines.base import (
    EngineState,
    PcmAudio,
    Segment,
    STTEngine,
    STTOptions,
    Transcript,
)
from app.models_spec import STT_SPECS, is_downloaded

logger = logging.getLogger(__name__)

_SR = 16000
_TAG_RUN = re.compile(r"^(?:<\|[^|<>]+\|>)+")
_ONE_TAG = re.compile(r"<\|([^|<>]+)\|>")
_LANGS = {"zh", "en", "ja", "ko", "yue", "aut"}
_EMOTIONS = {"NEUTRAL", "HAPPY", "SAD", "ANGRY", "ANGERY", "FEARFUL",
             "DISGUSTED", "SURPRISED", "EMOTION_UNKNOWN"}
_ITN_TAGS = {"withitn", "woitn"}

# 短于此直接用单段转写，跳过 VAD（VAD 本身也要一次推理）
_VAD_THRESHOLD_S = 5.0
# SenseVoice 建议单段 <=30s
_MAX_SEGMENT_MS = 30_000


def parse_sensevoice_output(raw: str) -> tuple[str, list[str]]:
    """解析 `<|zh|><|HAPPY|><|Speech|><|withitn|>你好…` 形式输出。

    返回 (纯文本, 标签列表)；无标签时原样返回。
    """
    m = _TAG_RUN.match(raw)
    if not m:
        return raw, []
    tags = _ONE_TAG.findall(m.group(0))
    return raw[m.end():].strip(), tags


def tags_to_fields(tags: list[str]) -> dict:
    out = {"language": "", "emotion": None, "events": []}
    for t in tags:
        low = t.lower()
        if t in _LANGS or low in _LANGS:
            out["language"] = "" if low == "aut" else low
        elif t.upper() in _EMOTIONS:
            e = "NEUTRAL" if t.upper() in ("NEUTRAL", "EMOTION_UNKNOWN") else t.upper()
            out["emotion"] = e
        elif low not in _ITN_TAGS:
            out["events"].append(t)
    return out


class SenseVoiceEngine(STTEngine):
    name = "sensevoice-small"
    required_sample_rate = _SR

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings(_env_file=None)
        self.model = None
        self.vad = None
        self._device = "cpu"
        self._load_error = ""

    # ---------- 生命周期 ----------

    def _model_dirs(self) -> tuple[Path, Path | None]:
        root = Path(self._settings.model_dir)
        spec_sv = STT_SPECS[0]
        spec_vad = STT_SPECS[1]
        if not is_downloaded(root, spec_sv):
            raise RuntimeError(
                f"SenseVoice 权重缺失（{root / spec_sv.dir_name}），"
                "请先运行 scripts/download_models.py --only stt"
            )
        vad_dir = root / spec_vad.dir_name
        return root / spec_sv.dir_name, vad_dir if is_downloaded(root, spec_vad) else None

    async def load(self, device: str, fp16: bool) -> None:
        sv_dir, vad_dir = self._model_dirs()
        self._device = "cuda" if device == "cuda" else "cpu"
        target_device = self._device

        def _load():
            from funasr import AutoModel

            sv = AutoModel(model=str(sv_dir), device=target_device,
                           disable_update=True)
            vad = (AutoModel(model=str(vad_dir), device="cpu",
                             disable_update=True)
                   if vad_dir else None)
            return sv, vad

        self.model, self.vad = await asyncio.to_thread(_load)
        self.state = EngineState.READY
        logger.info("SenseVoice 加载完成: device=%s, vad=%s",
                    self._device, bool(self.vad))

    async def unload(self) -> None:
        self.model = None
        self.vad = None
        self.state = EngineState.UNLOADED

    def warmup(self) -> None:
        silence = np.zeros(_SR, dtype=np.float32)
        try:
            self.transcribe(PcmAudio(silence, _SR), STTOptions())
        except Exception as e:  # noqa: BLE001
            logger.warning("warmup 失败（不影响服务）: %s", e)

    # ---------- 推理（同步，由 InferenceGuard 放线程） ----------

    def transcribe(self, audio: PcmAudio, opts: STTOptions) -> Transcript:
        assert self.model is not None
        if audio.sample_rate != _SR:
            raise ValueError("SenseVoiceEngine 要求 16kHz 输入（codec 已重采样）")
        duration_ms = audio.duration_ms

        ranges = self._speech_ranges(audio, duration_ms)
        parts: list[Segment] = []
        texts: list[str] = []
        languages: list[str] = []
        emotions: list[str] = []
        events: set[str] = set()

        for beg, end in ranges:
            chunk = audio.samples[int(beg / 1000 * _SR): int(end / 1000 * _SR)]
            if chunk.size < int(0.1 * _SR):
                continue
            res = self.model.generate(
                input=chunk, cache={},
                language=opts.language, use_itn=opts.use_itn,
                batch_size_s=60, disable_pbar=True,
            )
            raw = (res[0].get("text", "") if res else "") or ""
            text, tags = parse_sensevoice_output(raw)
            fields = tags_to_fields(tags)
            if not text:
                continue
            texts.append(text)
            if opts.timestamps:
                parts.append(Segment(start_ms=beg, end_ms=end, text=text))
            if fields["language"]:
                languages.append(fields["language"])
            if fields["emotion"]:
                emotions.append(fields["emotion"])
            events.update(fields["events"])

        language = (opts.language if opts.language != "auto"
                    else (languages[0] if languages else "und"))
        emotion = None
        if opts.output_rich and emotions:
            emotion = ("NEUTRAL" if all(e == "NEUTRAL" for e in emotions)
                       else max(set(emotions), key=emotions.count))
        return Transcript(
            text="".join(texts),
            language=language,
            segments=parts,
            emotion=emotion,
            events=sorted(events) if opts.output_rich else [],
        )

    def _speech_ranges(self, audio: PcmAudio, duration_ms: int) -> list[tuple[int, int]]:
        """返回 [(beg_ms, end_ms)]：短音频单段；长音频用 FSMN-VAD 并保证 <=30s/段。"""
        if self.vad is None or duration_ms < _VAD_THRESHOLD_S * 1000:
            return self._cap([(0, duration_ms)])
        try:
            res = self.vad.generate(input=audio.samples, cache={},
                                    max_single_segment_time=_MAX_SEGMENT_MS,
                                    disable_pbar=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("VAD 失败，回退单段转写: %s", e)
            return self._cap([(0, duration_ms)])
        pairs = []
        for item in res or []:
            for p in item.get("value") or []:
                beg, end = int(p[0]), int(p[1])
                if end == -1:  # 语音未闭合的尾部
                    end = duration_ms
                if end > beg >= 0:
                    pairs.append((beg, min(end, duration_ms)))
        if not pairs:
            logger.info("VAD 未检出语音（可能为静音/纯噪声）")
            return []
        return self._cap(pairs)

    @staticmethod
    def _cap(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out = []
        for beg, end in pairs:
            while end - beg > _MAX_SEGMENT_MS:
                out.append((beg, beg + _MAX_SEGMENT_MS))
                beg += _MAX_SEGMENT_MS
            if end > beg:
                out.append((beg, end))
        return out

    def extra_status(self) -> dict:
        return {"vad_enabled": self.vad is not None}
