"""CosyVoice2-0.5B TTS 引擎（vendored 于 third_party/CosyVoice，sys.path 引入）。

要点：
- CosyVoice 的构造器内部自行选择 cuda/cpu（无 device 参数）；Mac 开发机自然落在
  CPU fp32，生产 Windows 落在 CUDA fp16。
- 三种合成模式：sft（预置音色）/ instruct2（语气指令，需打包参考音频）/
  zero_shot（用户上传参考，受 ALLOW_ZERO_SHOT 门控）。
- inference_* 返回 generator（stream=False 时通常一个块），全部收集后拼接。
- WeTextProcessing（pynini）缺失时仓库按 find_spec 探测自动跳过 TN，
  服务层再用 app/utils/textnorm.py 的轻量 regex 兜底。
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

from app.config import Settings
from app.engines.base import (
    EngineState,
    PcmAudio,
    TTSEngine,
    TTSOptions,
    VoiceInfo,
)
from app.errors import InvalidRequestError
from app.models_spec import TTS_SPECS, is_downloaded

logger = logging.getLogger(__name__)

_VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "CosyVoice"

# instruct2 模式要求指令以结束标记收尾（CosyVoice example.py 用法）
END_OF_PROMPT = "<|endofprompt|>"

# 打包音色注册表：CosyVoice2-0.5B 权重不内置 sft 音色（spk2info.pt 缺省），
# 官方定位 zero-shot。启动时用 add_zero_shot_spk 把仓库 asset 参考音频注册为
# 命名音色，API 侧即可当预置音色用（特征只算一次并持久化到 spk2info.pt）。
BUNDLED_SPEAKERS = [
    {
        "id": "default",
        "name": "默认女声（预置参考音）",
        "prompt_text": "希望你以后能够做的比我还好呦。",
        "wav": _VENDOR / "asset" / "zero_shot_prompt.wav",
    },
]


def _bootstrap_path() -> None:
    """注入 vendored 包路径（幂等）。"""
    for p in (_VENDOR, _VENDOR / "third_party" / "Matcha-TTS"):
        if not (p / ("cosyvoice" if p == _VENDOR else "matcha") / "__init__.py"
                ).exists():
            raise RuntimeError(
                f"CosyVoice 源码缺失（{p}），请执行: "
                "git submodule update --init --recursive --depth 1")
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


class CosyVoice2Engine(TTSEngine):
    name = "cosyvoice2"
    native_sample_rate = 24000

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings(_env_file=None)
        self.model = None
        self.native_sample_rate = 24000
        self._spks: list[str] = []          # 权重内置 sft 音色（通常为空）
        self._registered: dict[str, VoiceInfo] = {}  # 注册式打包音色

    # ---------- 生命周期 ----------

    async def load(self, device: str, fp16: bool) -> None:
        _bootstrap_path()
        spec = TTS_SPECS[0]
        root = Path(self._settings.model_dir)
        if not is_downloaded(root, spec):
            raise RuntimeError(
                f"CosyVoice2 权重缺失（{root / spec.dir_name}），"
                "请先运行 scripts/download_models.py --only tts")
        model_dir = str(root / spec.dir_name)
        use_fp16 = bool(fp16 and device == "cuda")

        def _load():
            from cosyvoice.cli.cosyvoice import CosyVoice2

            return CosyVoice2(model_dir, load_jit=False, load_trt=False,
                              fp16=use_fp16)

        self.model = await asyncio.to_thread(_load)
        self.native_sample_rate = int(self.model.sample_rate)
        if not use_fp16:
            self._cast_qwen_to_fp32()
        try:
            self._spks = list(self.model.list_available_spks())
        except Exception as e:  # noqa: BLE001
            logger.warning("枚举预置音色失败: %s", e)
            self._spks = []
        self._register_bundled_speakers()
        self.state = EngineState.READY
        logger.info("CosyVoice2 加载完成: fp16=%s, sft音色=%s, 注册音色=%s",
                    use_fp16, self._spks, list(self._registered))

    def _cast_qwen_to_fp32(self) -> None:
        """transformers 的 from_pretrained 按 config 以 bf16 加载 Qwen2 权重，
        而 fp32 推理路径（CPU 开发机 / FP16=false）会报
        'mat1 and mat2 must have the same dtype'，统一转回 fp32。"""
        assert self.model is not None
        try:
            self.model.model.llm.llm.model.float()
            logger.info("Qwen2 权重已转 fp32（fp16 未启用）")
        except AttributeError:  # pragma: no cover —— 结构漂移时兜底扫描
            n = 0
            for mod in self.model.model.llm.llm.modules():
                if type(mod).__name__ == "Qwen2ForCausalLM":
                    mod.float()
                    n += 1
            logger.info("Qwen2 float() 兜底扫描: %d 处", n)

    def _register_bundled_speakers(self) -> None:
        """把打包参考音频注册为命名音色（幂等：已在 spk2info 则跳过注册）。"""
        custom_dir = Path("assets/prompts")
        specs = list(BUNDLED_SPEAKERS)
        for wav in sorted(custom_dir.glob("*.wav")) if custom_dir.is_dir() else []:
            txt = wav.with_suffix(".txt")
            if txt.exists():
                specs.append({"id": wav.stem, "name": wav.stem,
                              "prompt_text": txt.read_text("utf8").strip(),
                              "wav": wav})
        changed = False
        for sp in specs:
            vid, wav = sp["id"], Path(sp["wav"])
            if not wav.exists():
                logger.warning("打包音色参考音频缺失，跳过 %s: %s", vid, wav)
                continue
            if vid not in self._spks:
                try:
                    assert self.model is not None
                    self.model.add_zero_shot_spk(sp["prompt_text"], str(wav), vid)
                    changed = True
                except Exception as e:  # noqa: BLE001
                    logger.warning("注册音色 %s 失败: %s", vid, e)
                    continue
            self._registered[vid] = VoiceInfo(
                id=vid, name=sp["name"], engine=self.name, mode="zero_shot",
                languages=["zh", "en"], sample_rate=self.native_sample_rate,
                tags=["bundled"])
        if changed:
            try:  # 持久化到权重目录 spk2info.pt，下次启动免重算参考特征
                self.model.save_spkinfo()
            except Exception as e:  # 目录只读等情况不影响服务
                logger.warning("spk2info 持久化跳过: %s", e)

    async def unload(self) -> None:
        self.model = None
        self.state = EngineState.UNLOADED

    def warmup(self) -> None:
        try:
            self.synthesize("你好。", TTSOptions())
        except Exception as e:  # noqa: BLE001
            logger.warning("warmup 失败（不影响服务）: %s", e)

    # ---------- 推理（同步，由 InferenceGuard 放线程） ----------

    def synthesize(self, text: str, opts: TTSOptions) -> PcmAudio:
        assert self.model is not None
        text = self._prenormalize(text)
        if opts.seed is not None:
            # LLM 采样有随机性：固定种子使同文本输出可复现，便于客户端可控重试
            from cosyvoice.utils.common import set_all_random_seed

            set_all_random_seed(opts.seed)

        if opts.zero_shot is not None:
            gen_fn, extra = self._mode_zero_shot(opts)
        elif opts.instruct:
            gen_fn, extra = self._mode_instruct(opts)
        else:
            gen_fn, extra = self._mode_voice(opts)

        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = extra(tmpdir)
            stream = gen_fn(text, **kwargs, stream=False, speed=opts.speed)
            chunks = [out["tts_speech"] for out in stream]
        if not chunks:
            raise RuntimeError("CosyVoice2 未产出音频块")

        import torch

        speech = torch.cat(chunks, dim=1).squeeze(0).float().cpu().numpy()
        return PcmAudio(np.ascontiguousarray(speech, dtype=np.float32),
                        self.native_sample_rate)

    def _mode_voice(self, opts: TTSOptions):
        spk = opts.voice or self.default_voice()
        if spk in self._registered:
            # add_zero_shot_spk 注册的音色：走 zero_shot 缓存路径（免每次算参考特征）
            return (self.model.inference_zero_shot,
                    lambda td: {"prompt_text": "", "prompt_wav": "",
                                "zero_shot_spk_id": spk})
        if spk in self._spks:  # 权重自带 sft 音色（CosyVoice-300M-SFT 类模型）
            return self.model.inference_sft, lambda td: {"spk_id": spk}
        raise InvalidRequestError(
            f"音色 {spk!r} 不存在，可用: {sorted(set(self._spks) | set(self._registered))}"
            "（见 GET /api/v1/voices）")

    def _mode_instruct(self, opts: TTSOptions):
        prompt_wav = self._instruct_prompt_wav()
        instruct = opts.instruct
        if not instruct.endswith(END_OF_PROMPT):
            instruct += END_OF_PROMPT  # 官方 example 要求显式结束标记
        return (self.model.inference_instruct2,
                lambda td: {"instruct_text": instruct, "prompt_wav": prompt_wav})

    def _mode_zero_shot(self, opts: TTSOptions):
        ref = opts.zero_shot
        assert ref is not None
        return (self.model.inference_zero_shot,
                lambda td: self._write_ref(td, ref))

    @staticmethod
    def _write_ref(tmpdir: str, ref) -> dict:
        import soundfile as sf

        path = str(Path(tmpdir) / "prompt.wav")
        sf.write(path, ref.prompt_audio.samples, ref.prompt_audio.sample_rate,
                 format="WAV", subtype="PCM_16")
        return {"prompt_text": ref.prompt_text, "prompt_wav": path}

    # ---------- 音色与文本预处理 ----------

    def default_voice(self) -> str:
        if self._settings.default_voice:
            return self._settings.default_voice
        if "default" in self._registered:
            return "default"
        return self._spks[0] if self._spks else (next(iter(self._registered), ""))

    def _instruct_prompt_wav(self) -> str:
        """instruct2 需要一个参考音频提供音色：优先仓库自带 asset。"""
        candidates = [
            Path(self._settings.model_dir) / "instruct_prompt.wav",
            _VENDOR / "asset" / "zero_shot_prompt.wav",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        raise InvalidRequestError(
            "instruct 模式缺少参考音频（models/instruct_prompt.wav 或仓库 asset）")

    def _prenormalize(self, text: str) -> str:
        """WeTextProcessing 可用时交给模型前端；缺失时启用轻量 regex TN。"""
        if _has_wetext():
            return text
        from app.utils.textnorm import normalize
        return normalize(text)

    def list_voices(self) -> list[VoiceInfo]:
        # _spks 可能含已注册的打包音色（持久化进 spk2info 后枚举可见），去重
        voices = [
            VoiceInfo(id=spk, name=spk, engine=self.name, mode="sft",
                      languages=["zh", "en"], sample_rate=self.native_sample_rate,
                      tags=["preset"])
            for spk in self._spks if spk not in self._registered
        ]
        voices.extend(self._registered.values())
        try:
            prompt = self._instruct_prompt_wav()
        except InvalidRequestError:
            prompt = ""
        if prompt:
            voices.append(VoiceInfo(
                id="__instruct__", name="语气指令（任意音色+instruct）",
                engine=self.name, mode="instruct2",
                languages=["zh", "en"],
                sample_rate=self.native_sample_rate,
                tags=["instruct"],
                default_instruct="用开心的语气说"))
        return voices

    def extra_status(self) -> dict:
        return {"sft_spks": len(self._spks),
                "registered_voices": list(self._registered),
                "wetext_available": _has_wetext()}


_tn_check: bool | None = None
_tn_lock = threading.Lock()


def _has_wetext() -> bool:
    """vendored frontend 用 wetext（新，纯 pip 可装）做 TN；缺失时走降级。"""
    global _tn_check
    with _tn_lock:
        if _tn_check is None:
            import importlib.util

            _tn_check = importlib.util.find_spec("wetext") is not None
        return _tn_check
