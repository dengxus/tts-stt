# 架构与决策记录

## 总体

```
                 ┌────────────────────────── FastAPI (workers=1) ──────────────────────────┐
 客户端 ──HTTP──▶ │ /api/v1/tts  /api/v1/stt  /api/v1/voices  /health*                     │
                 │      │            │                                                     │
                 │ audio/codec    audio/codec        InferenceGuard(全局GPU信号量=1,        │
                 │ (PyAV+sf+soxr) (PyAV+sf+soxr)     队列上限, 超时, to_thread)             │
                 │      │            │            │                                        │
                 │      ▼            ▼            ▼                                        │
                 │  CosyVoice2Engine   SenseVoiceEngine   ← engines/base.py ABC            │
                 │  (vendored          (funasr AutoModel                                   │
                 │   third_party/       + 独立 FSMN-VAD)                                    │
                 │   CosyVoice)                                                             │
                 └──────────────────────────────────────────────────────────────────────────┘
```

## 关键决策（ADR 摘要）

### A1. 全链路纯 PyTorch —— 排除 faster-whisper / onnxruntime-gpu
生产机 RTX 5070 是 Blackwell（sm_120），仅 PyTorch ≥2.7(cu128) 官方支持；
CTranslate2 预编译 kernel 长期最高 sm_90，ONNX Runtime GPU 对 sm_120 支持不确定。
SenseVoice/CosyVoice 均为 PyTorch 实现，一套运行时通吃。CosyVoice 内含的两个 ONNX
组件（speech_tokenizer/campplus）被仓库硬编码为 **CPUExecutionProvider**，无 GPU 内核依赖。

### A2. torch 钉 2.7.1，不追新
首个带官方 cu128 稳定 wheel 的版本。2.8+/torchaudio 2.9 进入 deprecated 迁移
（功能拆往 torchcodec），而 CosyVoice/FunASR 依赖 torchaudio 经典 API。

### A3. GPU 全局信号量 = 1（双模型常驻、分时计算）
8GB 显存放得下两个模型的权重（≈5.5–6.5GB 稳态），但放不下降级余量下的并发激活。
常驻避免反复加载（加载一次要几十秒），信号量=1 保证任何时刻只有一个模型在算，
瞬时显存叠加不会击穿。这是"高可用 > 高吞吐"的取舍：单机吞吐上限 ≈ 1/RTF。

### A4. SenseVoice 配独立 FSMN-VAD，手动分段
不用 funasr 内置 vad-merge：其输出格式随版本漂移，且 SenseVoice 无字级时间戳。
手动"VAD 出边界 → 切块 → 逐块转写"使段级时间戳由我们的代码保证，可单测。
字级时间戳升级路径：新增 paraformer-zh 引擎（同生态、带 timestamp predictor），
实现 STTEngine ABC 即可，API 已预留 `timestamps` 语义分级。

### A5. 引擎同步方法 + Guard 线程化
引擎 `synthesize/transcribe` 是同步阻塞的（模型天然如此），asyncio 适配全部收敛在
`InferenceGuard`。好处：引擎可被脚本/测试直接同步调用；将来换 ProcessPool 不碰引擎。

### A6. 音频层：PyAV 为主干（不依赖系统 ffmpeg）
PyAV wheel 自带 FFmpeg（解码 m4a/opus/webm + 编码 mp3），soundfile 负责
wav/flac/ogg 快捷路径，soxr 重采样。规避 pydub（要系统 ffmpeg）与 librosa（依赖重）。
Windows 生产机零系统依赖。

### A7. 中文 TN：wetext（可选件），降级链内置
CosyVoice 新仓库已弃用 WeTextProcessing/pynini（Windows 大坑），改用 `wetext`
（kaldifst 预编译 wheel，pip 通吃 win/mac/linux）；其 FST 词表首次使用时联网拉取。
三层降级：wetext → vendored frontend 基础处理 → 我们的 regex TN
（`app/utils/textnorm.py`，数字/百分比/年份/长号）。

### A8. 模型生命周期
lifespan 后台预加载（服务秒起，加载中业务端点 503 `model_loading` + Retry-After），
Event 门闩合并并发等待者；懒加载模式（PRELOAD_ENGINES=false）复用同一门闩；
加载后 warmup 一次哑推理预分配 cuDNN workspace。

## Vendoring（third_party/CosyVoice）

- git submodule，基线 commit：`074ca6d`（2025-xx，`Add FunAudioLLM ecosystem section`）
- 不 pip 安装其 requirements.txt（torch/deepspeed/tensorrt 等钉版冲突源），
  等价运行时依赖收敛在 `requirements/base.txt`
- sys.path 注入点在 `engines/tts_cosyvoice2._bootstrap_path()`（含 Matcha-TTS 子路径）
- **升级流程**：`git submodule update --remote` → 双平台跑
  `pytest -m model` + `smoke_test.py` → 更新此处基线 commit
- 已知无需打补丁：TN 缺失降级、fp16 无 CUDA 自动关、ONNX 强制 CPU provider 均为仓库内建

## 流式（v0.2 预留）

`TTSEngine.synthesize_stream()` 已占位。CosyVoice2 本身支持 `stream=True` 逐块产出、
FastAPI 侧可走 chunked/SSE；STT 侧流式需换 paraformer-streaming 或 SenseVoice+
WebSocket，届时新增 `GET /ws` 而非改造 REST。

## 扩展路线

- paraformer-zh（字级时间戳）、CosyVoice3（多语种/方言，核许可与显存后接入）
- 多实例横向扩展：每台机一个服务进程（信号量=1 是单机吞吐上限），前置 nginx 轮询
- 可选增强：`TTS_STT_SPLIT_DEVICE`（TTS 用 GPU、STT 钉 CPU 的混合模式，进一步压显存）
