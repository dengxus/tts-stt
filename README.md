# tts-stt —— 开源文本转语音 / 语音转文本基础服务

基于 FastAPI 的单体语音服务，全部采用开源模型：

| 能力 | 模型 | 许可证 | 说明 |
|---|---|---|---|
| TTS | [CosyVoice2-0.5B](https://github.com/FunAudioLLM/CosyVoice) | Apache-2.0 | 中英双语，预置音色 + 语气指令 + zero-shot 克隆 |
| STT | [SenseVoice-Small](https://github.com/FunAudioLLM/SenseVoice) + FSMN-VAD | MIT | 中英日韩粤，标点/ITN/语种/情绪，段级时间戳 |

架构与选型的完整决策记录见 `docs/`。

## 环境与部署矩阵

| | 开发机 (macOS) | 生产机 (Windows + RTX 5070) |
|---|---|---|
| Python | 3.10（.venv） | 3.10 |
| torch | 2.7.1（MPS/CPU, fp32） | **2.7.1+cu128**（Blackwell sm_120 必需, fp16） |
| 显存 | 统一内存双模型 ~5.5GB | 双模型常驻 ≈5.5–6.8GB / 8GB（GPU 推理并发=1 分时） |

> RTX 50 系（sm_120）**必须**使用 cu128 版 torch，且推理栈全部走纯 PyTorch
> （已刻意排除依赖 CTranslate2 的 faster-whisper、依赖 onnxruntime-gpu 的方案）。

## 快速开始（macOS 开发）

```bash
git submodule update --init --recursive --depth 1   # CosyVoice + Matcha-TTS（TTS 引擎源码）
uv venv --python 3.10 && source .venv/bin/activate
pip install -r requirements/torch-mac.txt -r requirements/base.txt -r requirements/dev.txt
# macOS 若 pyworld 编译报 'algorithm' file not found：Command Line Tools 的 C++ 头文件残缺，
# export SDKROOT=$(xcrun --show-sdk-path) CXXFLAGS="-isystem $SDKROOT/usr/include/c++/v1" 后重装，
# 或 sudo xcode-select --install 重装 CLT
python scripts/check_env.py                       # 环境自检
python scripts/download_models.py --only stt      # 先只下 STT（~1GB）
cp .env.example .env
python -m uvicorn app.main:app --port 8000 --workers 1
```

冒烟验证：

```bash
curl -s localhost:8000/health | python -m json.tool

# TTS
curl -s -X POST localhost:8000/api/v1/tts -H 'Content-Type: application/json' \
     -d '{"text":"你好，这是一次冒烟测试。","format":"mp3"}' -o out.mp3 && afplay out.mp3

# STT
curl -s -F "file=@out.mp3" localhost:8000/api/v1/stt
```

## Windows 生产部署（RTX 5070）

```powershell
py -3.10 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements\torch-win-cu128.txt   # 必须最先装，锁定 cu128
pip install -r requirements\base.txt
python scripts\check_env.py   # 验收: capability=(12,0) 且 arch_list 含 sm_120
```

CosyVoice 中文文本正则化（pynini）在 Windows 的安装策略 A/B/C，以及 NSSM 服务化、
显存调优、降级链，见 [docs/deployment-windows.md](docs/deployment-windows.md)。

## API 一览

| 端点 | 说明 |
|---|---|
| `POST /api/v1/tts` | 文本 → 音频（wav/mp3/ogg/flac；二进制或 base64 JSON） |
| `POST /api/v1/stt` | 音频上传（wav/mp3/m4a/ogg/opus/flac/webm）→ 文本+时间戳 |
| `GET /api/v1/voices` | TTS 音色列表 |
| `GET /api/v1/engines` | 引擎能力与状态 |
| `GET /health` `/health/live` `/health/ready` | 状态 / 存活 / 就绪探针 |

请求/响应字段、错误码（429 队列满 / 503 加载中 / 504 超时 / 413 / 415）与 curl 示例全集
见 [docs/api.md](docs/api.md)；交互式文档启动后访问 `/docs`。

## 并发模型（8GB 显存下的关键纪律）

- `uvicorn --workers 1`：模型驻留进程内，多 worker = 双份显存
- 全局 GPU 推理信号量 = 1：TTS/STT 分时用卡（模型常驻显存，计算互斥）
- 排队上限 `MAX_QUEUE=8`，超出 429；单请求超时 `REQUEST_TIMEOUT_S=120` 返回 504
- 文本 ≤5000 字（自动分块拼接）、音频 ≤10 分钟 / ≤50MB

## 开发与测试

```bash
pytest -m "unit or api"          # 无模型快速回归（mock 引擎）
pytest -m model                  # 需要已下载权重的真实模型测试
ruff check app tests scripts
```

## 路线图

- v0.1（本期）：REST TTS/STT 全链路 + Windows 生产化
- v0.2：SSE/chunked 流式合成（引擎 `synthesize_stream` 已预留）、
  Paraformer 字级时间戳引擎、CosyVoice3 升级评估
