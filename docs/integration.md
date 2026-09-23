# 接口对接指南

面向业务方/前端的对接文档：怎么调、怎么处理错误、怎么踩准并发纪律。
字段级完整参考见 [api.md](api.md)，交互式 OpenAPI 文档见 `GET /docs`。

## 0. 速览

| 项 | 值 |
|---|---|
| Base URL | `http://<host>:8000`，业务前缀 `/api/v1` |
| 鉴权 | 服务端 `API_KEY` 非空时，业务请求带 `Authorization: Bearer <API_KEY>`；`/health*` 不鉴权 |
| 请求 ID | 每个响应都带 `X-Request-ID`；请求方可自带该头，服务端原样透传（排查日志时对账用） |
| CORS | 默认 `*`（浏览器直连可用）；Swagger `/docs` 与 playground `/playground` 不鉴权，公网部署请挂网关收口 |
| 编码 | 请求/响应 JSON 均为 UTF-8 |

## 1. 对接第一步：探就绪

服务秒起、模型后台加载（TTS ~10s / STT ~8s）。**加载完成前业务端点返回 503**，
客户端启动后先轮询就绪探针：

```bash
curl -s -o /dev/null -w "%{http_code}" http://<host>:8000/health/ready
# 200 = 业务可用；503 = 加载中（响应带 Retry-After: 5）
```

K8s/NSSM/负载均衡探针建议：存活 `/health/live`，就绪 `/health/ready`。

## 2. TTS 快速上手

`POST /api/v1/tts`，JSON 请求体，二进制或 base64 音频响应。

### curl

```bash
# 最小请求（默认音色、wav）
curl -X POST http://<host>:8000/api/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，欢迎使用语音服务。"}' \
  -o out.wav -D resp.hdr
# resp.hdr 里的 X-Duration-Ms / X-RTF 可用于监控与计费口径

# 指定音色 + mp3 + 倍速
curl -X POST http://<host>:8000/api/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world","voice":"default","format":"mp3","speed":1.2}' \
  -o out.mp3
```

### Python

```python
import requests

BASE, HDR = "http://<host>:8000", {"Content-Type": "application/json"}

def tts(text: str, *, voice: str | None = None, fmt: str = "wav",
        speed: float = 1.0, timeout: float = 150) -> bytes:
    r = requests.post(f"{BASE}/api/v1/tts", headers=HDR, timeout=timeout, json={
        "text": text, "voice": voice, "format": fmt, "speed": speed,
    })
    r.raise_for_status()
    return r.content

open("out.wav", "wb").write(tts("你好，世界。"))
```

### 浏览器 JS

```js
const resp = await fetch("http://<host>:8000/api/v1/tts", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "你好，世界。", format: "mp3" }),
});
if (!resp.ok) throw new Error((await resp.json()).error.message);
document.getElementById("audio").src = URL.createObjectURL(await resp.blob());
```

### 关键参数

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `text` | str | 必填 | 1–5000 字符，纯空白返回 422 |
| `voice` | str | `default` | 见 `GET /api/v1/voices`；未知音色 400 |
| `format` | str | `wav` | `wav`(PCM_16) / `mp3`(192kbps mono) / `ogg` / `flac` |
| `sample_rate` | int | 24000 | 8000/16000/22050/24000/44100/48000，其它 422 |
| `speed` | float | 1.0 | 0.5–2.0 |
| `seed` | int | — | 随机种子：固定后同一文本可复现；偶发读音错误换个种子重试 |
| `instruct` | str | — | 语气指令（instruct2 模式），如 `"用开心的语气说"` |
| `zero_shot` | obj | — | 声音克隆，需服务端 `ALLOW_ZERO_SHOT=true`（默认 false，开 400） |
| `response_format` | str | `audio` | `audio`=二进制；`json`=`audio_base64` 字段 |

**三种模式互斥，优先级 `zero_shot` > `instruct` > `voice`**：

- 预置音色：`{"voice":"default"}`（本部署的打包音色，中英双语）
- 语气指令：`{"instruct":"用开心的语气说"}`（音色取服务端参考音，`voice` 被忽略）
- 声音克隆：`{"zero_shot":{"prompt_text":"参考音频的转写文本","prompt_audio_base64":"..."}}`
  ——参考音频 5–15s、清晰单人朗读效果最佳（时长为建议值，服务端不强制）。

### 响应

二进制模式（`response_format=audio`）返回音频字节流，响应头带计量信息
（**json 模式没有这些头**，需要 RTF/时长监控请用二进制模式读头，或读 json 体字段）：

| 头 | 说明 |
|---|---|
| `X-Duration-Ms` / `X-Sample-Rate` | 音频时长 / 输出采样率 |
| `X-Inference-Ms` / `X-RTF` | 推理耗时 / 实时率（<1 快于实时） |
| `X-Chunks` | 长文本自动分块数（>800 字按标点切块、块间 80ms 静音拼接，对调用方透明） |
| `X-Engine` / `X-Device` / `X-Request-ID` | 引擎 / 设备 / 链路追踪 |

`response_format=json` 时响应体：

```json
{"audio_base64": "...", "format": "wav", "sample_rate": 24000,
 "duration_ms": 3120, "inference_ms": 890, "engine": "cosyvoice2"}
```

### 声音克隆

两种方式，按场景选：

| 方式 | 适用场景 | 调用方式 |
|---|---|---|
| **按请求克隆**（zero_shot） | 参考音频由客户端每次上传、音色不固定 | 请求体带 `zero_shot` 对象 |
| **注册为命名音色** | 固定音色反复使用（特征只算一次并落盘缓存） | `assets/prompts/` 放文件 + 重启，之后按 `voice` 调用 |

**方式 A：按请求克隆**（需服务端 `ALLOW_ZERO_SHOT=true`，默认关闭）

```python
import base64

ref = base64.b64encode(open("ref.wav", "rb").read()).decode()
r = requests.post(f"{BASE}/api/v1/tts", json={
    "text": "要合成的文本",
    "zero_shot": {
        "prompt_text": "参考音频的逐字转写（必须与音频内容一致，含标点）",
        "prompt_audio_base64": ref,
    }, "format": "mp3"})
```

- 参考音频 5–15s、安静无混响的单人朗读效果最佳；wav/16k+ 单声道优先（其它格式服务端自动解码）
- `prompt_text` 与音频**逐字一致**是克隆质量第一影响因素——转写错一个字，合成音会跟着错
- 开关未开时返回 400 `invalid_request`；公网开启后务必配合 `API_KEY` 鉴权

**方式 B：注册为命名音色**（无需开克隆开关，无需改代码）

```
assets/prompts/<音色id>.wav   # ≤15s 参考音频
assets/prompts/<音色id>.txt   # 该音频的逐字转写（一行）
```

重启后 `GET /api/v1/voices` 即出现该 id，之后 `"voice": "<音色id>"` 与预置音色用法
完全一致。特征仅首次启动时提取并缓存进 `models/CosyVoice2-0.5B/spk2info.pt`；
**更新 wav 后必须删除该缓存文件**（或改用新文件名）再重启，否则不会重新注册。

**合规红线**：参考音频必须是本人录音或已获明确授权/CC0 素材，克隆他人声音前先取得同意。

### 读音不准的处置（中英混读常见）

按问题类型对症下药，代价从低到高：

1. **偶发读错**（同文本时对时错）：LLM 采样有随机性——传 `seed` 复现问题，换种子重试通常即解
2. **多音字/专有名词读错**：改写文本最有效（换同义表达、补语境）；难读的外文名可用谐音字标注
3. **英文走调**：参考音频是什么语种，英文发音就越靠近它——中文人声参考下英文变差是克隆固有现象；
   双语场景可分别注册中/英文参考音色，按文本语种路由
4. **长文本后段跑调/漏读**：调小服务端 `TTS_CHUNK_CHARS`（如 800→200）让分块更细
5. **系统性不满意**：评估 CosyVoice3 升级（v0.2 路线图）

## 3. STT 快速上手

`POST /api/v1/stt`，`multipart/form-data` 上传。

### curl

```bash
curl -F "file=@record.m4a" http://<host>:8000/api/v1/stt
curl -F "file=@meeting.mp3" -F "language=zh" -F "output_rich=true" \
     http://<host>:8000/api/v1/stt
```

### Python

```python
def stt(path: str, language: str = "auto") -> dict:
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/api/v1/stt",
                          files={"file": f},          # 文件名后缀请真实
                          data={"language": language, "use_itn": "true"},
                          timeout=150)
    r.raise_for_status()
    return r.json()
```

### 浏览器录音上传（MediaRecorder 直接可用）

```js
const form = new FormData();
form.append("file", blob, "record.webm");   // webm/opus 服务端原生支持
form.append("language", "auto");
const { text, segments } = await (await fetch(".../api/v1/stt",
  { method: "POST", body: form })).json();
```

### 表单字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `file` | 必填 | wav/mp3/m4a/ogg/opus/flac/webm；≤50MB 且 ≤600s（解码后校验时长） |
| `language` | `auto` | auto/zh/en/ja/ko/yue；已知语种时指定可提升精度 |
| `use_itn` | `true` | 数字/日期倒格式化（"十二点五"→"12.5"） |
| `timestamps` | `true` | 返回段级时间戳 |
| `output_rich` | `false` | 附加情绪（`emotion`）与事件（`events`）标签 |

### 响应语义（注意边界情况）

```json
{"text": "...", "language": "zh", "duration_ms": 3240,
 "segments": [{"start_ms": 120, "end_ms": 3200, "text": "..."}],
 "emotion": "NEUTRAL", "events": [], "engine": "sensevoice-small",
 "itn": true, "inference_ms": 210, "rtf": 0.065}
```

- `segments` 是 **VAD 段级**边界（±百毫秒），不是字级；<5s 的短音频跳过 VAD、整段返回
- `language` 传 `auto` 时返回检出语种；**完全未检出时为 `"und"`**
- 静音/纯噪声返回 HTTP 200 + `text:""`、`segments:[]`——不是错误，按空结果处理
- 情绪/事件标签仅 `output_rich=true` 时非空

## 4. 错误处理与重试策略

统一错误体（`422` 与业务错误同构）：

```json
{"error": {"code": "queue_full", "message": "推理队列已满，请稍后重试",
           "request_id": "7a4428eaa7da4eaa"}}
```

| HTTP | code | 触发场景 | 客户端处置 |
|---|---|---|---|
| 400 | `invalid_request` | 音色不存在 / 克隆未开放 / 文本超 5000 字 / language 非法 | **不重试**，修正参数 |
| 401 | `unauthorized` | Bearer 缺失或错误 | 检查 key，不重试 |
| 413 | `payload_too_large` | 文件 >50MB 或时长 >600s | 不重试，切分音频 |
| 415 | `unsupported_format` | 无法解码（扩展名撒谎/加密文件） | 不重试，转码后重传 |
| 422 | `validation_error` | 字段校验失败（message 含具体字段） | 不重试 |
| 429 | `queue_full` | 排队超上限（8） | **按 `Retry-After`(2s) 退避重试**，建议叠加抖动 |
| 503 | `model_loading` | 模型加载中 | 等就绪探针 200 后重试（`Retry-After: 5`） |
| 503 | `inference_failed` | 推理异常（服务端已含一次 OOM 自动重试） | 可重试 1 次；持续出现联系运维 |
| 503 | `engine_error` | 引擎未注册/加载失败 | 联系运维，不重试 |
| 504 | `inference_timeout` | 单次推理超时（默认 120s） | **勿立即原样重发**：服务端线程仍在跑完该请求，等 3–5s 或减小输入 |
| 500 | `internal_error` | 未预期异常 | 带 `X-Request-ID` 找运维 |

Python 侧一个够用的重试模板（只对可重试错误退避）：

```python
import time, requests

RETRYABLE = {429, 503, 504}

def post_with_retry(fn, *, tries: int = 3):
    for i in range(tries):
        try:
            r = fn()
            if r.status_code in RETRYABLE and i < tries - 1:
                time.sleep(float(r.headers.get("Retry-After", 2 * (i + 1))))
                continue
            r.raise_for_status()
            return r
        except requests.ConnectionError:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
```

## 5. 并发纪律（重要）

服务端为保 8GB 级显存稳定，**GPU 推理全局串行（TTS/STT 分时）+ 队列上限 8**：

- 客户端并发请控制在 **≤2**，更多请求自行本地排队；猛打并发只会换来 429
- 单请求文本 ≤5000 字（服务端自动分块）；需要批量合成时按条循环串行调用
- 请求超时上限默认 120s，客户端超时请设 **>120s**（否则客户端先断，服务端白算）

## 6. 联调自检

```bash
python scripts/smoke_test.py --base-url http://<host>:8000 [--api-key <key>]
```

跑 TTS→STT 回环（中/英/数字三句 + instruct 模式），全部 `[ok]` 即对接环境就绪。
浏览器手工联调直接访问 `http://<host>:8000/playground`。

## 7. 路线图

- v0.2：SSE/chunked 流式合成（首包延迟显著下降）、Paraformer 字级时间戳引擎
- 流式未上线前，长文本以分块机制兜底，延迟 ≈ 各块推理时间之和
