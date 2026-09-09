# API 参考

Base URL：`http://<host>:8000`　前缀：`/api/v1`　交互式文档：`GET /docs`（Swagger UI）

鉴权：`API_KEY` 环境变量非空时，`/api/v1/*` 需携带 `Authorization: Bearer <API_KEY>`；
`/health*` 探针不鉴权。

统一错误体：

```json
{"error": {"code": "queue_full", "message": "推理队列已满，请稍后重试", "request_id": "..."}}
```

| HTTP | code | 含义 |
|---|---|---|
| 400 | invalid_request | 参数非法（音色不存在、克隆未开放、超长文本等） |
| 401 | unauthorized | Bearer token 缺失/错误 |
| 413 | payload_too_large | 音频文件超限（字节/时长） |
| 415 | unsupported_format | 音频无法解码（格式不支持） |
| 422 | validation_error | 请求体字段校验失败 |
| 429 | queue_full | 推理排队已满（带 `Retry-After`） |
| 503 | model_loading / engine_error / inference_failed | 模型加载中 / 加载失败 / 推理异常（含 OOM） |
| 504 | inference_timeout | 单次推理超时 |

---

## POST /api/v1/tts —— 文本转语音

请求体（JSON）：

```json
{
  "text": "你好，欢迎使用语音合成服务。",   // 必填，≤5000 字符
  "voice": "中文女",                       // 可选，见 GET /api/v1/voices；默认取 DEFAULT_VOICE 或第一个预置音色
  "format": "wav",                         // wav(默认) | mp3 | ogg | flac
  "sample_rate": 24000,                    // 可选 8000|16000|22050|24000|44100|48000；默认模型原生 24000
  "speed": 1.0,                            // 0.5 ~ 2.0
  "instruct": "用开心的语气说",             // 可选，触发语气指令模式（instruct2）
  "zero_shot": {                           // 可选，声音克隆；需服务端 ALLOW_ZERO_SHOT=true
    "prompt_text": "参考音频的转写文本",
    "prompt_audio_base64": "<≤15s 参考音频 base64>"
  },
  "response_format": "audio"               // audio(默认，二进制) | json(base64)
}
```

模式优先级：`zero_shot > instruct > voice(sft 预置)`。

**响应（response_format=audio）**：二进制音频，`Content-Type: audio/wav|audio/mpeg|...`，
响应头：

| 头 | 说明 |
|---|---|
| `X-Duration-Ms` | 音频时长 |
| `X-Sample-Rate` | 输出采样率 |
| `X-Inference-Ms` | 推理耗时（不含编解码） |
| `X-RTF` | 实时率 = 推理耗时/音频时长，<1 即快于实时 |
| `X-Engine` / `X-Device` | cosyvoice2 / cuda·mps·cpu |
| `X-Chunks` | 长文本自动分块数 |

**响应（response_format=json）**：

```json
{"audio_base64": "...", "format": "wav", "sample_rate": 24000,
 "duration_ms": 3120, "inference_ms": 890, "engine": "cosyvoice2"}
```

curl 示例：

```bash
# 基本合成（预置音色，输出 wav 文件）
curl -X POST localhost:8000/api/v1/tts -H 'Content-Type: application/json' \
  -d '{"text":"你好世界"}' -o out.wav

# 指定音色 + mp3 + 语速
curl -X POST localhost:8000/api/v1/tts -H 'Content-Type: application/json' \
  -d '{"text":"Hello world","voice":"英文女","format":"mp3","speed":1.2}' -o out.mp3

# 语气指令
curl -X POST localhost:8000/api/v1/tts -H 'Content-Type: application/json' \
  -d '{"text":"今天真是个好日子！","instruct":"用非常开心的语气说","speed":1.0}' -o happy.wav
```

## POST /api/v1/stt —— 语音转文本

`multipart/form-data`：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `file` | 文件 | 必填 | wav/mp3/m4a/ogg/opus/flac/webm；≤50MB 且 ≤600s |
| `language` | str | auto | auto/zh/en/ja/ko/yue（指定语种可提高精度） |
| `use_itn` | bool | true | 数字/日期规范化（“十二点五”→“12.5”） |
| `timestamps` | bool | true | 返回段级时间戳（FSMN-VAD 分段） |
| `output_rich` | bool | false | 返回情绪/事件标签（SenseVoice 特有） |

响应：

```json
{
  "text": "今天天气不错，我们去公园走走吧。",
  "language": "zh",
  "duration_ms": 3240,
  "segments": [{"start_ms": 120, "end_ms": 3200, "text": "今天天气不错，我们去公园走走吧。"}],
  "emotion": "NEUTRAL",
  "events": ["Speech"],
  "engine": "sensevoice-small",
  "itn": true,
  "inference_ms": 210,
  "rtf": 0.065
}
```

> **已知限制**：SenseVoice 为非自回归模型，无**字级**时间戳，`segments` 为 VAD
> 段级边界（误差 ~百毫秒）；需要字级时间戳时规划中的 `paraformer-zh` 引擎（v0.2）。

```bash
curl -F "file=@record.m4a" localhost:8000/api/v1/stt
curl -F "file=@song.mp3" -F "language=en" -F "output_rich=true" localhost:8000/api/v1/stt
```

## GET /api/v1/voices

```json
{"voices": [
  {"id": "中文女", "name": "中文女", "engine": "cosyvoice2", "mode": "sft",
   "languages": ["zh", "en"], "sample_rate": 24000, "tags": ["preset"]},
  {"id": "__instruct__", "name": "语气指令（任意音色+instruct）", "mode": "instruct2",
   "tags": ["instruct"], "default_instruct": "用开心的语气说"}
]}
```

`mode=sft` 的 `id` 直接用作 TTS 请求的 `voice` 字段。实际清单以运行时权重内
`spk2info` 为准。

## GET /api/v1/engines

各引擎状态与能力：`{"device": "cuda", "tts": {"name": "cosyvoice2", "state": "ready",
"precision": "fp16", "wetext_available": true}, "stt": {...}}`

## GET /health | /health/live | /health/ready

- `/health`：完整状态（设备、各引擎 state、队列深度、运行时长）
- `/health/live`：进程存活（200 即正常）
- `/health/ready`：预加载引擎全部就绪才 200（否则 503 + Retry-After），供服务管理器/负载均衡探活

引擎 `state`：`unloaded | loading | ready | degraded | error`

## 行为约定

- **长文本**：>800 字自动按标点分块合成、块间 80ms 静音拼接，对客户端透明（`X-Chunks` 标注块数）
- **并发**：GPU 推理同时只跑一个（TTS/STT 分时），排队上限 8，超出 429
- **加载期**：服务秒起，模型后台加载；加载完成前业务端点返回 503 `model_loading`
- **流式接口**：规划中（v0.2），引擎层 `synthesize_stream` 已预留
