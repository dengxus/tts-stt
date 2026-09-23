# Windows 生产部署（RTX 5070 / 8GB）

## 0. 前置

- NVIDIA 驱动 ≥ 572（RTX 50 系 Blackwell 所需最低驱动线），`nvidia-smi` 正常输出
- Python **3.10**（与 CosyVoice 官方验证版本一致；全团队统一版本）
- 磁盘 ≥ 10GB（依赖 ~5GB + 模型 ~3GB + 缓存）

## 1. 安装（顺序不可变！）

```powershell
cd tts-stt
git submodule update --init --recursive --depth 1   # CosyVoice + Matcha-TTS
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# ① 先装 cu128 torch —— 若先装 base.txt，pip 可能拉默认 CUDA 版 torch 覆盖
pip install -r requirements\torch-win-cu128.txt

# ② 再装业务依赖
pip install -r requirements\base.txt

# ③ 验收（必须看到 capability=(12, 0) 且 arch_list 含 sm_120）
python scripts\check_env.py
```

> 报错 `CUDA error: no kernel image is available for execution on the device`
> ＝ torch 不是 cu128 版，回到第①步重装。

## 2. 下载模型（约 3GB）

```powershell
python scripts\download_models.py --only all
```

- 主通道 ModelScope（国内直连）；失败时换 `--source hf --hf-endpoint https://hf-mirror.com`
- **离线机器**：在能上网的同平台机器执行一次，然后拷贝整个 `models/` 目录过来
- wetext 的中文 TN 词表（FST，约 60MB）会在**首次启动 TTS 引擎时**从 modelscope 下载
  到 `C:\Users\<user>\.cache\modelscope\models\pengzhendong--wetext\`；离线机器请在联网
  环境跑一次 `python -c "from wetext import Normalizer; Normalizer()"` 后把该缓存目录一并拷过来。
  没有它也仅影响数字/日期读法（服务自动降级为内置轻量 TN），不影响可用性。

## 3. 启动

```powershell
scripts\run_windows.bat
# 或手工：
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

- **必须 `--workers 1`**：模型驻留进程内，多 worker 会复制显存导致 OOM
- 验证：`curl http://localhost:8000/health/ready` → 200；
  `python scripts\smoke_test.py`（TTS→STT 回环）
- 防火墙放行 8000 端口（内网）

## 4. 注册为 Windows 服务（NSSM）

```powershell
# 下载 nssm: https://nssm.cc/download ，解压到 C:\nssm
nssm install tts-stt "C:\path\to\tts-stt\scripts\run_windows.bat"
nssm set tts-stt AppDirectory "C:\path\to\tts-stt"
nssm set tts-stt AppExit Default Restart
nssm set tts-stt AppRestartDelay 5000
nssm set tts-stt DisplayName "TTS/STT 基础服务"
nssm start tts-stt
```

- 服务以 SYSTEM 账户运行时 `nvidia-smi`/CUDA 可用，但 `~` 指向 `C:\Windows\System32`；
  建议 `nssm set tts-stt ObjectName ".\你的用户"` 或配置
  `MODEL_DIR`/`HOME` 环境变量让 wetext 缓存可写。
- 崩溃自动重启 + `/health/live` 探活已内置。

## 5. 显存与并发纪律（8GB 双模型常驻）

| 项 | 值 |
|---|---|
| 稳态显存（两模型常驻） | ≈5.5–6.5GB |
| TTS | fp16（自动，CUDA 上生效） |
| STT | fp32（SenseVoice 仅 ~1.3GB，不值当冒 half 兼容风险） |
| GPU 推理并发 | 全局信号量 = 1（TTS/STT 分时用卡），勿调大 |
| 上限 | 文本 ≤5000 字，音频 ≤10min/50MB |

**OOM 处置链**（自动 + 手动）：
1. 偶发 OOM → 请求返回 503 `inference_failed`，服务不崩，降低单请求长度即可
2. 持续 OOM（桌面程序抢显存）→ `.env` 设 `DEVICE=cpu` 只降 STT：SenseVoice CPU
   RTF 仍 <0.2（把 `STT_ENGINE` 保持 sensevoice，代码在 cuda 上才用 GPU；如需强制
   STT 走 CPU 而 TTS 走 GPU 的混合模式，设 `TTS_STT_SPLIT` 开发项见 roadmap）
3. 最后手段：只部署 TTS（`STT_ENGINE=none`，未注册即返回 503）或反之

## 6. 常见故障排查

| 症状 | 原因与处置 |
|---|---|
| `no kernel image is available` | torch 非 cu128，重装 `torch-win-cu128.txt` |
| TTS 合成中文数字读法不对 | wetext FST 没下载成功（离线机器），见 §2；或检查启动日志 `use wetext frontend` |
| TTS 音频时长异常膨胀、内容为乱码 | transformers 装到 4.52+，破坏 CosyVoice2 LLM 的 speech token 生成（flow/hift 正常）。锁定 `requirements/base.txt` 的 `transformers>=4.51,<4.52` 重装 |
| 日志出现 `no frontend is avaliable` | wetext 未装/加载失败 → 已自动降级内置 regex TN |
| 首次 TTS 特别慢 | 冷启动 warmup 前的首个请求；确认日志有 warmup 完成记录 |
| 503 model_loading 持续 | 模型未下载完整：重跑 download_models.py（有校验与重试） |
| STT 返回 415 | 上传的不是真音频（扩展名撒谎）或用了不支持的编码 |
| 端口 10048 起不动 | 上一次实例没退干净：`Get-Process python \| Stop-Process` |
| Windows 更新后 GPU 失效 | 驱动回滚/重装，`python scripts\check_env.py` 重新验收 |

## 7. 版本回退

全部依赖版本下限见 `requirements/`。若 CosyVoice vendored 代码与
torch 2.7.1 出现不兼容（升级 vendored commit 后尤甚），按
`docs/architecture.md §vendoring` 记录的基线 commit 回退：

```powershell
cd third_party\CosyVoice
git checkout <baseline-commit>
```
