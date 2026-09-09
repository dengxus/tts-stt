@echo off
REM ============================================================
REM tts-stt Windows 生产启动脚本（RTX 5070 / cu128 环境）
REM 用法：双击运行，或 NSSM 里把 Startup script 指到本文件
REM ============================================================
setlocal
cd /d %~dp0..

if not exist .venv\Scripts\activate.bat (
  echo [FAIL] 未找到 .venv，请先按 docs/deployment-windows.md 安装环境
  exit /b 1
)
call .venv\Scripts\activate.bat

set PYTHONUNBUFFERED=1

REM 启动前环境验收：capability=(12,0) + arch_list 含 sm_120
python scripts\check_env.py || (echo [FAIL] 环境检查未通过 & exit /b 1)

echo [INFO] 启动 tts-stt (workers=1，模型常驻显存)...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
