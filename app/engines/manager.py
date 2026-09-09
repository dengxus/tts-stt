"""EngineManager：引擎注册 / 后台预加载 / 懒加载门闩 / InferenceGuard 并发控制。

并发模型（8GB 显存双模型常驻的关键）：
    请求 → InferenceGuard：
        ├─ 等待数 ≥ MAX_QUEUE → 立即 429 + Retry-After
        ├─ 全局 GPU 信号量（默认 1：TTS 与 STT 分时用 GPU）
        │     └─ 引擎互斥锁（模型非线程安全）
        │           └─ asyncio.to_thread + 超时 → 504
        └─ 线程内 OOM → InferenceFailedError（降级策略由引擎 load/run 时处理）
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from app.config import Settings
from app.engines.base import EngineState, STTEngine, TTSEngine
from app.errors import (
    EngineError,
    InferenceFailedError,
    InferenceTimeoutError,
    ModelLoadingError,
    QueueFullError,
)
from app.utils.device import check_blackwell_compat, detect_device, resolve_fp16

logger = logging.getLogger(__name__)

KINDS = ("tts", "stt")


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower()


class InferenceGuard:
    """全局推理并发守卫：信号量 + 排队上限 + 超时。"""

    def __init__(self, max_concurrent: int, max_queue: int, timeout_s: float):
        self._max_concurrent = max_concurrent
        self._max_queue = max_queue
        self._timeout_s = timeout_s
        self._sem: asyncio.Semaphore | None = None
        self._waiters = 0

    @property
    def stats(self) -> dict:
        return {
            "gpu_sem": self._max_concurrent,
            "waiters": self._waiters,
            "max_queue": self._max_queue,
        }

    async def run(self, sync_fn: Callable[[], Any]) -> Any:
        if self._sem is None:  # 延迟到运行中的 loop 内创建
            self._sem = asyncio.Semaphore(self._max_concurrent)
        if self._waiters >= self._max_queue:
            raise QueueFullError()
        self._waiters += 1
        try:
            for attempt in (0, 1):
                async with self._sem:
                    try:
                        return await asyncio.wait_for(
                            asyncio.to_thread(sync_fn), self._timeout_s
                        )
                    except asyncio.TimeoutError as e:
                        # 注意：to_thread 无法真正取消，线程仍会跑完并持有引擎锁
                        logger.error("推理超时（%.0fs），后台线程仍持有引擎锁直至结束",
                                     self._timeout_s)
                        raise InferenceTimeoutError(
                            f"推理超过 {self._timeout_s:.0f}s 超时"
                        ) from e
                    except Exception as e:  # noqa: BLE001
                        if _is_oom(e) and attempt == 0:
                            # 显存碎片/瞬时挤占常见：清缓存后串行重试一次
                            logger.warning("推理 OOM，empty_cache() 后重试一次: %s", e)
                            try:
                                import torch

                                torch.cuda.empty_cache()
                            except Exception:  # noqa: BLE001
                                pass
                            continue
                        if _is_oom(e):
                            logger.error("推理 OOM（重试仍失败）: %s", e)
                            raise InferenceFailedError(
                                f"显存不足(OOM): {e}") from e
                        logger.exception("推理异常")
                        raise InferenceFailedError(f"推理失败: {e}") from e
        finally:
            self._waiters -= 1


class EngineManager:
    def __init__(self, settings: Settings):
        self._settings = settings
        self.device = detect_device(settings.device)
        self.fp16 = resolve_fp16(settings.fp16, self.device)
        for w in check_blackwell_compat(self.device):
            logger.warning(w)
        logger.info("推理设备=%s，fp16=%s", self.device, self.fp16)

        self.guard = InferenceGuard(
            settings.max_concurrent_infer, settings.max_queue,
            settings.request_timeout_s,
        )
        self._engines: dict[str, TTSEngine | STTEngine] = {}
        self._states: dict[str, EngineState] = {}
        self._errors: dict[str, str] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._load_tasks: dict[str, asyncio.Task] = {}
        self._started_at = time.monotonic()

    # ---------- 注册与加载 ----------

    def register(self, kind: str, engine: TTSEngine | STTEngine) -> None:
        assert kind in KINDS, f"未知引擎类别: {kind}"
        self._engines[kind] = engine
        self._states[kind] = EngineState.UNLOADED
        self._events[kind] = asyncio.Event()
        self._locks[kind] = threading.Lock()

    def start_background_load(self) -> None:
        """lifespan 中调用：不阻塞端口监听，加载中 /health 如实上报。"""
        for kind in list(self._engines):
            self._ensure_load_task(kind)

    def _ensure_load_task(self, kind: str) -> asyncio.Task:
        task = self._load_tasks.get(kind)
        if task is None or task.done():
            task = asyncio.create_task(self._load_kind(kind))
            self._load_tasks[kind] = task
        return task

    async def _load_kind(self, kind: str) -> None:
        engine = self._engines[kind]
        self._states[kind] = EngineState.LOADING
        self._events[kind].clear()
        t0 = time.monotonic()
        try:
            await engine.load(self.device, self.fp16)
            engine.warmup()
            self._states[kind] = engine.state
            self._errors[kind] = ""
            logger.info(
                "引擎 %s(%s) 加载完成: %.1fs, device=%s, fp16=%s",
                kind, engine.name, time.monotonic() - t0, self.device, self.fp16,
            )
        except Exception as e:  # noqa: BLE001
            self._states[kind] = EngineState.ERROR
            self._errors[kind] = str(e)
            logger.exception("引擎 %s(%s) 加载失败 (%.1fs)", kind, engine.name,
                             time.monotonic() - t0)
        finally:
            self._events[kind].set()

    # ---------- 请求路径 ----------

    async def get_ready(self, kind: str) -> TTSEngine | STTEngine:
        """返回就绪引擎；加载中 503；错误 503；未注册 503。"""
        if kind not in self._engines:
            raise EngineError(f"引擎 {kind} 未注册")
        state = self._states[kind]
        if state in (EngineState.UNLOADED, EngineState.LOADING):
            self._ensure_load_task(kind)  # 懒加载模式复用同一门闩
            try:
                await asyncio.wait_for(
                    self._events[kind].wait(), self._settings.load_timeout_s
                )
            except asyncio.TimeoutError as e:
                raise ModelLoadingError(
                    f"引擎 {kind} 加载超过 {self._settings.load_timeout_s:.0f}s"
                ) from e
            state = self._states[kind]
        if state in (EngineState.ERROR, EngineState.UNLOADED, EngineState.LOADING):
            detail = self._errors.get(kind, "")
            err = EngineError(f"引擎 {kind} 不可用: {detail or state}")
            err.retry_after = 5
            raise err
        return self._engines[kind]

    async def run(self, kind: str, fn: Callable, *args) -> Any:
        """在守卫下执行引擎推理。fn(engine, *args)，同步阻塞。"""
        engine = await self.get_ready(kind)
        lock = self._locks[kind]

        def _wrapped():
            with lock:
                return fn(engine, *args)

        return await self.guard.run(_wrapped)

    # ---------- 状态与生命周期 ----------

    def status(self) -> dict:
        engines = {}
        for kind, engine in self._engines.items():
            engines[kind] = {
                "name": engine.name,
                "state": self._states[kind].value,
                "device": self.device,
                "precision": "fp16" if self.fp16 else "fp32",
                "error": self._errors.get(kind, "") or None,
                **engine.extra_status(),
            }
        return {
            "status": "ok",
            "version": _app_version(),
            "device": self.device,
            "engines": engines,
            "queue": self.guard.stats,
            "uptime_s": round(time.monotonic() - self._started_at),
        }

    def ready_for_readiness(self) -> bool:
        """预加载配置的引擎全部 READY/DEGRADED 才算 ready。"""
        preload_needed = [k for k in self._engines if self._settings.preload_engines]
        return all(
            self._states[k] in (EngineState.READY, EngineState.DEGRADED)
            for k in preload_needed
        )

    async def shutdown(self) -> None:
        for kind in reversed(list(self._engines)):
            try:
                await self._engines[kind].unload()
            except Exception:  # noqa: BLE001
                logger.exception("引擎 %s 卸载失败", kind)
        try:
            import gc

            import torch

            if self.device == "cuda":
                torch.cuda.empty_cache()
            del gc
        except ImportError:
            pass
        logger.info("所有引擎已卸载，显存已释放")


def _app_version() -> str:
    from app import __version__

    return __version__
