"""统一错误体系：AppError 层级 + FastAPI handler 注册。

错误响应体::

    {"error": {"code": "queue_full", "message": "...", "request_id": "..."}}
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("app")


class AppError(Exception):
    """业务错误基类。子类覆写 code/http_status/retry_after。"""

    code: str = "internal_error"
    http_status: int = 500
    retry_after: int | None = None

    def __init__(self, message: str = "", *, request_id: str = ""):
        self.message = message or self.code
        self.request_id = request_id
        super().__init__(self.message)

    def to_response(self) -> JSONResponse:
        headers = {} if self.retry_after is None else {"Retry-After": str(self.retry_after)}
        return JSONResponse(
            status_code=self.http_status,
            content={"error": {"code": self.code, "message": self.message,
                               "request_id": self.request_id}},
            headers=headers,
        )


class InvalidRequestError(AppError):
    code = "invalid_request"
    http_status = 400


class UnsupportedFormatError(AppError):
    code = "unsupported_format"
    http_status = 415


class PayloadTooLargeError(AppError):
    code = "payload_too_large"
    http_status = 413


class QueueFullError(AppError):
    code = "queue_full"
    http_status = 429
    retry_after = 2

    def __init__(self, message: str = "推理队列已满，请稍后重试", **kw):
        super().__init__(message, **kw)


class ModelLoadingError(AppError):
    code = "model_loading"
    http_status = 503
    retry_after = 5

    def __init__(self, message: str = "模型加载中，请稍后重试", **kw):
        super().__init__(message, **kw)


class EngineError(AppError):
    """引擎未注册或加载失败。"""

    code = "engine_error"
    http_status = 503


class InferenceTimeoutError(AppError):
    code = "inference_timeout"
    http_status = 504

    def __init__(self, message: str = "推理超时", **kw):
        super().__init__(message, **kw)


class InferenceFailedError(AppError):
    """推理过程异常（含 OOM 降级后仍失败）。"""

    code = "inference_failed"
    http_status = 503


class AuthError(AppError):
    code = "unauthorized"
    http_status = 401


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        exc.request_id = getattr(request.state, "request_id", "")
        if exc.http_status >= 500:
            logger.error("[%s] %s", exc.code, exc.message)
        return exc.to_response()

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        errors = [
            {"loc": [str(p) for p in e["loc"][1:]], "msg": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "; ".join(
                        f"{'.'.join(e['loc']) or 'body'}: {e['msg']}" for e in errors
                    ),
                    "request_id": getattr(request.state, "request_id", ""),
                }
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("未处理异常: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": f"内部错误: {exc}",
                    "request_id": getattr(request.state, "request_id", ""),
                }
            },
        )
