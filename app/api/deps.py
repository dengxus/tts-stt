"""FastAPI 依赖：引擎管理器获取 + 可选 Bearer 鉴权。"""

from __future__ import annotations

import hmac

from fastapi import Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings
from app.engines.manager import EngineManager
from app.errors import AuthError

_bearer = HTTPBearer(auto_error=False)


def get_manager(request: Request) -> EngineManager:
    return request.app.state.manager


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """settings.api_key 为空则放行，否则校验 Bearer token。"""
    settings: Settings = request.app.state.settings
    if not settings.api_key:
        return
    if credentials is None or not hmac.compare_digest(
        credentials.credentials, settings.api_key
    ):
        raise AuthError("缺少或无效的 Authorization: Bearer <API_KEY>")
