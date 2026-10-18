from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.core.errors import UnauthorizedError

API_KEY_HEADER = "X-API-Key"


def resolve_owner(api_key: str | None, settings: Settings) -> str:
    """Map an API key to an owner id. Raises 401 for missing or unknown keys."""
    if not api_key:
        raise UnauthorizedError(f"Missing '{API_KEY_HEADER}' header.")
    owner = settings.api_keys.get(api_key)
    if owner is None:
        raise UnauthorizedError("Invalid API key.")
    return owner


def get_current_owner(request: Request) -> str:
    """FastAPI dependency: resolve the request's owner from the X-API-Key header."""
    settings: Settings = request.app.state.settings
    api_key = request.headers.get(API_KEY_HEADER)
    return resolve_owner(api_key, settings)
