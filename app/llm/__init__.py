from __future__ import annotations

from app.config import Settings
from app.llm.provider import LLMCompletion, LLMError, LLMProvider, OpenAICompatProvider


def build_llm_provider(settings: Settings) -> LLMProvider | None:
    """Build the configured LLM provider, or None when no endpoint is set."""
    if not settings.llm_base_url:
        return None
    return OpenAICompatProvider(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
    )


__all__ = [
    "LLMCompletion",
    "LLMError",
    "LLMProvider",
    "OpenAICompatProvider",
    "build_llm_provider",
]
