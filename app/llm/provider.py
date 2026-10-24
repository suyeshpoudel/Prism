from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import httpx


class LLMError(Exception):
    """Raised when the LLM provider fails to produce a completion."""


@dataclass(frozen=True)
class LLMCompletion:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(Protocol):
    """Replaceable answer-generation backend (OpenAI-compatible baseline)."""

    @property
    def model_name(self) -> str: ...

    def complete(
        self,
        messages: Sequence[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> LLMCompletion: ...


class OpenAICompatProvider:
    """Minimal client for any OpenAI-compatible ``/chat/completions`` endpoint.

    Works with OpenAI, Ollama (``/v1``), vLLM, LM Studio, etc. via
    ``base_url``. ``api_key`` is optional for local servers that do not
    require authentication.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        messages: Sequence[dict],
        *,
        max_tokens: int,
        temperature: float,
    ) -> LLMCompletion:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions", json=payload, headers=headers
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(f"LLM request failed with status {response.status_code}.")

        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM response was malformed.") from exc

        usage = data.get("usage") or {}
        return LLMCompletion(
            text=text if isinstance(text, str) else "",
            model=str(data.get("model", self._model)),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    def close(self) -> None:
        self._client.close()
