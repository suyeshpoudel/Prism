"""Deterministic offline LLM stubs for tests."""

from __future__ import annotations

from app.llm.provider import LLMCompletion, LLMError


class StubLLMProvider:
    """Records calls and returns a fixed reply."""

    def __init__(
        self,
        reply: str = "Stub answer.",
        model_name: str = "stub-model",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        self._reply = reply
        self._model_name = model_name
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self.calls: list[dict] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, messages, *, max_tokens: int, temperature: float) -> LLMCompletion:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return LLMCompletion(
            text=self._reply,
            model=self._model_name,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )


class FailingLLMProvider:
    """Raises LLMError on every call; used to prove the LLM is not invoked."""

    @property
    def model_name(self) -> str:
        return "failing-model"

    def complete(self, messages, *, max_tokens: int, temperature: float) -> LLMCompletion:
        raise LLMError("stub failure")
