import json

import httpx
import pytest

from app.llm.provider import LLMError, OpenAICompatProvider


def make_provider(handler, *, api_key: str = "") -> OpenAICompatProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatProvider(
        base_url="https://llm.example/v1/",
        model="test-model",
        api_key=api_key,
        client=client,
    )


def test_complete_success_parses_content_and_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://llm.example/v1/chat/completions")
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["max_tokens"] == 128
        assert body["temperature"] == 0.0
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": "hello [1]"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    provider = make_provider(handler)
    completion = provider.complete(
        [{"role": "user", "content": "hi"}], max_tokens=128, temperature=0.0
    )

    assert completion.text == "hello [1]"
    assert completion.model == "test-model"
    assert completion.input_tokens == 10
    assert completion.output_tokens == 5


def test_authorization_header_sent_when_key_present():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("Authorization", "")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    make_provider(handler, api_key="secret").complete(
        [{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.0
    )

    assert seen["authorization"] == "Bearer secret"


def test_no_authorization_header_when_key_empty():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("Authorization", "")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    make_provider(handler).complete(
        [{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.0
    )

    assert seen["authorization"] == ""


def test_http_error_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    provider = make_provider(handler)
    with pytest.raises(LLMError, match="status 500"):
        provider.complete([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.0)


def test_transport_error_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = make_provider(handler)
    with pytest.raises(LLMError, match="failed"):
        provider.complete([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.0)


def test_malformed_response_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    provider = make_provider(handler)
    with pytest.raises(LLMError, match="malformed"):
        provider.complete([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.0)


def test_non_json_response_raises_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    provider = make_provider(handler)
    with pytest.raises(LLMError, match="malformed"):
        provider.complete([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.0)
