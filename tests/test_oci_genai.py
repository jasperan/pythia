"""Tests for OciGenAIClient — mocks httpx.AsyncClient."""
import json

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pythia.server.oci_genai import OciGenAIClient


@pytest.mark.asyncio
async def test_oci_generate_parses_response():
    """generate() should parse choices[0].message.content from OpenAI-compat response."""
    client = OciGenAIClient(base_url="http://localhost:9999/v1", model="xai.grok-4")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "The answer is 42."}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_http

        result = await client.generate("system prompt", "user prompt")
        assert result == "The answer is 42."

        call_args = mock_http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model"] == "xai.grok-4"
        assert payload["stream"] is False
        assert payload["messages"][0] == {"role": "system", "content": "system prompt"}
        assert payload["messages"][1] == {"role": "user", "content": "user prompt"}

        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert headers["Authorization"] == "Bearer oci-genai"


@pytest.mark.asyncio
async def test_oci_generate_authorization_header_custom():
    """Authorization header should use provided api_key."""
    client = OciGenAIClient(
        base_url="http://localhost:9999/v1",
        model="m",
        api_key="custom-key",  # pragma: allowlist secret
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_http

        await client.generate("sys", "user")
        headers = mock_http.post.call_args.kwargs.get("headers")
        assert headers["Authorization"] == "Bearer custom-key"


@pytest.mark.asyncio
async def test_oci_generate_json_mode_sets_response_format():
    """json_mode should add response_format=json_object."""
    client = OciGenAIClient(base_url="http://localhost:9999/v1", model="m")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"key": "value"}'}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_http

        result = await client.generate("sys", "user", json_mode=True)
        payload = mock_http.post.call_args.kwargs.get("json")
        assert payload["response_format"] == {"type": "json_object"}
        assert result == '{"key": "value"}'


@pytest.mark.asyncio
async def test_oci_generate_json_mode_strips_fences():
    """json_mode should strip ```json fences around the content."""
    client = OciGenAIClient(base_url="http://localhost:9999/v1", model="m")

    fenced = '```json\n{"a": 1}\n```'
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": fenced}}]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_http

        result = await client.generate("sys", "user", json_mode=True)
        assert result == '{"a": 1}'


@pytest.mark.asyncio
async def test_oci_generate_model_override():
    """Model argument should override the default model."""
    client = OciGenAIClient(base_url="http://localhost:9999/v1", model="default-m")

    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_http

        await client.generate("sys", "user", model="override-m")
        payload = mock_http.post.call_args.kwargs.get("json")
        assert payload["model"] == "override-m"


@pytest.mark.asyncio
async def test_oci_generate_stream_parses_sse():
    """generate_stream() should parse SSE data: lines and stop on [DONE]."""
    client = OciGenAIClient(base_url="http://localhost:9999/v1", model="m")

    sse_lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
        "",
        "data: " + json.dumps({"choices": [{"delta": {"content": " world"}}]}),
        "data: [DONE]",
        "data: " + json.dumps({"choices": [{"delta": {"content": "ignored"}}]}),
    ]

    async def aiter_lines():
        for line in sse_lines:
            yield line

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = aiter_lines

    class MockStream:
        async def __aenter__(self_inner):
            return mock_response

        async def __aexit__(self_inner, *a):
            return False

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = MagicMock()
        mock_http.is_closed = False
        mock_http.stream = MagicMock(return_value=MockStream())
        MockClient.return_value = mock_http

        client._client = mock_http
        chunks = []
        async for c in client.generate_stream("sys", "user"):
            chunks.append(c)

        assert chunks == ["Hello", " world"]
        call_args = mock_http.stream.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["stream"] is True


@pytest.mark.asyncio
async def test_oci_health_ok():
    """health() should return True on 200 from /health."""
    client = OciGenAIClient(base_url="http://localhost:9999/v1", model="m")

    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_http

        assert await client.health() is True


@pytest.mark.asyncio
async def test_oci_health_failure():
    """health() should return False on connection error."""
    client = OciGenAIClient(base_url="http://localhost:9999/v1", model="m")

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        MockClient.return_value = mock_http

        assert await client.health() is False


@pytest.mark.asyncio
async def test_oci_suggestions_success():
    """generate_suggestions() should return a 3-item list from JSON array response."""
    client = OciGenAIClient(base_url="http://localhost:9999/v1", model="m")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '["q1", "q2", "q3", "q4"]'}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_http

        result = await client.generate_suggestions("query", "answer")
        assert result == ["q1", "q2", "q3"]


@pytest.mark.asyncio
async def test_oci_suggestions_failure_returns_empty():
    """generate_suggestions() should return [] on invalid JSON."""
    client = OciGenAIClient(base_url="http://localhost:9999/v1", model="m")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "not valid json"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_http

        result = await client.generate_suggestions("q", "a")
        assert result == []


@pytest.mark.asyncio
async def test_oci_generate_http_error():
    """generate() should raise on HTTP errors."""
    client = OciGenAIClient(base_url="http://localhost:9999/v1", model="m")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        MockClient.return_value = mock_http

        with pytest.raises(httpx.HTTPStatusError):
            await client.generate("sys", "user")
