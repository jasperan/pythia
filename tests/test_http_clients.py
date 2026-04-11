"""Tests for HTTP clients — Ollama and SearXNG with mocked httpx."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from pythia.server.ollama import OllamaClient
from pythia.server.searxng import SearxngClient


# --- OllamaClient ---

@pytest.mark.asyncio
async def test_ollama_generate_parses_response():
    """generate() should return content from Ollama chat response."""
    client = OllamaClient(base_url="http://localhost:11434", model="test-model")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "The answer is 42."},
        "done": True,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        result = await client.generate("system prompt", "user prompt")
        assert result == "The answer is 42."

        # Verify correct payload
        call_args = mock_http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model"] == "test-model"
        assert payload["stream"] is False


@pytest.mark.asyncio
async def test_ollama_generate_json_mode():
    """json_mode should set format=json in payload."""
    client = OllamaClient(base_url="http://localhost:11434", model="test")

    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": '{"key": "value"}'}}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        await client.generate("sys", "user", json_mode=True)
        payload = mock_http.post.call_args.kwargs.get("json") or mock_http.post.call_args[1].get("json")
        assert payload["format"] == "json"


@pytest.mark.asyncio
async def test_ollama_generate_model_override():
    """Model parameter should override default model."""
    client = OllamaClient(base_url="http://localhost:11434", model="default")

    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "ok"}}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        await client.generate("sys", "user", model="override-model")
        payload = mock_http.post.call_args.kwargs.get("json") or mock_http.post.call_args[1].get("json")
        assert payload["model"] == "override-model"


@pytest.mark.asyncio
async def test_ollama_generate_http_error():
    """generate() should raise on HTTP errors."""
    client = OllamaClient(base_url="http://localhost:11434", model="test")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        with pytest.raises(httpx.HTTPStatusError):
            await client.generate("sys", "user")


@pytest.mark.asyncio
async def test_ollama_health_ok():
    """health() should return True when Ollama responds 200."""
    client = OllamaClient(base_url="http://localhost:11434", model="test")

    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        assert await client.health() is True


@pytest.mark.asyncio
async def test_ollama_health_failure():
    """health() should return False on connection error."""
    client = OllamaClient(base_url="http://localhost:11434", model="test")

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        assert await client.health() is False


# --- SearxngClient ---

@pytest.mark.asyncio
async def test_searxng_search_http_call():
    """search() should call SearXNG with correct params and parse results."""
    client = SearxngClient(base_url="http://localhost:8889", max_results=5, categories=["general", "science"])

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"title": "Result 1", "url": "https://r1.com", "content": "snippet 1"},
            {"title": "Result 2", "url": "https://r2.com", "content": "snippet 2"},
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        results = await client.search("test query")
        assert len(results) == 2
        assert results[0].title == "Result 1"

        call_args = mock_http.get.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["q"] == "test query"
        assert params["format"] == "json"
        assert "general,science" in params["categories"]


@pytest.mark.asyncio
async def test_searxng_search_http_error():
    """search() should raise on HTTP errors."""
    client = SearxngClient(base_url="http://localhost:8889")

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503", request=MagicMock(), response=MagicMock()
    )

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        with pytest.raises(httpx.HTTPStatusError):
            await client.search("test")


@pytest.mark.asyncio
async def test_searxng_health_ok():
    client = SearxngClient(base_url="http://localhost:8889")

    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        assert await client.health() is True


@pytest.mark.asyncio
async def test_searxng_health_failure():
    client = SearxngClient(base_url="http://localhost:8889")

    with patch("httpx.AsyncClient") as MockClient:
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_http

        assert await client.health() is False
