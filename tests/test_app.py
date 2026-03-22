"""Tests for FastAPI application endpoints."""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi.testclient import TestClient

from pythia.config import PythiaConfig
from pythia.server.app import create_app


@pytest.fixture
def mock_config():
    return PythiaConfig()


@pytest.fixture
def app(mock_config):
    """Create app with fully mocked backends."""
    with patch("pythia.server.app.OracleCache") as MockCache, \
         patch("pythia.server.app.OllamaClient") as MockOllama, \
         patch("pythia.server.app.SearxngClient") as MockSearxng:

        mock_cache = AsyncMock()
        mock_cache.health = AsyncMock(return_value=True)
        mock_cache.get_cache_size = AsyncMock(return_value=5)
        mock_cache.get_history = AsyncMock(return_value=[
            {"query": "test", "cache_hit": False, "response_time_ms": 100, "model_used": "qwen3.5:9b", "created_at": "2025-01-01T00:00:00"}
        ])
        mock_cache.get_stats = AsyncMock(return_value={
            "total_searches": 10, "cache_hits": 3, "cache_hit_rate": 0.3, "avg_response_ms": 200
        })
        mock_cache.clear_cache = AsyncMock(return_value=5)
        mock_cache.connect = AsyncMock()
        mock_cache.close = AsyncMock()
        MockCache.return_value = mock_cache

        mock_ollama = MagicMock()
        mock_ollama.health = AsyncMock(return_value=True)
        mock_ollama.model = "qwen3.5:9b"
        MockOllama.return_value = mock_ollama

        mock_searxng = MagicMock()
        mock_searxng.health = AsyncMock(return_value=True)
        MockSearxng.return_value = mock_searxng

        application = create_app(mock_config)
        # Store mocks for test access
        application.state.mock_cache = mock_cache
        application.state.mock_ollama = mock_ollama
        application.state.mock_searxng = mock_searxng
        yield application


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["oracle"] is True
    assert data["searxng"] is True
    assert data["ollama"] is True
    assert data["cache_size"] == 5


def test_history_endpoint(client):
    resp = client.get("/history?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["query"] == "test"


def test_history_validation(client):
    resp = client.get("/history?limit=0")
    assert resp.status_code == 422

    resp = client.get("/history?limit=101")
    assert resp.status_code == 422


def test_stats_endpoint(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_searches"] == 10
    assert data["cache_hit_rate"] == 0.3


def test_clear_cache_endpoint(client):
    resp = client.delete("/cache")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 5


def test_embed_endpoint(client):
    resp = client.post("/embed", json={"text": "hello world"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "hello world"
    assert data["dimensions"] == 384
    assert len(data["embedding"]) == 384


def test_embed_validation_empty(client):
    resp = client.post("/embed", json={"text": ""})
    assert resp.status_code == 422


def test_embed_validation_too_long(client):
    resp = client.post("/embed", json={"text": "x" * 10001})
    assert resp.status_code == 422


def test_search_validation_empty(client):
    resp = client.post("/search", json={"query": ""})
    assert resp.status_code == 422


def test_search_validation_too_long(client):
    resp = client.post("/search", json={"query": "x" * 4001})
    assert resp.status_code == 422


def test_research_validation_empty(client):
    resp = client.post("/research", json={"query": ""})
    assert resp.status_code == 422


def test_research_validation_max_rounds(client):
    resp = client.post("/research", json={"query": "test", "max_rounds": 0})
    assert resp.status_code == 422

    resp = client.post("/research", json={"query": "test", "max_rounds": 11})
    assert resp.status_code == 422


def test_cors_headers(client):
    resp = client.options("/health", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
    })
    assert "access-control-allow-origin" in resp.headers
