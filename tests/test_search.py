"""Tests for search orchestrator."""
import pytest
from unittest.mock import AsyncMock
from pythia.server.search import SearchOrchestrator, SearchEvent, EventType
from pythia.server.searxng import SearchResult
from pythia.server.oracle_cache import CacheEntry


@pytest.mark.asyncio
async def test_search_cache_hit():
    mock_ollama = AsyncMock()
    mock_ollama.model = "qwen3.5:9b"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    cached = CacheEntry(
        query="What is RLHF?",
        answer="RLHF is a technique for aligning LLMs.",
        sources=[{"index": 1, "title": "Test", "url": "https://test.com", "snippet": "test"}],
        model_used="qwen3.5:9b",
        similarity=0.92,
    )
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(cached, "[0.1,0.2]"))
    mock_cache.record_search = AsyncMock()

    mock_searxng = AsyncMock()

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)

    events = []
    async for event in orch.search("How does RLHF work?"):
        events.append(event)

    mock_cache.lookup.assert_called_once_with("How does RLHF work?")
    # With parallel prefetch, web search starts concurrently but is cancelled on cache hit
    types = [e.event_type for e in events]
    assert EventType.STATUS in types
    assert EventType.DONE in types


@pytest.mark.asyncio
async def test_search_cache_miss():
    mock_ollama = AsyncMock()
    mock_ollama.model = "qwen3.5:9b"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    async def fake_stream(system, user, model=None):
        for word in ["RLHF ", "is ", "great."]:
            yield word

    mock_ollama.generate_stream = fake_stream

    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1,0.2]"))
    mock_cache.store = AsyncMock()
    mock_cache.record_search = AsyncMock()

    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(
        return_value=[
            SearchResult(index=1, title="RLHF", url="https://test.com", snippet="RLHF explained"),
        ]
    )

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)

    events = []
    async for event in orch.search("What is RLHF?"):
        events.append(event)

    mock_cache.lookup.assert_called_once_with("What is RLHF?")
    mock_searxng.search.assert_called_once()
    types = [e.event_type for e in events]
    assert EventType.SOURCE in types
    assert EventType.TOKEN in types
    assert EventType.DONE in types
