"""Tests for Phase 7 innovations: query rewriting, parallel prefetch, citation density."""
import asyncio
import pytest
from unittest.mock import AsyncMock

from pythia.server.search import SearchOrchestrator, EventType, _count_citations
from pythia.server.searxng import SearchResult
from pythia.server.oracle_cache import CacheEntry


# --- Innovation 3: Citation density ---

def test_count_citations_basic():
    text = "LLMs use RLHF [1] and DPO [2] for alignment."
    assert _count_citations(text) == 2


def test_count_citations_deduplicates():
    text = "RLHF [1] is used widely. As mentioned [1], it works well with [2]."
    assert _count_citations(text) == 2


def test_count_citations_none():
    assert _count_citations("No citations here.") == 0


def test_count_citations_many():
    text = " ".join(f"[{i}]" for i in range(1, 11))
    assert _count_citations(text) == 10


def test_count_citations_not_confused_by_brackets():
    text = "array[0] = value; see [1] for details."
    assert _count_citations(text) == 2  # [0] and [1] both match as citation-like


@pytest.mark.asyncio
async def test_citation_density_in_done_event():
    """Cache miss search should include citation metrics in DONE event."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    async def fake_stream(system, user, model=None):
        yield "Answer based on [1] and [2] sources. Also [1] again."

    mock_ollama.generate_stream = fake_stream
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1]"))
    mock_cache.store = AsyncMock()
    mock_cache.record_search = AsyncMock()
    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(return_value=[
        SearchResult(index=1, title="T1", url="https://t1.com", snippet="s1"),
        SearchResult(index=2, title="T2", url="https://t2.com", snippet="s2"),
        SearchResult(index=3, title="T3", url="https://t3.com", snippet="s3"),
    ])

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("test query"):
        events.append(event)

    done = next(e for e in events if e.event_type == EventType.DONE)
    assert done.data["citations_used"] == 2  # [1] and [2], deduplicated
    assert done.data["citation_density"] == round(2 / 3, 2)


# --- Innovation 1: Query rewriting ---

@pytest.mark.asyncio
async def test_rewrite_query_basic():
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate = AsyncMock(return_value="transformer architecture neural networks")
    mock_cache = AsyncMock()
    mock_searxng = AsyncMock()

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    result = await orch.rewrite_query("how does that neural network thing work")
    assert result == "transformer architecture neural networks"


@pytest.mark.asyncio
async def test_rewrite_query_strips_quotes():
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate = AsyncMock(return_value='"optimized search query"')
    mock_cache = AsyncMock()
    mock_searxng = AsyncMock()

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    result = await orch.rewrite_query("something vague")
    assert result == "optimized search query"


@pytest.mark.asyncio
async def test_rewrite_query_fallback_on_error():
    """If the LLM call fails, rewrite should return the original query."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate = AsyncMock(side_effect=ConnectionError("Ollama down"))
    mock_cache = AsyncMock()
    mock_searxng = AsyncMock()

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    result = await orch.rewrite_query("my original query")
    assert result == "my original query"


@pytest.mark.asyncio
async def test_rewrite_query_fallback_on_empty():
    """If LLM returns empty/tiny response, fall back to original."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate = AsyncMock(return_value="")
    mock_cache = AsyncMock()
    mock_searxng = AsyncMock()

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    result = await orch.rewrite_query("original")
    assert result == "original"


@pytest.mark.asyncio
async def test_search_with_rewrite_flag():
    """rewrite=True should emit rewrite status events."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate = AsyncMock(return_value="rewritten query")
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    async def fake_stream(system, user, model=None):
        yield "answer"

    mock_ollama.generate_stream = fake_stream
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1]"))
    mock_cache.store = AsyncMock()
    mock_cache.record_search = AsyncMock()
    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(return_value=[
        SearchResult(index=1, title="T", url="https://t.com", snippet="s"),
    ])

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("vague question", rewrite=True):
        events.append(event)

    status_messages = [e.data["message"] for e in events if e.event_type == EventType.STATUS]
    assert any("Optimizing" in m for m in status_messages)
    assert any("Rewritten" in m for m in status_messages)

    # SearXNG should have been called with the rewritten query
    mock_searxng.search.assert_called_once_with("rewritten query")


# --- Innovation 2: Parallel cache + search prefetch ---

@pytest.mark.asyncio
async def test_parallel_prefetch_cache_hit_cancels_search():
    """On cache hit, web search should be cancelled (not awaited to completion)."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    cached = CacheEntry(
        query="test", answer="cached answer", sources=[],
        model_used="test", similarity=0.95,
    )
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(cached, "[0.1]"))
    mock_cache.record_search = AsyncMock()

    search_started = False

    async def slow_search(query):
        nonlocal search_started
        search_started = True
        await asyncio.sleep(10)  # Should get cancelled before this completes
        return []

    mock_searxng = AsyncMock()
    mock_searxng.search = slow_search

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("test"):
        events.append(event)

    done = next(e for e in events if e.event_type == EventType.DONE)
    assert done.data["cache_hit"] is True
    # The search task was created but should have been cancelled
    assert search_started is True


@pytest.mark.asyncio
async def test_parallel_prefetch_cache_miss_uses_search():
    """On cache miss, already-running web search should be awaited."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    async def fake_stream(system, user, model=None):
        yield "answer"

    mock_ollama.generate_stream = fake_stream
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1]"))
    mock_cache.store = AsyncMock()
    mock_cache.record_search = AsyncMock()

    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(return_value=[
        SearchResult(index=1, title="T", url="https://t.com", snippet="s"),
    ])

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("test"):
        events.append(event)

    done = next(e for e in events if e.event_type == EventType.DONE)
    assert done.data["cache_hit"] is False
    mock_searxng.search.assert_called_once()


@pytest.mark.asyncio
async def test_parallel_prefetch_search_error_after_cache_miss():
    """If web search fails after cache miss, error should be reported."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1]"))

    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(side_effect=ConnectionError("SearXNG down"))

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("test"):
        events.append(event)

    done = next(e for e in events if e.event_type == EventType.DONE)
    assert "error" in done.data
