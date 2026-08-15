"""Tests for freshness-aware caching and knowledge evolution features."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from unittest.mock import AsyncMock

from pythia.server.search import SearchOrchestrator, EventType, is_time_sensitive
from pythia.server.searxng import SearchResult
from pythia.server.oracle_cache import CacheEntry


# --- Time-sensitivity heuristic ---


def test_is_time_sensitive_year():
    assert is_time_sensitive("What are the best laptops of 2026?")
    assert is_time_sensitive("RISC-V news 2025")


def test_is_time_sensitive_markers():
    assert is_time_sensitive("latest AI developments")
    assert is_time_sensitive("breaking news today")
    assert is_time_sensitive("current stock price of NVDA")
    assert is_time_sensitive("latest release notes")


def test_is_time_sensitive_negative():
    assert not is_time_sensitive("What is RLHF?")
    assert not is_time_sensitive("Explain quantum entanglement")
    assert not is_time_sensitive("history of the Roman Empire")


def test_is_time_sensitive_word_boundaries():
    # Markers must not match as substrings of longer words (false positives).
    assert not is_time_sensitive("How does knowledge distillation work?")  # "now"
    assert not is_time_sensitive("Read the newspaper archives")  # "news"
    assert not is_time_sensitive("The package was delivered")  # "live"
    # Inflected forms and phrases still count.
    assert is_time_sensitive("currently best practices")
    assert is_time_sensitive("what changed this week")


# --- Stale cache bypass ---


def _cached_entry(created_at: datetime | None = None) -> CacheEntry:
    return CacheEntry(
        query="latest AI news",
        answer="Some cached answer.",
        sources=[{"index": 1, "title": "T", "url": "https://t.com", "snippet": "s"}],
        model_used="test",
        similarity=0.95,
        created_at=created_at,
    )


def test_stale_entry_old_and_time_sensitive():
    orch = SearchOrchestrator(
        ollama=AsyncMock(), cache=AsyncMock(), searxng=AsyncMock(), cache_max_age_hours=24
    )
    old = _cached_entry(datetime.now(timezone.utc) - timedelta(hours=48))
    assert orch._is_stale_cache_entry(old, "latest AI news") is True


def test_stale_entry_fresh_and_time_sensitive():
    orch = SearchOrchestrator(
        ollama=AsyncMock(), cache=AsyncMock(), searxng=AsyncMock(), cache_max_age_hours=24
    )
    fresh = _cached_entry(datetime.now(timezone.utc) - timedelta(hours=1))
    assert orch._is_stale_cache_entry(fresh, "latest AI news") is False


def test_stale_entry_old_but_not_time_sensitive():
    orch = SearchOrchestrator(
        ollama=AsyncMock(), cache=AsyncMock(), searxng=AsyncMock(), cache_max_age_hours=24
    )
    old = _cached_entry(datetime.now(timezone.utc) - timedelta(days=30))
    assert orch._is_stale_cache_entry(old, "Explain RLHF") is False


def test_stale_disabled_when_max_age_zero():
    orch = SearchOrchestrator(
        ollama=AsyncMock(), cache=AsyncMock(), searxng=AsyncMock(), cache_max_age_hours=0
    )
    old = _cached_entry(datetime.now(timezone.utc) - timedelta(days=30))
    assert orch._is_stale_cache_entry(old, "latest AI news") is False


def test_stale_handles_naive_created_at():
    """Oracle returns naive timestamps; treat them as UTC."""
    orch = SearchOrchestrator(
        ollama=AsyncMock(), cache=AsyncMock(), searxng=AsyncMock(), cache_max_age_hours=24
    )
    old = _cached_entry(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=3))
    assert orch._is_stale_cache_entry(old, "latest AI news") is True


@pytest.mark.asyncio
async def test_search_stale_cache_hit_bypasses_and_researches():
    """An old cache entry for a time-sensitive query must be bypassed: web search
    still runs, a fresh answer is produced, and the DONE event is a cache miss."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    async def fake_stream(system, user, model=None):
        yield "Fresh "

    mock_ollama.generate_stream = fake_stream

    old = _cached_entry(datetime.now(timezone.utc) - timedelta(days=10))
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(old, "[0.1]"))
    mock_cache.store = AsyncMock()
    mock_cache.record_search = AsyncMock()

    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(
        return_value=[
            SearchResult(index=1, title="Fresh", url="https://t.com", snippet="fresh news"),
        ]
    )

    orch = SearchOrchestrator(
        ollama=mock_ollama,
        cache=mock_cache,
        searxng=mock_searxng,
        cache_max_age_hours=24,
    )

    events = []
    async for event in orch.search("latest AI news"):
        events.append(event)

    # Web search must have been awaited (not cancelled) and a fresh answer produced
    mock_searxng.search.assert_called_once()
    mock_cache.store.assert_called_once()

    done = next(e for e in events if e.event_type == EventType.DONE)
    assert done.data["cache_hit"] is False

    statuses = [e.data.get("message", "") for e in events if e.event_type == EventType.STATUS]
    assert any("older than" in msg for msg in statuses)


@pytest.mark.asyncio
async def test_search_fresh_cache_hit_serves_cached_answer():
    """A recent cache entry for a time-sensitive query is served normally."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    fresh = _cached_entry(datetime.now(timezone.utc) - timedelta(hours=1))
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(fresh, "[0.1]"))
    mock_cache.record_search = AsyncMock()

    mock_searxng = AsyncMock()

    orch = SearchOrchestrator(
        ollama=mock_ollama,
        cache=mock_cache,
        searxng=mock_searxng,
        cache_max_age_hours=24,
    )

    events = []
    async for event in orch.search("latest AI news"):
        events.append(event)

    done = next(e for e in events if e.event_type == EventType.DONE)
    assert done.data["cache_hit"] is True
    # Web search should have been cancelled; store should NOT be called
    mock_cache.store.assert_not_called()
