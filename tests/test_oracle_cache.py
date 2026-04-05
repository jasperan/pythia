"""Tests for Oracle vector cache — unit tests with mock connection."""
import pytest
from pythia.server.oracle_cache import OracleCache, CacheEntry


def test_cache_entry_creation():
    entry = CacheEntry(
        query="How does RLHF work?",
        answer="RLHF is a technique...",
        sources=[{"index": 1, "title": "Test", "url": "https://test.com", "snippet": "test"}],
        model_used="qwen3.5:9b",
        similarity=0.92,
    )
    assert entry.query == "How does RLHF work?"
    assert entry.similarity == 0.92
    assert len(entry.sources) == 1


def test_is_cache_hit():
    cache = OracleCache.__new__(OracleCache)
    cache.similarity_threshold = 0.85
    assert cache._is_cache_hit(0.92) is True
    assert cache._is_cache_hit(0.85) is True
    assert cache._is_cache_hit(0.80) is False
    assert cache._is_cache_hit(0.0) is False


@pytest.mark.asyncio
async def test_store_research_accepts_extended_fields():
    """store_research should accept slug, parent_id, verification, and provenance fields."""
    cache = OracleCache(dsn="localhost:1523/FREEPDB1", user="pythia", password="pythia")
    # No pool — should return "" gracefully, but shouldn't raise TypeError on the signature
    result = await cache.store_research(
        query="test", report="report", sub_queries=["q1"],
        rounds_used=1, total_sources=2, model_used="test", elapsed_ms=100,
        slug="test-slug", parent_id=None,
        verification_status="pass", verification_summary="All good.",
        provenance="# Provenance",
    )
    assert result == ""
