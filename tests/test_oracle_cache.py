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
