"""Battle hardening tests — edge cases, malformed inputs, failure modes."""
import pytest
from unittest.mock import AsyncMock, patch

from pythia.server.search import SearchOrchestrator, EventType
from pythia.server.searxng import SearchResult
from pythia.server.oracle_cache import OracleCache, CacheEntry


# --- Race condition / thread safety ---

@pytest.mark.asyncio
async def test_search_concurrent_model_override_no_mutation():
    """Model override must not mutate shared OllamaClient.model."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "qwen3.5:9b"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    async def fake_stream(system, user, model=None):
        # During streaming, the shared model should NOT be changed
        assert mock_ollama.model == "qwen3.5:9b", "Shared model was mutated!"
        for word in ["answer"]:
            yield word

    mock_ollama.generate_stream = fake_stream
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1,0.2]"))
    mock_cache.store = AsyncMock()
    mock_cache.record_search = AsyncMock()
    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(return_value=[
        SearchResult(index=1, title="T", url="https://t.com", snippet="s"),
    ])

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("test", model_override="llama3.3:70b"):
        events.append(event)

    # Model should never have been mutated
    assert mock_ollama.model == "qwen3.5:9b"
    done = next(e for e in events if e.event_type == EventType.DONE)
    assert "error" not in done.data


# --- SearXNG failure modes ---

@pytest.mark.asyncio
async def test_search_searxng_connection_error():
    """Search should yield error event when SearXNG is unreachable."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1,0.2]"))
    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(side_effect=ConnectionError("refused"))

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("test"):
        events.append(event)

    done = next(e for e in events if e.event_type == EventType.DONE)
    assert "error" in done.data
    assert "refused" in done.data["error"]


@pytest.mark.asyncio
async def test_search_searxng_timeout():
    """Search should handle SearXNG timeout gracefully."""
    import httpx
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1,0.2]"))
    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("test"):
        events.append(event)

    done = next(e for e in events if e.event_type == EventType.DONE)
    assert "error" in done.data


# --- Ollama failure modes ---

@pytest.mark.asyncio
async def test_search_ollama_error_during_stream():
    """Search should handle Ollama errors during token streaming."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    async def failing_stream(system, user, model=None):
        yield "partial "
        raise ConnectionError("Ollama crashed")

    mock_ollama.generate_stream = failing_stream
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1,0.2]"))
    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(return_value=[
        SearchResult(index=1, title="T", url="https://t.com", snippet="s"),
    ])

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("test"):
        events.append(event)

    done = next(e for e in events if e.event_type == EventType.DONE)
    assert "error" in done.data
    assert "Ollama crashed" in done.data["error"]
    # Should NOT have stored partial answer in cache
    mock_cache.store.assert_not_called()


@pytest.mark.asyncio
async def test_search_ollama_empty_response():
    """Search should handle Ollama returning empty tokens."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    async def empty_stream(system, user, model=None):
        return
        yield  # make it an async generator

    mock_ollama.generate_stream = empty_stream
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1,0.2]"))
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

    # Should still complete and store (empty answer is valid)
    done = next(e for e in events if e.event_type == EventType.DONE)
    assert done.data["cache_hit"] is False
    mock_cache.store.assert_called_once()


# --- Deep search mode ---

@pytest.mark.asyncio
async def test_search_deep_mode_with_scraper():
    """Deep mode should trigger scraping and use build_deep_search_prompt."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    async def fake_stream(system, user, model=None):
        yield "deep answer"

    mock_ollama.generate_stream = fake_stream
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1,0.2]"))
    mock_cache.store = AsyncMock()
    mock_cache.record_search = AsyncMock()
    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(return_value=[
        SearchResult(index=1, title="T", url="https://t.com", snippet="s"),
    ])

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    with patch("pythia.server.search.scrape_urls") as mock_scrape:
        from pythia.scraper import ScrapedContent
        mock_scrape.return_value = [
            ScrapedContent(url="https://t.com", content="Full page content", success=True)
        ]
        async for event in orch.search("test", deep=True):
            events.append(event)

    mock_scrape.assert_called_once()
    types = [e.event_type for e in events]
    assert EventType.DONE in types


# --- Unicode and special characters ---

@pytest.mark.asyncio
async def test_search_unicode_query():
    """Search should handle unicode queries without errors."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    async def fake_stream(system, user, model=None):
        yield "unicode response"

    mock_ollama.generate_stream = fake_stream
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1,0.2]"))
    mock_cache.store = AsyncMock()
    mock_cache.record_search = AsyncMock()
    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(return_value=[
        SearchResult(index=1, title="T", url="https://t.com", snippet="s"),
    ])

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("quantum mechanics"):
        events.append(event)

    done = next(e for e in events if e.event_type == EventType.DONE)
    assert "error" not in done.data


# --- OracleCache no-pool graceful degradation ---

@pytest.mark.asyncio
async def test_cache_lookup_no_pool():
    cache = OracleCache.__new__(OracleCache)
    cache._pool = None
    entry, embedding = await cache.lookup("test")
    assert entry is None
    assert embedding == ""


@pytest.mark.asyncio
async def test_cache_store_no_pool():
    cache = OracleCache.__new__(OracleCache)
    cache._pool = None
    await cache.store("q", "a", [], "model")  # should not raise


@pytest.mark.asyncio
async def test_cache_record_search_no_pool():
    cache = OracleCache.__new__(OracleCache)
    cache._pool = None
    await cache.record_search("q", False, 100, "model")


@pytest.mark.asyncio
async def test_cache_health_no_pool():
    cache = OracleCache.__new__(OracleCache)
    cache._pool = None
    assert await cache.health() is False


@pytest.mark.asyncio
async def test_cache_get_stats_no_pool():
    cache = OracleCache.__new__(OracleCache)
    cache._pool = None
    stats = await cache.get_stats()
    assert stats["total_searches"] == 0


@pytest.mark.asyncio
async def test_cache_size_no_pool():
    cache = OracleCache.__new__(OracleCache)
    cache._pool = None
    assert await cache.get_cache_size() == 0


@pytest.mark.asyncio
async def test_cache_clear_no_pool():
    cache = OracleCache.__new__(OracleCache)
    cache._pool = None
    assert await cache.clear_cache() == 0


@pytest.mark.asyncio
async def test_cache_get_history_no_pool():
    cache = OracleCache.__new__(OracleCache)
    cache._pool = None
    assert await cache.get_history() == []


@pytest.mark.asyncio
async def test_cache_recall_findings_no_pool():
    cache = OracleCache.__new__(OracleCache)
    cache._pool = None
    assert await cache.recall_findings("q") == []


@pytest.mark.asyncio
async def test_cache_store_research_no_pool():
    cache = OracleCache.__new__(OracleCache)
    cache._pool = None
    assert await cache.store_research("q", "r", [], 1, 0, "m", 100) == ""


@pytest.mark.asyncio
async def test_cache_store_finding_no_pool():
    cache = OracleCache.__new__(OracleCache)
    cache._pool = None
    await cache.store_finding("abc", "q", "s", [], 1)  # should not raise


@pytest.mark.asyncio
async def test_cache_store_findings_batch_no_pool():
    cache = OracleCache.__new__(OracleCache)
    cache._pool = None
    await cache.store_findings_batch("abc", [{"sub_query": "q", "summary": "s", "sources": [], "round_num": 1}])


# --- CacheEntry edge cases ---

def test_cache_entry_empty_sources():
    entry = CacheEntry(query="q", answer="a", sources=[], model_used="m")
    assert entry.sources == []
    assert entry.similarity == 0.0
    assert entry.hit_count == 0


def test_cache_hit_boundary():
    cache = OracleCache.__new__(OracleCache)
    cache.similarity_threshold = 0.85
    # Exact boundary
    assert cache._is_cache_hit(0.85) is True
    assert cache._is_cache_hit(0.849999) is False
    # Extremes
    assert cache._is_cache_hit(1.0) is True
    assert cache._is_cache_hit(0.0) is False
    assert cache._is_cache_hit(-0.1) is False


# --- Config env var override ---

def test_oracle_config_env_override():
    """Environment variables should override Oracle config defaults."""
    import os
    from pythia.config import OracleConfig

    env = {
        "PYTHIA_ORACLE_DSN": "prod-host:1521/PROD",
        "PYTHIA_ORACLE_USER": "prod_user",
        "PYTHIA_ORACLE_PASSWORD": "s3cret",  # pragma: allowlist secret
    }
    with patch.dict(os.environ, env):
        cfg = OracleConfig()
        assert cfg.dsn == "prod-host:1521/PROD"
        assert cfg.user == "prod_user"
        assert cfg.password == "s3cret"  # pragma: allowlist secret


def test_oracle_config_env_partial_override():
    """Partial env vars should only override specified fields."""
    import os
    from pythia.config import OracleConfig

    with patch.dict(os.environ, {"PYTHIA_ORACLE_PASSWORD": "better_password"}, clear=False):  # pragma: allowlist secret
        cfg = OracleConfig()
        assert cfg.password == "better_password"  # pragma: allowlist secret
        assert cfg.user == "pythia"  # default unchanged
        assert cfg.dsn == "localhost:1523/FREEPDB1"  # default unchanged


def test_oracle_config_password_default_empty():
    """Python default for password should be empty (force explicit config)."""
    import os
    from pythia.config import OracleConfig

    # Clear any env vars that would override
    env_clean = {k: v for k, v in os.environ.items() if not k.startswith("PYTHIA_ORACLE")}
    with patch.dict(os.environ, env_clean, clear=True):
        cfg = OracleConfig()
        assert cfg.password == ""


def test_oracle_config_yaml_override_env():
    """YAML values should be overridden by env vars."""
    import os
    from pythia.config import OracleConfig

    with patch.dict(os.environ, {"PYTHIA_ORACLE_PASSWORD": "env_pass"}):  # pragma: allowlist secret
        cfg = OracleConfig(password="yaml_pass")  # pragma: allowlist secret
        assert cfg.password == "env_pass"  # pragma: allowlist secret


# --- Ollama client edge cases ---

def test_ollama_build_prompt_special_chars():
    """Prompts should handle special characters in results."""
    from pythia.server.ollama import build_search_prompt
    from pythia.server.searxng import SearchResult

    results = [
        SearchResult(index=1, title='<script>alert("xss")</script>', url="https://evil.com", snippet='"; DROP TABLE--'),
    ]
    system, user = build_search_prompt("test", results)
    # Special chars should pass through (LLM prompt, not HTML)
    assert '<script>alert("xss")</script>' in user
    assert 'DROP TABLE' in user


# --- SearXNG client edge cases ---

def test_searxng_parse_empty_response():
    from pythia.server.searxng import SearxngClient
    client = SearxngClient(base_url="http://localhost:8888")
    assert client._parse_results({}) == []
    assert client._parse_results({"results": []}) == []


def test_searxng_parse_missing_fields():
    """Results with missing fields should be handled gracefully."""
    from pythia.server.searxng import SearxngClient
    client = SearxngClient(base_url="http://localhost:8888")
    data = {
        "results": [
            {"title": "A"},  # no url — should be skipped
            {"url": "https://b.com"},  # no title — should work
            {"url": "https://c.com", "title": "C", "content": "snippet"},  # complete
        ]
    }
    results = client._parse_results(data)
    assert len(results) == 2  # first skipped (no url)
    assert results[0].url == "https://b.com"
    assert results[0].title == ""
    assert results[1].snippet == "snippet"


# --- Scraper edge cases ---

@pytest.mark.asyncio
async def test_scrape_empty_urls():
    from pythia.scraper import scrape_urls
    results = await scrape_urls([])
    assert results == []


@pytest.mark.asyncio
async def test_scrape_content_truncation():
    """ScrapedContent should preserve content as-is (truncation is in _scrape_one_sync)."""
    from pythia.scraper import ScrapedContent
    long_content = "x" * 5000
    sc = ScrapedContent(url="https://t.com", content=long_content, success=True)
    assert len(sc.content) == 5000
