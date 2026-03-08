"""Tests for SearXNG client."""
import pytest
from pythia.server.searxng import SearxngClient, SearchResult


@pytest.fixture
def mock_searxng_response():
    return {
        "results": [
            {
                "title": "RLHF Overview",
                "url": "https://example.com/rlhf",
                "content": "RLHF is a technique for aligning language models.",
            },
            {
                "title": "Learning from Human Feedback",
                "url": "https://example.com/lhf",
                "content": "Reinforcement learning from human feedback explained.",
            },
        ]
    }


@pytest.mark.asyncio
async def test_parse_results(mock_searxng_response):
    client = SearxngClient(base_url="http://localhost:8888", max_results=8)
    results = client._parse_results(mock_searxng_response)
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].title == "RLHF Overview"
    assert results[0].url == "https://example.com/rlhf"
    assert results[0].snippet == "RLHF is a technique for aligning language models."
    assert results[0].index == 1
    assert results[1].index == 2


@pytest.mark.asyncio
async def test_parse_results_respects_max():
    client = SearxngClient(base_url="http://localhost:8888", max_results=1)
    data = {
        "results": [
            {"title": "A", "url": "https://a.com", "content": "aaa"},
            {"title": "B", "url": "https://b.com", "content": "bbb"},
        ]
    }
    results = client._parse_results(data)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_parse_results_deduplicates():
    client = SearxngClient(base_url="http://localhost:8888", max_results=8)
    data = {
        "results": [
            {"title": "A", "url": "https://a.com", "content": "aaa"},
            {"title": "A duplicate", "url": "https://a.com", "content": "aaa again"},
            {"title": "B", "url": "https://b.com", "content": "bbb"},
        ]
    }
    results = client._parse_results(data)
    assert len(results) == 2
