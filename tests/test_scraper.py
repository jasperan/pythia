"""Tests for scraper module."""
import pytest
from unittest.mock import patch
from pythia.scraper import scrape_urls, ScrapedContent


def test_scraped_content_dataclass():
    sc = ScrapedContent(url="https://example.com", content="Hello", success=True)
    assert sc.url == "https://example.com"
    assert sc.content == "Hello"
    assert sc.success is True


def test_scraped_content_failure():
    sc = ScrapedContent(url="https://example.com", content="", success=False, error="timeout")
    assert sc.success is False
    assert sc.error == "timeout"


@pytest.mark.asyncio
async def test_scrape_urls_without_scrapling():
    """When scrapling is not installed, all URLs return fallback snippets."""
    urls_snippets = [
        ("https://example.com/1", "snippet 1"),
        ("https://example.com/2", "snippet 2"),
    ]
    with patch("pythia.scraper._SCRAPLING_AVAILABLE", False):
        results = await scrape_urls(urls_snippets)
    assert len(results) == 2
    assert results[0].content == "snippet 1"
    assert results[0].success is False
    assert results[1].content == "snippet 2"
