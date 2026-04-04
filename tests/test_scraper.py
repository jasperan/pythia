"""Tests for scraper module."""
import pytest
from unittest.mock import patch
from pythia.scraper import scrape_urls, ScrapedContent, _scrape_one_sync


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
async def test_scrape_urls_uses_scrapling_content():
    urls_snippets = [
        ("https://example.com/1", "snippet 1"),
        ("https://example.com/2", "snippet 2"),
    ]
    fake_results = [
        ScrapedContent(url="https://example.com/1", content="full content 1", success=True),
        ScrapedContent(url="https://example.com/2", content="full content 2", success=True),
    ]

    with patch("pythia.scraper._scrape_one_sync", side_effect=fake_results):
        results = await scrape_urls(urls_snippets)

    assert len(results) == 2
    assert results[0].content == "full content 1"
    assert results[0].success is True
    assert results[1].content == "full content 2"
    assert results[1].success is True


@pytest.mark.asyncio
async def test_scrape_urls_falls_back_on_fetch_error():
    urls_snippets = [
        ("https://example.com/1", "snippet 1"),
    ]

    fake_result = ScrapedContent(
        url="https://example.com/1",
        content="snippet 1",
        success=False,
        error="boom",
    )

    with patch("pythia.scraper._scrape_one_sync", return_value=fake_result):
        results = await scrape_urls(urls_snippets)

    assert len(results) == 1
    assert results[0].content == "snippet 1"
    assert results[0].success is False
    assert results[0].error == "boom"


def test_scrape_one_sync_falls_back_on_fetch_error():
    with patch("pythia.scraper.Fetcher.get", side_effect=RuntimeError("boom")):
        result = _scrape_one_sync("https://example.com/1", "snippet 1")

    assert result.content == "snippet 1"
    assert result.success is False
    assert result.error == "boom"


def test_scrape_one_sync_falls_back_on_short_content():
    class FakePage:
        def get_all_text(self, ignore_tags=()):
            return "too short"

    with patch("pythia.scraper.Fetcher.get", return_value=FakePage()):
        result = _scrape_one_sync("https://example.com/1", "snippet 1")

    assert result.content == "snippet 1"
    assert result.success is False
    assert result.error == "insufficient content"
