"""Tests for scraper module."""

import socket
from unittest.mock import patch

import pytest

from pythia.scraper import (
    ScrapedContent,
    _fetch_public,
    _is_public_http_url,
    _scrape_one_sync,
    scrape_urls,
)


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
    with (
        patch("pythia.scraper._is_public_http_url", return_value=True),
        patch("pythia.scraper.Fetcher.get", side_effect=RuntimeError("boom")),
    ):
        result = _scrape_one_sync("https://example.com/1", "snippet 1")

    assert result.content == "snippet 1"
    assert result.success is False
    assert result.error == "boom"


def test_scrape_one_sync_falls_back_on_short_content():
    class FakePage:
        def get_all_text(self, ignore_tags=()):
            return "too short"

    with (
        patch("pythia.scraper._is_public_http_url", return_value=True),
        patch("pythia.scraper.Fetcher.get", return_value=FakePage()),
    ):
        result = _scrape_one_sync("https://example.com/1", "snippet 1")

    assert result.content == "snippet 1"
    assert result.success is False
    assert result.error == "insufficient content"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost/admin",
        "http://10.0.0.5/admin",
        "file:///etc/passwd",
    ],
)
def test_scrape_one_sync_blocks_non_public_urls(url):
    with patch("pythia.scraper.Fetcher.get") as fetch:
        result = _scrape_one_sync(url, "fallback")

    fetch.assert_not_called()
    assert result.content == "fallback"
    assert result.success is False
    assert result.error == "blocked non-public URL"


def test_is_public_http_url_blocks_unresolvable_hosts():
    with patch("pythia.scraper.socket.getaddrinfo", side_effect=socket.gaierror("no dns")):
        assert _is_public_http_url("https://unresolvable.example") is False


# ── Redirect handling ─────────────────────────────────────────────────────────
# _is_public_http_url can only vet the URL it is handed, so redirects are followed one hop at a
# time and each Location is validated before the next request. These tests pin that the fetcher is
# not allowed to follow them itself, and that a hop to a non-public address is refused.


def _fake_response(status: int = 200, location: str | None = None):
    class FakeResponse:
        def __init__(self) -> None:
            self.status = status
            self.headers = {"location": location} if location else {}

        def get_all_text(self, ignore_tags=()):
            return "plenty of scraped text to satisfy the length check " * 3

    return FakeResponse()


def test_fetch_public_disables_fetcher_redirects():
    with patch("pythia.scraper.Fetcher.get", return_value=_fake_response()) as fetch:
        _fetch_public("https://example.com/one")

    assert fetch.call_args.kwargs["follow_redirects"] is False


def test_fetch_public_follows_a_validated_redirect_chain():
    pages = [_fake_response(302, "https://example.com/two"), _fake_response(200)]

    with (
        patch("pythia.scraper.Fetcher.get", side_effect=pages) as fetch,
        patch("pythia.scraper._is_public_http_url", return_value=True),
    ):
        page, final_url = _fetch_public("https://example.com/one")

    assert fetch.call_count == 2
    assert final_url == "https://example.com/two"
    assert page.status == 200


def test_fetch_public_refuses_too_many_redirects():
    always_redirect = [_fake_response(302, "https://example.com/next")] * 10

    with (
        patch("pythia.scraper.Fetcher.get", side_effect=always_redirect),
        patch("pythia.scraper._is_public_http_url", return_value=True),
        pytest.raises(ValueError, match="too many redirects"),
    ):
        _fetch_public("https://example.com/one")


def test_scrape_one_sync_falls_back_when_a_redirect_target_is_not_public():
    def validator(url: str) -> bool:
        return "169.254.169.254" not in url

    with (
        patch("pythia.scraper._is_public_http_url", side_effect=validator),
        patch(
            "pythia.scraper.Fetcher.get",
            return_value=_fake_response(302, "http://169.254.169.254/latest/meta-data/"),
        ),
    ):
        result = _scrape_one_sync("https://example.com/one", "snippet")

    assert result.success is False
    assert "blocked redirect" in result.error
    assert result.content == "snippet"
