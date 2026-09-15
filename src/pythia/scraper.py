"""Deep scraping via Scrapling with per-URL fallback."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from scrapling.fetchers import Fetcher

logger = logging.getLogger(__name__)

_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def _fetch_public(url: str, *, max_redirects: int = _MAX_REDIRECTS) -> tuple[object, str]:
    """Fetch ``url``, validating every redirect hop before it is followed.

    ``_is_public_http_url`` can only vet the URL it is handed, so letting the fetcher follow
    redirects internally gave a public host a way to reach a private address: the redirect target
    was never checked. Redirects are therefore followed one hop at a time here, and each Location is
    passed through the same predicate before another request is made.

    Returns the response and the URL that produced it.
    """
    current = url
    for _ in range(max_redirects + 1):
        page = Fetcher.get(current, timeout=10, follow_redirects=False)
        if getattr(page, "status", 200) not in _REDIRECT_STATUSES:
            return page, current
        headers = getattr(page, "headers", None) or {}
        location = headers.get("location") or headers.get("Location")
        if not location:
            return page, current
        next_url = urljoin(current, location)
        if not _is_public_http_url(next_url):
            raise ValueError(f"blocked redirect to non-public URL: {next_url}")
        current = next_url
    raise ValueError(f"too many redirects from {url}")


@dataclass
class ScrapedContent:
    url: str
    content: str
    success: bool
    error: str = ""


def _scrape_one_sync(url: str, fallback_snippet: str) -> ScrapedContent:
    """Scrape a single URL synchronously. Returns fallback on failure."""
    if not _is_public_http_url(url):
        return ScrapedContent(
            url=url,
            content=fallback_snippet,
            success=False,
            error="blocked non-public URL",
        )

    try:
        page, _final_url = _fetch_public(url)
        text = page.get_all_text(ignore_tags=("script", "style", "nav", "footer", "header"))
        if not text or len(text.strip()) < 50:
            return ScrapedContent(
                url=url, content=fallback_snippet, success=False, error="insufficient content"
            )
        content = text.strip()[:4000]
        return ScrapedContent(url=url, content=content, success=True)
    except Exception as e:
        logger.debug(f"Scrape failed for {url}: {e}")
        return ScrapedContent(url=url, content=fallback_snippet, success=False, error=str(e))


def _is_public_http_url(url: str) -> bool:
    """Return True for HTTP(S) URLs that do not resolve to local/private addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return False

    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        return literal.is_global

    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False

    for *_, sockaddr in infos:
        try:
            address = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            return False
        if not address.is_global:
            return False
    return True


async def scrape_urls(
    urls_snippets: list[tuple[str, str]],
    max_concurrent: int = 3,
) -> list[ScrapedContent]:
    """Scrape multiple URLs concurrently, falling back to snippets on scrape failure."""
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(max_concurrent)

    async def _scrape_with_sem(url: str, snippet: str) -> ScrapedContent:
        async with sem:
            return await loop.run_in_executor(None, _scrape_one_sync, url, snippet)

    tasks = [_scrape_with_sem(url, snippet) for url, snippet in urls_snippets]
    return list(await asyncio.gather(*tasks))
