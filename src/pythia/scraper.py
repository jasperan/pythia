"""Optional deep scraping via Scrapling with per-URL fallback."""
from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from scrapling.fetchers import Fetcher
    _SCRAPLING_AVAILABLE = True
except ImportError:
    _SCRAPLING_AVAILABLE = False


@dataclass
class ScrapedContent:
    url: str
    content: str
    success: bool
    error: str = ""


def _scrape_one_sync(url: str, fallback_snippet: str) -> ScrapedContent:
    """Scrape a single URL synchronously. Returns fallback on failure."""
    if not _SCRAPLING_AVAILABLE:
        return ScrapedContent(url=url, content=fallback_snippet, success=False, error="scrapling not installed")
    try:
        page = Fetcher.get(url, timeout=10)
        text = page.get_all_text(ignore_tags=("script", "style", "nav", "footer", "header"))
        if not text or len(text.strip()) < 50:
            return ScrapedContent(url=url, content=fallback_snippet, success=False, error="insufficient content")
        content = text.strip()[:4000]
        return ScrapedContent(url=url, content=content, success=True)
    except Exception as e:
        logger.debug(f"Scrape failed for {url}: {e}")
        return ScrapedContent(url=url, content=fallback_snippet, success=False, error=str(e))


async def scrape_urls(
    urls_snippets: list[tuple[str, str]],
    max_concurrent: int = 3,
) -> list[ScrapedContent]:
    """Scrape multiple URLs concurrently, falling back to snippets on failure."""
    if not _SCRAPLING_AVAILABLE:
        print("Warning: scrapling not installed, using snippets only. Install with: pip install scrapling", file=sys.stderr)
        return [
            ScrapedContent(url=url, content=snippet, success=False, error="scrapling not installed")
            for url, snippet in urls_snippets
        ]

    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(max_concurrent)

    async def _scrape_with_sem(url: str, snippet: str) -> ScrapedContent:
        async with sem:
            return await loop.run_in_executor(None, _scrape_one_sync, url, snippet)

    tasks = [_scrape_with_sem(url, snippet) for url, snippet in urls_snippets]
    return list(await asyncio.gather(*tasks))
