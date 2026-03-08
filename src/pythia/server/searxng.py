"""SearXNG client — free, unlimited web search."""
from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class SearchResult:
    index: int
    title: str
    url: str
    snippet: str


class SearxngClient:
    """Async client for SearXNG JSON API."""

    def __init__(self, base_url: str, max_results: int = 8, categories: list[str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.max_results = max_results
        self.categories = categories or ["general"]

    async def search(self, query: str) -> list[SearchResult]:
        """Search SearXNG and return parsed results."""
        params = {
            "q": query,
            "format": "json",
            "categories": ",".join(self.categories),
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{self.base_url}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        return self._parse_results(data)

    def _parse_results(self, data: dict) -> list[SearchResult]:
        """Parse SearXNG JSON response into SearchResult objects."""
        raw = data.get("results", [])
        seen_urls: set[str] = set()
        results: list[SearchResult] = []
        for item in raw:
            url = item.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                SearchResult(
                    index=len(results) + 1,
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("content", ""),
                )
            )
            if len(results) >= self.max_results:
                break
        return results

    async def health(self) -> bool:
        """Check if SearXNG is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(self.base_url)
                return resp.status_code == 200
        except Exception:
            return False
