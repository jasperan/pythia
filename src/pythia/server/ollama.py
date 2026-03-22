"""Ollama client — LLM inference and embedding generation."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from pythia.server.searxng import SearchResult


_SYSTEM_PROMPT = """You are Pythia, an AI search engine. Answer the user's question using ONLY the provided search results. Follow these rules strictly:

1. Cite sources using [N] notation inline (e.g., "LLMs are trained using RLHF [1].")
2. Be concise and accurate. Do not add information not found in the sources.
3. If the sources don't contain enough information, say so honestly.
4. Use markdown formatting for readability.
5. Start your answer directly — do not repeat the question."""


def build_search_prompt(query: str, results: list[SearchResult]) -> tuple[str, str]:
    """Build system and user prompts for search synthesis."""
    if not results:
        user = f"Question: {query}\n\nNo search results were found. Answer based on general knowledge and note the lack of sources."
        return _SYSTEM_PROMPT, user

    context_parts = []
    for r in results:
        context_parts.append(f"[{r.index}] {r.title}\nURL: {r.url}\n{r.snippet}")

    context = "\n\n".join(context_parts)
    user = f"Search Results:\n\n{context}\n\n---\n\nQuestion: {query}"
    return _SYSTEM_PROMPT, user


def build_deep_search_prompt(
    query: str, results: list[SearchResult], scraped_content: dict[str, str]
) -> tuple[str, str]:
    """Build prompt using full scraped page content where available, snippets as fallback."""
    if not results:
        user = f"Question: {query}\n\nNo search results were found. Answer based on general knowledge and note the lack of sources."
        return _SYSTEM_PROMPT, user

    context_parts = []
    for r in results:
        content = scraped_content.get(r.url, r.snippet)
        context_parts.append(f"[{r.index}] {r.title}\nURL: {r.url}\n{content}")

    context = "\n\n".join(context_parts)
    user = f"Search Results:\n\n{context}\n\n---\n\nQuestion: {query}"
    return _SYSTEM_PROMPT, user


class OllamaClient:
    """Async client for Ollama API — LLM inference only (embeddings handled by Oracle ONNX)."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init a reusable HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def generate_stream(self, system: str, user: str, model: str | None = None) -> AsyncIterator[str]:
        """Stream tokens from Ollama chat completion."""
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "think": False,
        }
        client = self._get_client()
        async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
                if chunk.get("done", False):
                    break

    async def generate(self, system: str, user: str, json_mode: bool = False, model: str | None = None) -> str:
        """Non-streaming generation. Returns complete response text."""
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
        }
        if json_mode:
            payload["format"] = "json"
        client = self._get_client()
        resp = await client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    async def health(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            client = self._get_client()
            resp = await client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False
