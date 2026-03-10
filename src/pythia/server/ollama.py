"""Ollama client — LLM inference and embedding generation."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

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

    async def generate_stream(self, system: str, user: str) -> AsyncIterator[str]:
        """Stream tokens from Ollama chat completion."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "think": False,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
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

    async def generate(self, system: str, user: str, json_mode: bool = False) -> str:
        """Non-streaming generation. Returns complete response text."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
        }
        if json_mode:
            payload["format"] = "json"
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")

    async def health(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
