"""Ollama client — LLM inference and embedding generation."""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from pythia.server.searxng import SearchResult

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are Pythia, an AI search engine. Answer the user's question using ONLY the provided search results. Follow these rules strictly:

1. Cite sources using [N] notation inline (e.g., "LLMs are trained using RLHF [1].")
2. Be concise and accurate. Do not add information not found in the sources.
3. If the sources don't contain enough information, say so honestly.
4. Use markdown formatting for readability.
5. Start your answer directly — do not repeat the question.
6. If conversation history is provided, use it to understand context and resolve references (e.g., "tell me more about that" refers to the previous answer)."""


_SUGGESTIONS_PROMPT = """Based on the question and answer below, suggest exactly 3 follow-up questions the user might want to explore next. Return ONLY a JSON array of 3 strings, nothing else.

Question: {query}
Answer: {answer}

Return format: ["question 1", "question 2", "question 3"]"""


def build_search_prompt(
    query: str, results: list[SearchResult],
    conversation_history: list[dict] | None = None,
) -> tuple[str, str]:
    """Build system and user prompts for search synthesis."""
    if not results:
        user = f"Question: {query}\n\nNo search results were found. Answer based on general knowledge and note the lack of sources."
        return _SYSTEM_PROMPT, user

    context_parts = []
    for r in results:
        context_parts.append(f"[{r.index}] {r.title}\nURL: {r.url}\n{r.snippet}")

    context = "\n\n".join(context_parts)

    # Include conversation history for multi-turn context
    history_section = ""
    if conversation_history:
        history_parts = []
        for msg in conversation_history[-6:]:  # Last 3 exchanges max
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Truncate long previous answers to save context
            if role == "assistant" and len(content) > 500:
                content = content[:500] + "..."
            history_parts.append(f"{role.upper()}: {content}")
        history_section = "\n\nConversation History:\n" + "\n".join(history_parts) + "\n"

    user = f"Search Results:\n\n{context}\n\n---{history_section}\n\nQuestion: {query}"
    return _SYSTEM_PROMPT, user


def build_deep_search_prompt(
    query: str, results: list[SearchResult], scraped_content: dict[str, str],
    conversation_history: list[dict] | None = None,
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

    history_section = ""
    if conversation_history:
        history_parts = []
        for msg in conversation_history[-6:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant" and len(content) > 500:
                content = content[:500] + "..."
            history_parts.append(f"{role.upper()}: {content}")
        history_section = "\n\nConversation History:\n" + "\n".join(history_parts) + "\n"

    user = f"Search Results:\n\n{context}\n\n---{history_section}\n\nQuestion: {query}"
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

    async def generate_suggestions(self, query: str, answer: str, model: str | None = None) -> list[str]:
        """Generate follow-up question suggestions based on query and answer."""
        try:
            prompt = _SUGGESTIONS_PROMPT.format(query=query, answer=answer[:1000])
            result = await self.generate(
                "You are a helpful assistant. Return only valid JSON.",
                prompt,
                json_mode=True,
                model=model,
            )
            parsed = json.loads(result)
            if isinstance(parsed, list):
                return [str(s) for s in parsed[:3]]
            if isinstance(parsed, dict) and "suggestions" in parsed:
                return [str(s) for s in parsed["suggestions"][:3]]
            return []
        except Exception:
            logger.debug("Suggestion generation failed", exc_info=True)
            return []

    async def health(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            client = self._get_client()
            resp = await client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False
