"""OCI GenAI client — OpenAI-compatible HTTP backend for Pythia."""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from pythia.server.ollama import _SUGGESTIONS_PROMPT, _strip_json_fences

logger = logging.getLogger(__name__)


class OciGenAIClient:
    """Async client for an OpenAI-compatible OCI GenAI proxy."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "oci-genai",
        timeout_read: int = 180,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_read = timeout_read
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0, read=float(self.timeout_read), write=10.0, pool=10.0
                ),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def generate_stream(
        self, system: str, user: str, model: str | None = None
    ) -> AsyncIterator[str]:
        """Stream tokens from an OpenAI-compatible chat completion endpoint."""
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
        }
        client = self._get_client()
        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content", "")
                if content:
                    yield content

    async def generate(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        model: str | None = None,
    ) -> str:
        """Non-streaming generation. Returns complete response text."""
        payload: dict = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        client = self._get_client()
        resp = await client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content", "") or ""
        if json_mode and content:
            content = _strip_json_fences(content)
        return content

    async def generate_suggestions(
        self, query: str, answer: str, model: str | None = None
    ) -> list[str]:
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
        except (httpx.HTTPError, json.JSONDecodeError, ValueError):
            logger.debug("Suggestion generation failed", exc_info=True)
            return []

    async def health(self) -> bool:
        """Check if the OCI GenAI proxy is reachable."""
        try:
            client = self._get_client()
            resp = await client.get(f"{self.base_url}/health", headers=self._headers())
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
