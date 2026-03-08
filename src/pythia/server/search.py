"""Search orchestrator — ties SearXNG, Ollama, and Oracle cache together."""
from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum

from pythia.server.ollama import OllamaClient, build_search_prompt
from pythia.server.oracle_cache import OracleCache
from pythia.server.searxng import SearxngClient


class EventType(str, Enum):
    STATUS = "status"
    SOURCE = "source"
    TOKEN = "token"
    DONE = "done"


@dataclass
class SearchEvent:
    event_type: EventType
    data: dict


class SearchOrchestrator:
    def __init__(self, ollama: OllamaClient, cache: OracleCache, searxng: SearxngClient):
        self.ollama = ollama
        self.cache = cache
        self.searxng = searxng

    async def search(self, query: str) -> AsyncIterator[SearchEvent]:
        start = time.monotonic()

        yield SearchEvent(EventType.STATUS, {"message": "Generating embedding..."})
        embedding = await self.ollama.embed(query)

        yield SearchEvent(EventType.STATUS, {"message": "Checking cache..."})
        cached = await self.cache.lookup(embedding)

        if cached:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            yield SearchEvent(EventType.STATUS, {"message": f"Cache hit ({cached.similarity:.2f} similarity)"})

            for source in cached.sources:
                yield SearchEvent(EventType.SOURCE, source)

            chunk_size = 20
            for i in range(0, len(cached.answer), chunk_size):
                yield SearchEvent(EventType.TOKEN, {"content": cached.answer[i : i + chunk_size]})

            await self.cache.record_search(query, cache_hit=True, response_time_ms=elapsed_ms, model_used=cached.model_used)
            yield SearchEvent(
                EventType.DONE,
                {"cache_hit": True, "similarity": cached.similarity, "response_time_ms": elapsed_ms, "sources_count": len(cached.sources)},
            )
            return

        yield SearchEvent(EventType.STATUS, {"message": "Searching web..."})
        results = await self.searxng.search(query)
        yield SearchEvent(EventType.STATUS, {"message": f"Found {len(results)} results"})

        for r in results:
            yield SearchEvent(EventType.SOURCE, {"index": r.index, "title": r.title, "url": r.url, "snippet": r.snippet})

        yield SearchEvent(EventType.STATUS, {"message": "Synthesizing answer..."})
        system, user = build_search_prompt(query, results)

        full_answer = []
        async for token in self.ollama.generate_stream(system, user):
            full_answer.append(token)
            yield SearchEvent(EventType.TOKEN, {"content": token})

        answer_text = "".join(full_answer)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        sources_dicts = [{"index": r.index, "title": r.title, "url": r.url, "snippet": r.snippet} for r in results]
        await self.cache.store(query=query, query_embedding=embedding, answer=answer_text, sources=sources_dicts, model_used=self.ollama.model)
        await self.cache.record_search(query, cache_hit=False, response_time_ms=elapsed_ms, model_used=self.ollama.model)

        yield SearchEvent(EventType.DONE, {"cache_hit": False, "response_time_ms": elapsed_ms, "sources_count": len(results)})
