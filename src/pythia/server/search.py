"""Search orchestrator — ties SearXNG, Ollama, and Oracle cache together."""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum

from pythia.server.grounding import verify_grounding
from pythia.server.ollama import OllamaClient, build_search_prompt, build_deep_search_prompt
from pythia.server.oracle_cache import OracleCache
from pythia.server.searxng import SearxngClient
from pythia.scraper import scrape_urls


class EventType(str, Enum):
    STATUS = "status"
    SOURCE = "source"
    TOKEN = "token"
    DONE = "done"
    SUGGESTIONS = "suggestions"
    GROUNDING = "grounding"


@dataclass
class SearchEvent:
    event_type: EventType
    data: dict


def _count_citations(text: str) -> int:
    """Count unique [N] citation references in an answer."""
    return len(set(re.findall(r'\[(\d+)\]', text)))


class SearchOrchestrator:
    def __init__(self, ollama: OllamaClient, cache: OracleCache, searxng: SearxngClient):
        self.ollama = ollama
        self.cache = cache
        self.searxng = searxng

    async def rewrite_query(self, query: str, model: str | None = None) -> str:
        """Use LLM to rewrite a conversational query into a search-optimized one."""
        system = (
            "Rewrite the user's question into a concise, search-engine-optimized query. "
            "Return ONLY the rewritten query, nothing else. No quotes, no explanation."
        )
        try:
            result = await self.ollama.generate(system, query, model=model)
            rewritten = result.strip().strip('"').strip("'")
            return rewritten if len(rewritten) > 2 else query
        except Exception:
            return query

    async def search(
        self, query: str, model_override: str | None = None,
        deep: bool = False, rewrite: bool = False,
        conversation_history: list[dict] | None = None,
    ) -> AsyncIterator[SearchEvent]:
        start = time.monotonic()
        model = model_override or self.ollama.model
        original_query = query

        # Innovation 1: Query rewriting for conversational inputs
        if rewrite:
            yield SearchEvent(EventType.STATUS, {"message": "Optimizing query..."})
            query = await self.rewrite_query(query, model=model)
            if query != original_query:
                yield SearchEvent(EventType.STATUS, {"message": f"Rewritten: {query}"})

        # Innovation 2: Parallel cache + web search prefetch
        yield SearchEvent(EventType.STATUS, {"message": "Searching..."})
        cache_task = asyncio.create_task(self.cache.lookup(query))
        search_task = asyncio.create_task(self.searxng.search(query))

        cached, query_embedding = await cache_task

        if cached:
            search_task.cancel()
            try:
                await search_task
            except (asyncio.CancelledError, Exception):
                pass

            elapsed_ms = int((time.monotonic() - start) * 1000)
            yield SearchEvent(EventType.STATUS, {"message": f"Cache hit ({cached.similarity:.2f} similarity)"})

            for source in cached.sources:
                yield SearchEvent(EventType.SOURCE, source)

            chunk_size = 20
            for i in range(0, len(cached.answer), chunk_size):
                yield SearchEvent(EventType.TOKEN, {"content": cached.answer[i : i + chunk_size]})

            # Innovation 4: Grounding verification on cached answers too
            grounding = verify_grounding(cached.answer, cached.sources)
            yield SearchEvent(EventType.GROUNDING, {
                "score": grounding.score,
                "label": grounding.label,
                "total_claims": grounding.total_claims,
                "grounded_claims": grounding.grounded_claims,
            })

            await self.cache.record_search(query, cache_hit=True, response_time_ms=elapsed_ms, model_used=cached.model_used)

            # Innovation 5: Follow-up suggestions (before DONE so clients don't close early)
            suggestions = await self.ollama.generate_suggestions(query, cached.answer, model=model)
            if suggestions:
                yield SearchEvent(EventType.SUGGESTIONS, {"suggestions": suggestions})

            yield SearchEvent(
                EventType.DONE,
                {"cache_hit": True, "similarity": cached.similarity, "response_time_ms": elapsed_ms, "sources_count": len(cached.sources)},
            )
            return

        # Cache miss — web search is already running
        yield SearchEvent(EventType.STATUS, {"message": "Searching web..."})
        try:
            results = await search_task
        except Exception as e:
            yield SearchEvent(EventType.STATUS, {"message": f"SearXNG error: {e}"})
            yield SearchEvent(EventType.DONE, {"cache_hit": False, "error": str(e), "response_time_ms": int((time.monotonic() - start) * 1000)})
            return
        yield SearchEvent(EventType.STATUS, {"message": f"Found {len(results)} results"})

        for r in results:
            yield SearchEvent(EventType.SOURCE, {"index": r.index, "title": r.title, "url": r.url, "snippet": r.snippet})

        if deep:
            yield SearchEvent(EventType.STATUS, {"message": "Scraping pages for full content..."})
            urls_snippets = [(r.url, r.snippet) for r in results[:3]]
            scraped = await scrape_urls(urls_snippets)
            scraped_content = {s.url: s.content for s in scraped}
            success_count = sum(1 for s in scraped if s.success)
            yield SearchEvent(EventType.STATUS, {"message": f"Scraped {success_count}/{len(scraped)} pages"})
            system, user = build_deep_search_prompt(query, results, scraped_content, conversation_history)
        else:
            system, user = build_search_prompt(query, results, conversation_history)

        yield SearchEvent(EventType.STATUS, {"message": "Synthesizing answer..."})

        full_answer = []
        try:
            async for token in self.ollama.generate_stream(system, user, model=model):
                full_answer.append(token)
                yield SearchEvent(EventType.TOKEN, {"content": token})
        except Exception as e:
            yield SearchEvent(EventType.STATUS, {"message": f"Ollama error: {e}"})
            yield SearchEvent(EventType.DONE, {"cache_hit": False, "error": str(e), "response_time_ms": int((time.monotonic() - start) * 1000)})
            return

        answer_text = "".join(full_answer)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Innovation 3: Citation density metric — quality signal with zero extra latency
        citation_count = _count_citations(answer_text)
        citation_density = citation_count / max(len(results), 1)

        sources_dicts = [{"index": r.index, "title": r.title, "url": r.url, "snippet": r.snippet} for r in results]

        # Innovation 4: Answer grounding — verify claims against sources
        grounding = verify_grounding(answer_text, sources_dicts)
        yield SearchEvent(EventType.GROUNDING, {
            "score": grounding.score,
            "label": grounding.label,
            "total_claims": grounding.total_claims,
            "grounded_claims": grounding.grounded_claims,
        })

        await self.cache.store(query=query, answer=answer_text, sources=sources_dicts, model_used=model, query_embedding=query_embedding)
        await self.cache.record_search(query, cache_hit=False, response_time_ms=elapsed_ms, model_used=model)

        # Innovation 5: Follow-up suggestions (before DONE so clients don't close early)
        suggestions = await self.ollama.generate_suggestions(query, answer_text, model=model)
        if suggestions:
            yield SearchEvent(EventType.SUGGESTIONS, {"suggestions": suggestions})

        yield SearchEvent(EventType.DONE, {
            "cache_hit": False,
            "response_time_ms": elapsed_ms,
            "sources_count": len(results),
            "citations_used": citation_count,
            "citation_density": round(citation_density, 2),
            "grounding_score": grounding.score,
            "grounding_label": grounding.label,
        })
