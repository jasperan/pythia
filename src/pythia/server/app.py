"""FastAPI server — exposes search as SSE endpoint."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from pythia.config import PythiaConfig
from pythia.server.ollama import OllamaClient
from pythia.server.oracle_cache import OracleCache
from pythia.server.searxng import SearxngClient
from pythia.server.search import SearchOrchestrator


class SearchRequest(BaseModel):
    query: str
    model: str | None = None


def create_app(config: PythiaConfig) -> FastAPI:
    ollama = OllamaClient(
        base_url=config.ollama.base_url,
        model=config.ollama.model,
        embedding_model=config.ollama.embedding_model,
    )
    cache = OracleCache(
        dsn=config.oracle.dsn,
        user=config.oracle.user,
        password=config.oracle.password,
        similarity_threshold=config.oracle.cache_similarity_threshold,
    )
    searxng = SearxngClient(
        base_url=config.searxng.base_url,
        max_results=config.searxng.max_results,
        categories=config.searxng.categories,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await cache.connect()
        yield
        await cache.close()

    app = FastAPI(title="Pythia", version="0.1.0", lifespan=lifespan)
    orchestrator = SearchOrchestrator(ollama=ollama, cache=cache, searxng=searxng)

    @app.post("/search")
    async def search(req: SearchRequest):
        if req.model:
            ollama.model = req.model

        async def event_generator():
            async for event in orchestrator.search(req.query):
                yield {"event": event.event_type.value, "data": json.dumps(event.data)}

        return EventSourceResponse(event_generator())

    @app.get("/health")
    async def health():
        oracle_ok = await cache.health()
        searxng_ok = await searxng.health()
        ollama_ok = await ollama.health()
        cache_size = await cache.get_cache_size()
        return {"oracle": oracle_ok, "searxng": searxng_ok, "ollama": ollama_ok, "cache_size": cache_size}

    @app.get("/history")
    async def history(limit: int = Query(20, ge=1, le=100)):
        return await cache.get_history(limit)

    @app.get("/stats")
    async def stats():
        return await cache.get_stats()

    @app.delete("/cache")
    async def clear_cache():
        deleted = await cache.clear_cache()
        return {"deleted": deleted}

    return app
