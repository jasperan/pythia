"""FastAPI server — exposes search as SSE endpoint."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from pythia.config import PythiaConfig
from pythia.server.ollama import OllamaClient
from pythia.server.oracle_cache import OracleCache
from pythia.server.searxng import SearxngClient
from pythia.server.search import SearchOrchestrator


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


def create_app(config: PythiaConfig) -> FastAPI:
    ollama = OllamaClient(
        base_url=config.ollama.base_url,
        model=config.ollama.model,
    )
    cache = OracleCache(
        dsn=config.oracle.dsn,
        user=config.oracle.user,
        password=config.oracle.password,
        similarity_threshold=config.oracle.cache_similarity_threshold,
        embedding_model=config.oracle.embedding_model,
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
        async def event_generator():
            async for event in orchestrator.search(req.query, model_override=req.model):
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

    @app.post("/embed")
    def embed_text(req: EmbedRequest):
        from pythia.embeddings import generate_embedding_list, MODEL_NAME, DIMENSIONS
        embedding = generate_embedding_list(req.text)
        return {"text": req.text, "embedding": embedding, "dimensions": DIMENSIONS, "model": MODEL_NAME}

    return app
