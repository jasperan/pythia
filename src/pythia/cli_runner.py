"""Async runners for CLI commands — direct mode, no API server needed."""
from __future__ import annotations

import json
import sys

from pythia.config import PythiaConfig
from pythia.embeddings import generate_embedding_list, MODEL_NAME, DIMENSIONS
from pythia.server.ollama import OllamaClient
from pythia.server.oracle_cache import OracleCache
from pythia.server.research import ResearchAgent, ResearchEventType
from pythia.server.search import EventType, SearchOrchestrator
from pythia.server.searxng import SearxngClient


def run_embed_single(text: str) -> str:
    """Generate embedding for a single text. Returns JSON string."""
    embedding = generate_embedding_list(text)
    return json.dumps({
        "text": text,
        "embedding": embedding,
        "dimensions": DIMENSIONS,
        "model": MODEL_NAME,
    })


def run_embed_batch(texts: list[str]) -> list[str]:
    """Generate embeddings for multiple texts. Returns list of JSON strings."""
    return [run_embed_single(t) for t in texts]


def _build_clients(cfg: PythiaConfig) -> tuple[OllamaClient, OracleCache, SearxngClient]:
    """Create the standard client trio from config."""
    ollama = OllamaClient(base_url=cfg.ollama.base_url, model=cfg.ollama.model)
    cache = OracleCache(
        dsn=cfg.oracle.dsn,
        user=cfg.oracle.user,
        password=cfg.oracle.password,
        similarity_threshold=cfg.oracle.cache_similarity_threshold,
        embedding_model=cfg.oracle.embedding_model,
    )
    searxng = SearxngClient(
        base_url=cfg.searxng.base_url,
        max_results=cfg.searxng.max_results,
        categories=cfg.searxng.categories,
    )
    return ollama, cache, searxng


async def run_query(
    cfg: PythiaConfig,
    query: str,
    *,
    include_embedding: bool = False,
    use_cache: bool = True,
    model_override: str | None = None,
    deep: bool = False,
    stream: bool = False,
) -> None:
    """Run a search query and print results to stdout."""
    ollama, cache, searxng = _build_clients(cfg)

    if use_cache:
        try:
            await cache.connect()
        except Exception as e:
            print(f"Warning: Oracle cache unavailable ({e}), proceeding without cache", file=sys.stderr)
            use_cache = False

    orchestrator = SearchOrchestrator(ollama=ollama, cache=cache, searxng=searxng)

    try:
        if stream:
            await _stream_query(orchestrator, query, model_override, deep)
        else:
            await _flat_query(orchestrator, query, model_override, deep, include_embedding)
    finally:
        if use_cache:
            await cache.close()


async def run_research(
    cfg: PythiaConfig,
    query: str,
    *,
    model_override: str | None = None,
    stream: bool = False,
    max_rounds: int | None = None,
) -> None:
    """Run a deep research session and print results to stdout."""
    ollama, cache, searxng = _build_clients(cfg)

    cache_connected = False
    try:
        await cache.connect()
        cache_connected = True
    except Exception as e:
        print(f"Warning: Oracle cache unavailable ({e}), research recall disabled", file=sys.stderr)

    research_config = cfg.research
    if max_rounds is not None:
        research_config = research_config.model_copy(update={"max_rounds": max_rounds})

    agent = ResearchAgent(ollama=ollama, cache=cache, searxng=searxng, config=research_config)

    try:
        if stream:
            await _stream_research(agent, query, model_override)
        else:
            await _flat_research(agent, query, model_override)
    finally:
        if cache_connected:
            await cache.close()


async def _stream_research(
    agent: ResearchAgent,
    query: str,
    model_override: str | None,
) -> None:
    """Print NDJSON research events to stdout."""
    async for event in agent.research(query, model_override=model_override):
        line = json.dumps({"event": event.event_type.value, "data": event.data})
        print(line, flush=True)


async def _flat_research(
    agent: ResearchAgent,
    query: str,
    model_override: str | None,
) -> None:
    """Collect all events and print a single flat JSON research report."""
    tokens = []
    findings = []
    plan = []
    recalled = []
    done_data = {}

    async for event in agent.research(query, model_override=model_override):
        if event.event_type == ResearchEventType.TOKEN:
            tokens.append(event.data.get("content", ""))
        elif event.event_type == ResearchEventType.FINDING:
            findings.append(event.data)
        elif event.event_type == ResearchEventType.PLAN:
            plan = event.data.get("sub_queries", [])
        elif event.event_type == ResearchEventType.RECALL:
            recalled = event.data.get("findings", [])
        elif event.event_type == ResearchEventType.DONE:
            done_data = event.data
        elif event.event_type == ResearchEventType.STATUS:
            print(json.dumps({"status": event.data.get("message", "")}), file=sys.stderr)
        elif event.event_type == ResearchEventType.GAP_ANALYSIS:
            print(json.dumps({"gap_analysis": event.data}), file=sys.stderr)

    result = {
        "query": query,
        "report": "".join(tokens),
        "sub_queries": plan,
        "findings": findings,
        "recalled_findings": recalled,
        "rounds_used": done_data.get("rounds_used", 0),
        "total_findings": done_data.get("total_findings", 0),
        "total_sources": done_data.get("total_sources", 0),
        "elapsed_ms": done_data.get("elapsed_ms", 0),
        "model": model_override or agent.ollama.model,
    }

    print(json.dumps(result))


async def _stream_query(
    orchestrator: SearchOrchestrator,
    query: str,
    model_override: str | None,
    deep: bool,
) -> None:
    """Print NDJSON events to stdout."""
    async for event in orchestrator.search(query, model_override=model_override, deep=deep):
        line = json.dumps({"event": event.event_type.value, "data": event.data})
        print(line, flush=True)


async def _flat_query(
    orchestrator: SearchOrchestrator,
    query: str,
    model_override: str | None,
    deep: bool,
    include_embedding: bool,
) -> None:
    """Collect all events and print a single flat JSON object."""
    sources = []
    tokens = []
    done_data = {}

    async for event in orchestrator.search(query, model_override=model_override, deep=deep):
        if event.event_type == EventType.SOURCE:
            sources.append(event.data)
        elif event.event_type == EventType.TOKEN:
            tokens.append(event.data.get("content", ""))
        elif event.event_type == EventType.DONE:
            done_data = event.data
        elif event.event_type == EventType.STATUS:
            print(json.dumps({"status": event.data.get("message", "")}), file=sys.stderr)

    result = {
        "query": query,
        "answer": "".join(tokens),
        "sources": sources,
        "cache_hit": done_data.get("cache_hit", False),
        "similarity": done_data.get("similarity"),
        "response_time_ms": done_data.get("response_time_ms", 0),
        "model": model_override or orchestrator.ollama.model,
        "sources_count": done_data.get("sources_count", 0),
    }

    if include_embedding:
        result["embedding"] = generate_embedding_list(query)

    if "error" in done_data:
        result["error"] = done_data["error"]

    print(json.dumps(result))
