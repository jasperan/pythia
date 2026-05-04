"""Async runners for CLI commands — direct mode, no API server needed."""
from __future__ import annotations

import json
import sys

from pythia.config import PythiaConfig
from pythia.embeddings import generate_embedding_list, MODEL_NAME, DIMENSIONS
from pythia.server.llm_client import LLMClient, create_llm_client
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


def _build_clients(
    cfg: PythiaConfig, model_override: str | None = None
) -> tuple[LLMClient, OracleCache, SearxngClient]:
    """Create the standard client trio from config."""
    ollama = create_llm_client(cfg, model_override=model_override)
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
    ollama, cache, searxng = _build_clients(cfg, model_override=model_override)

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
    ollama, cache, searxng = _build_clients(cfg, model_override=model_override)

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


async def run_continue_research(
    cfg: PythiaConfig,
    slug: str,
    *,
    focus: str | None = None,
    model_override: str | None = None,
    stream: bool = False,
    max_rounds: int | None = None,
) -> None:
    """Continue a stored research session by slug and print results to stdout."""
    ollama, cache, searxng = _build_clients(cfg, model_override=model_override)

    cache_connected = False
    try:
        await cache.connect()
        cache_connected = True
    except Exception as e:
        print(f"Warning: Oracle cache unavailable ({e}), continuation cannot load prior research", file=sys.stderr)

    research_config = cfg.research
    if max_rounds is not None:
        research_config = research_config.model_copy(update={"max_rounds": max_rounds})

    agent = ResearchAgent(ollama=ollama, cache=cache, searxng=searxng, config=research_config)
    events = agent.continue_research(slug, focus=focus, model_override=model_override)

    try:
        if stream:
            await _stream_research_events(events)
        else:
            await _flat_research_events(agent, slug, events, model_override)
    finally:
        if cache_connected:
            await cache.close()


async def run_refine_research(
    cfg: PythiaConfig,
    slug: str,
    directive: str,
    *,
    model_override: str | None = None,
    stream: bool = False,
    max_rounds: int | None = None,
) -> None:
    """Refine a stored research session by slug and print results to stdout."""
    ollama, cache, searxng = _build_clients(cfg, model_override=model_override)

    cache_connected = False
    try:
        await cache.connect()
        cache_connected = True
    except Exception as e:
        print(f"Warning: Oracle cache unavailable ({e}), refinement cannot load prior research", file=sys.stderr)

    research_config = cfg.research
    if max_rounds is not None:
        research_config = research_config.model_copy(update={"max_rounds": max_rounds})

    agent = ResearchAgent(ollama=ollama, cache=cache, searxng=searxng, config=research_config)
    events = agent.refine_research(slug, directive=directive, model_override=model_override)

    try:
        if stream:
            await _stream_research_events(events)
        else:
            await _flat_research_events(agent, slug, events, model_override)
    finally:
        if cache_connected:
            await cache.close()


async def _stream_research(
    agent: ResearchAgent,
    query: str,
    model_override: str | None,
) -> None:
    """Print NDJSON research events to stdout."""
    await _stream_research_events(agent.research(query, model_override=model_override))


async def _stream_research_events(event_stream) -> None:
    async for event in event_stream:
        line = json.dumps({"event": event.event_type.value, "data": event.data})
        print(line, flush=True)


async def _flat_research(
    agent: ResearchAgent,
    query: str,
    model_override: str | None,
) -> None:
    """Collect all events and print a single flat JSON research report."""
    await _flat_research_events(
        agent,
        query,
        agent.research(query, model_override=model_override),
        model_override,
    )


async def _flat_research_events(
    agent: ResearchAgent,
    query: str,
    event_stream,
    model_override: str | None,
) -> None:
    tokens = []
    findings = []
    plan = []
    recalled = []
    done_data = {}

    async for event in event_stream:
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
        "corpus_path": done_data.get("corpus_path"),
        "model": model_override or agent.ollama.model,
    }
    for optional_key in [
        "error",
        "verification_status",
        "continued_from",
        "refined_from",
        "directive",
        "prior_findings_loaded",
        "failed_findings",
    ]:
        if optional_key in done_data:
            result[optional_key] = done_data[optional_key]

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


async def run_autoresearch(
    cfg: PythiaConfig,
    *,
    target: str,
    benchmark_cmd: str,
    metric_name: str,
    metric_direction: str = "higher",
    max_iterations: int = 10,
    files_in_scope: list[str] | None = None,
    model_override: str | None = None,
    stream: bool = False,
) -> None:
    from pathlib import Path
    from pythia.autoresearch import AutoresearchAgent, AutoresearchEventType

    ollama = create_llm_client(cfg, model_override=model_override)

    agent = AutoresearchAgent(ollama=ollama, workspace_dir=Path.cwd())
    scoped_files = files_in_scope or []

    if stream:
        async for event in agent.run(
            metric_name=metric_name,
            benchmark_cmd=benchmark_cmd,
            files_in_scope=scoped_files,
            metric_direction=metric_direction,
            max_iterations=max_iterations,
            model=model_override or cfg.ollama.model,
            target=target,
        ):
            line = json.dumps({"event": event.event_type.value, "data": event.data})
            print(line, flush=True)
    else:
        iterations = []
        best_metric = None
        best_iteration = 0

        async for event in agent.run(
            metric_name=metric_name,
            benchmark_cmd=benchmark_cmd,
            files_in_scope=scoped_files,
            metric_direction=metric_direction,
            max_iterations=max_iterations,
            model=model_override or cfg.ollama.model,
            target=target,
        ):
            if event.event_type == AutoresearchEventType.STATUS:
                print(json.dumps({"status": event.data.get("message", "")}), file=sys.stderr)
            elif event.event_type == AutoresearchEventType.BASELINE:
                print(json.dumps({
                    "baseline": event.data.get("metric_value"),
                    "metric": event.data.get("metric_name"),
                }), file=sys.stderr)
            elif event.event_type == AutoresearchEventType.METRIC:
                iterations.append(event.data)
                if event.data.get("improved"):
                    best_metric = event.data.get("metric_value")
                    best_iteration = event.data.get("iteration")
            elif event.event_type == AutoresearchEventType.DONE:
                result = {
                    "target": target,
                    "best_metric": best_metric,
                    "best_iteration": best_iteration,
                    "total_iterations": len(iterations),
                    "iterations": iterations,
                    "improvement_pct": event.data.get("improvement", 0),
                    "elapsed_ms": event.data.get("elapsed_ms", 0),
                }
                print(json.dumps(result))
