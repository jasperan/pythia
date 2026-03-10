"""CLI entry point for Pythia."""
from __future__ import annotations

import sys

import typer

app = typer.Typer(name="pythia", help="Pythia — self-hosted AI search engine")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="API server host"),
    port: int = typer.Option(8900, help="API server port"),
    config: str = typer.Option("pythia.yaml", help="Config file path"),
) -> None:
    """Start the Pythia API server."""
    import uvicorn
    from pythia.config import load_config
    from pythia.server.app import create_app

    cfg = load_config(config)
    application = create_app(cfg)
    uvicorn.run(application, host=host, port=port)


@app.command("search")
def search_tui(
    config: str = typer.Option("pythia.yaml", help="Config file path"),
    model: str = typer.Option("", help="Override Ollama model"),
    no_auto_start: bool = typer.Option(False, help="Disable automatic service startup"),
    host: str = typer.Option("", help="Override API server host"),
    port: int = typer.Option(0, help="Override API server port"),
) -> None:
    """Launch the Pythia TUI with automatic service startup."""
    from pythia.config import load_config
    from pythia.tui.app import run_tui

    cfg = load_config(config)
    if model:
        cfg.ollama.model = model

    auto_start = not no_auto_start
    host_override = host if host else None
    port_override = port if port else None

    run_tui(cfg, auto_start=auto_start, host=host_override, port=port_override)


@app.command()
def query(
    text: str = typer.Argument("", help="Search query (reads stdin if omitted)"),
    config: str = typer.Option("pythia.yaml", help="Config file path"),
    model: str = typer.Option("", help="Override Ollama model"),
    embed: bool = typer.Option(False, "--embed", help="Include query embedding in output"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Skip cache lookup/storage"),
    deep: bool = typer.Option(False, "--deep", help="Scrape top URLs for full content"),
    stream: bool = typer.Option(False, "--stream", help="Stream NDJSON events instead of flat JSON"),
) -> None:
    """Run a search query and return structured JSON."""
    import asyncio
    from pythia.config import load_config
    from pythia.cli_runner import run_query as _run_query

    query_text = text or sys.stdin.read().strip()
    if not query_text:
        print('{"error": "No query provided. Pass as argument or pipe via stdin."}', file=sys.stderr)
        raise typer.Exit(1)

    cfg = load_config(config)
    model_override = model if model else None

    asyncio.run(_run_query(
        cfg,
        query_text,
        include_embedding=embed,
        use_cache=not no_cache,
        model_override=model_override,
        deep=deep,
        stream=stream,
    ))


@app.command()
def research(
    text: str = typer.Argument("", help="Research question (reads stdin if omitted)"),
    config: str = typer.Option("pythia.yaml", help="Config file path"),
    model: str = typer.Option("", help="Override Ollama model"),
    stream: bool = typer.Option(False, "--stream", help="Stream NDJSON events instead of flat JSON"),
    max_rounds: int = typer.Option(0, "--max-rounds", help="Override max research rounds (0 = use config)"),
) -> None:
    """Run autonomous deep research on a topic. Returns structured report with citations."""
    import asyncio
    from pythia.config import load_config
    from pythia.cli_runner import run_research as _run_research

    query_text = text or sys.stdin.read().strip()
    if not query_text:
        print('{"error": "No query provided. Pass as argument or pipe via stdin."}', file=sys.stderr)
        raise typer.Exit(1)

    cfg = load_config(config)
    model_override = model if model else None
    rounds_override = max_rounds if max_rounds > 0 else None

    asyncio.run(_run_research(
        cfg,
        query_text,
        model_override=model_override,
        stream=stream,
        max_rounds=rounds_override,
    ))


@app.command()
def embed(
    text: str = typer.Argument("", help="Text to embed (reads stdin if omitted)"),
    config: str = typer.Option("pythia.yaml", help="Config file path"),
    file: str = typer.Option("", "--file", help="JSONL file for batch embedding"),
    store: bool = typer.Option(False, "--store", help="Store embedding in Oracle cache"),
) -> None:
    """Generate embeddings for text. Returns JSON with 384-dim vector."""
    import asyncio
    import json
    from pathlib import Path
    from pythia.cli_runner import run_embed_single, run_embed_batch

    texts: list[str] = []

    if file:
        p = Path(file)
        if not p.exists():
            print(f'{{"error": "File not found: {file}"}}', file=sys.stderr)
            raise typer.Exit(1)
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                texts.append(obj.get("text", line))
            except json.JSONDecodeError:
                texts.append(line)
        results = run_embed_batch(texts)
        for r in results:
            print(r)
    else:
        embed_text = text or sys.stdin.read().strip()
        if not embed_text:
            print('{"error": "No text provided. Pass as argument, pipe via stdin, or use --file."}',
                  file=sys.stderr)
            raise typer.Exit(1)
        texts = [embed_text]
        print(run_embed_single(embed_text))

    if store:
        from pythia.config import load_config
        cfg = load_config(config)
        asyncio.run(_store_embeddings(cfg, texts))


async def _store_embeddings(cfg, texts: list[str]) -> None:
    """Store embeddings in Oracle."""
    from pythia.server.oracle_cache import OracleCache
    cache = OracleCache(
        dsn=cfg.oracle.dsn,
        user=cfg.oracle.user,
        password=cfg.oracle.password,
        similarity_threshold=cfg.oracle.cache_similarity_threshold,
        embedding_model=cfg.oracle.embedding_model,
    )
    try:
        await cache.connect()
    except Exception as e:
        print(f'{{"error": "Oracle connection failed: {e}"}}', file=sys.stderr)
        return
    try:
        for t in texts:
            await cache.store(query=t, answer="", sources=[], model_used="embed-cli")
        print(f'{{"stored": {len(texts)}}}', file=sys.stderr)
    finally:
        await cache.close()


def main() -> None:
    app()
