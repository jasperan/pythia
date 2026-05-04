"""CLI entry point for Pythia."""
from __future__ import annotations

import sys
import os
from pathlib import Path

import typer

app = typer.Typer(name="pythia", help="Pythia — self-hosted AI search engine")


def _load_required_config(
    config: str,
    *,
    command_name: str,
    auto_start: bool | None = None,
):
    """Load config for commands that require an explicit project config."""
    from pythia.config import DEFAULT_CONFIG_NAME, load_config, resolve_config_path

    resolved = resolve_config_path(config)
    if resolved is not None:
        return load_config(resolved), str(resolved)

    if Path(config) != Path(DEFAULT_CONFIG_NAME):
        prefix = f"Error: config file not found: {config}."
    elif "PYTHIA_CONFIG" in os.environ:
        env_path = os.environ["PYTHIA_CONFIG"]
        prefix = f"Error: config file not found: {env_path} (from PYTHIA_CONFIG)."
    else:
        prefix = "Error: no Pythia config found."

    guidance = "Use --config /abs/path/to/pythia.yaml or set PYTHIA_CONFIG."
    if command_name == "search":
        if auto_start:
            guidance += " Auto-start also needs access to the project Docker assets; run from the project root or use --no-auto-start once an explicit config is set."
        else:
            guidance += " Launch from the project root or keep using --no-auto-start with an explicit config."

    print(f"{prefix} {guidance}", file=sys.stderr)
    raise typer.Exit(2)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="API server host"),
    port: int = typer.Option(8900, help="API server port"),
    config: str = typer.Option("pythia.yaml", help="Config file path"),
    backend: str = typer.Option("", help="Override LLM backend (ollama or oci-genai)"),
) -> None:
    """Start the Pythia API server."""
    import uvicorn
    from pythia.server.app import create_app

    cfg, _ = _load_required_config(config, command_name="serve")
    if backend:
        cfg.backend = backend
    application = create_app(cfg)
    uvicorn.run(application, host=host, port=port)


@app.command("search")
def search_tui(
    config: str = typer.Option("pythia.yaml", help="Config file path"),
    model: str = typer.Option("", help="Override Ollama model"),
    backend: str = typer.Option("", help="Override LLM backend (ollama or oci-genai)"),
    no_auto_start: bool = typer.Option(False, help="Disable automatic service startup"),
    host: str = typer.Option("", help="Override API server host"),
    port: int = typer.Option(0, help="Override API server port"),
) -> None:
    """Launch the Pythia TUI with automatic service startup."""
    from pythia.tui.app import run_tui

    cfg, resolved_config = _load_required_config(
        config,
        command_name="search",
        auto_start=not no_auto_start,
    )
    if backend:
        cfg.backend = backend
    if model:
        cfg.ollama.model = model

    auto_start = not no_auto_start
    host_override = host if host else None
    port_override = port if port else None

    run_tui(
        cfg,
        auto_start=auto_start,
        host=host_override,
        port=port_override,
        config_path=resolved_config,
    )


@app.command()
def query(
    text: str = typer.Argument("", help="Search query (reads stdin if omitted)"),
    config: str = typer.Option("pythia.yaml", help="Config file path"),
    model: str = typer.Option("", help="Override Ollama model"),
    embed: bool = typer.Option(False, "--embed", help="Include query embedding in output"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Skip cache lookup/storage"),
    deep: bool = typer.Option(False, "--deep", help="Scrape top URLs for full content"),
    stream: bool = typer.Option(False, "--stream", help="Stream NDJSON events instead of flat JSON"),
    backend: str = typer.Option("", help="Override LLM backend (ollama or oci-genai)"),
) -> None:
    """Run a search query and return structured JSON."""
    import asyncio
    from pythia.cli_runner import run_query as _run_query

    query_text = text or sys.stdin.read().strip()
    if not query_text:
        print('{"error": "No query provided. Pass as argument or pipe via stdin."}', file=sys.stderr)
        raise typer.Exit(1)

    cfg, _ = _load_required_config(config, command_name="query")
    if backend:
        cfg.backend = backend
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
    backend: str = typer.Option("", help="Override LLM backend (ollama or oci-genai)"),
) -> None:
    """Run autonomous deep research on a topic. Returns structured report with citations."""
    import asyncio
    from pythia.cli_runner import run_research as _run_research

    query_text = text or sys.stdin.read().strip()
    if not query_text:
        print('{"error": "No query provided. Pass as argument or pipe via stdin."}', file=sys.stderr)
        raise typer.Exit(1)

    cfg, _ = _load_required_config(config, command_name="research")
    if backend:
        cfg.backend = backend
    model_override = model if model else None
    rounds_override = max_rounds if max_rounds > 0 else None

    asyncio.run(_run_research(
        cfg,
        query_text,
        model_override=model_override,
        stream=stream,
        max_rounds=rounds_override,
    ))


@app.command("research-continue")
def research_continue(
    slug: str = typer.Argument(..., help="Research slug to continue"),
    focus: str = typer.Option("", "--focus", "-f", help="Optional continuation focus"),
    config: str = typer.Option("pythia.yaml", help="Config file path"),
    model: str = typer.Option("", help="Override Ollama model"),
    stream: bool = typer.Option(False, "--stream", help="Stream NDJSON events instead of flat JSON"),
    max_rounds: int = typer.Option(0, "--max-rounds", help="Override max continuation rounds (0 = use config)"),
    backend: str = typer.Option("", help="Override LLM backend (ollama or oci-genai)"),
) -> None:
    """Continue a stored research session by slug."""
    import asyncio
    from pythia.cli_runner import run_continue_research as _run_continue_research

    cfg, _ = _load_required_config(config, command_name="research-continue")
    if backend:
        cfg.backend = backend
    model_override = model if model else None
    rounds_override = max_rounds if max_rounds > 0 else None

    asyncio.run(_run_continue_research(
        cfg,
        slug,
        focus=focus or None,
        model_override=model_override,
        stream=stream,
        max_rounds=rounds_override,
    ))


@app.command("research-refine")
def research_refine(
    slug: str = typer.Argument(..., help="Research slug to refine"),
    directive: str = typer.Argument(..., help="Refinement directive"),
    config: str = typer.Option("pythia.yaml", help="Config file path"),
    model: str = typer.Option("", help="Override Ollama model"),
    stream: bool = typer.Option(False, "--stream", help="Stream NDJSON events instead of flat JSON"),
    max_rounds: int = typer.Option(0, "--max-rounds", help="Override max refinement rounds (0 = use config)"),
    backend: str = typer.Option("", help="Override LLM backend (ollama or oci-genai)"),
) -> None:
    """Refine a stored research session by slug with a focused directive."""
    import asyncio
    from pythia.cli_runner import run_refine_research as _run_refine_research

    if not directive.strip():
        print('{"error": "No refinement directive provided."}', file=sys.stderr)
        raise typer.Exit(1)

    cfg, _ = _load_required_config(config, command_name="research-refine")
    if backend:
        cfg.backend = backend
    model_override = model if model else None
    rounds_override = max_rounds if max_rounds > 0 else None

    asyncio.run(_run_refine_research(
        cfg,
        slug,
        directive=directive,
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
        cfg, _ = _load_required_config(config, command_name="embed")
        asyncio.run(_store_embeddings(cfg, texts))


async def _store_embeddings(cfg, texts: list[str]) -> None:
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


skill_app = typer.Typer(help="Manage research skills")
app.add_typer(skill_app, name="skill")


@skill_app.command("list")
def skill_list(
    config: str = typer.Option("pythia.yaml", help="Config file path"),
) -> None:
    from pythia.skills import SkillLoader

    project_root = Path(__file__).parent.parent.parent.parent
    skills_dir = project_root / "skills"
    loader = SkillLoader(skills_dir)

    skills = loader.list_skills()
    for s in skills:
        triggers = ", ".join(s.triggers[:3])
        if len(s.triggers) > 3:
            triggers += f" (+{len(s.triggers) - 3} more)"
        print(f"  {s.name:20s} {s.description}")
        print(f"  {'':20s} triggers: {triggers}")
        print()


@skill_app.command("show")
def skill_show(
    name: str = typer.Argument(..., help="Skill name to inspect"),
    config: str = typer.Option("pythia.yaml", help="Config file path"),
) -> None:
    import json
    from pythia.skills import SkillLoader

    project_root = Path(__file__).parent.parent.parent.parent
    skills_dir = project_root / "skills"
    loader = SkillLoader(skills_dir)

    skill = loader.get(name)
    if not skill:
        print(f'{{"error": "Skill not found: {name}"}}', file=sys.stderr)
        raise typer.Exit(1)
    print(json.dumps({
        "name": skill.name,
        "description": skill.description,
        "triggers": skill.triggers,
        "max_rounds": skill.max_rounds,
        "max_sub_queries": skill.max_sub_queries,
        "requires_scrape": skill.requires_scrape,
        "output_format": skill.output_format,
    }, indent=2))


@app.command()
def autoresearch(
    target: str = typer.Argument("", help="What to optimize (e.g. 'test speed', 'bundle size')"),
    benchmark: str = typer.Option("", "--benchmark", "-b", help="Benchmark command to run"),
    metric: str = typer.Option("", "--metric", "-m", help="Metric name to track"),
    direction: str = typer.Option("higher", "--direction", "-d", help="Direction: higher or lower is better"),
    max_iterations: int = typer.Option(10, "--max-iterations", "-n", help="Max optimization iterations"),
    files: list[str] | None = typer.Option(
        None,
        "--file",
        "-f",
        help="Editable file in scope. Repeat to allow multiple files.",
    ),
    config: str = typer.Option("pythia.yaml", help="Config file path"),
    model: str = typer.Option("", help="Override Ollama model"),
    stream: bool = typer.Option(False, "--stream", help="Stream NDJSON events"),
    backend: str = typer.Option("", help="Override LLM backend (ollama or oci-genai)"),
) -> None:
    import asyncio
    from pythia.cli_runner import run_autoresearch as _run_autoresearch

    if not target:
        print('{"error": "No optimization target provided. Pass as argument."}', file=sys.stderr)
        raise typer.Exit(1)
    if not benchmark:
        print('{"error": "No benchmark command provided. Use --benchmark."}', file=sys.stderr)
        raise typer.Exit(1)
    if not metric:
        print('{"error": "No metric name provided. Use --metric."}', file=sys.stderr)
        raise typer.Exit(1)

    cfg, _ = _load_required_config(config, command_name="autoresearch")
    if backend:
        cfg.backend = backend
    model_override = model if model else None

    asyncio.run(_run_autoresearch(
        cfg,
        target=target,
        benchmark_cmd=benchmark,
        metric_name=metric,
        metric_direction=direction,
        max_iterations=max_iterations,
        files_in_scope=files or [],
        model_override=model_override,
        stream=stream,
    ))


def main() -> None:
    app()
