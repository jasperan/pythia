# Pythia

Self-hosted AI search engine: SearXNG (meta-search) + Ollama (LLM synthesis) + Oracle AI Vector Search (semantic caching via Python-generated embeddings stored as `VECTOR`).

## Architecture

```
Query -> SearchOrchestrator -> SearXNG (web results)
                            -> OracleCache (cosine similarity on pythia_cache.query_embedding)
                            -> Ollama (synthesize answer from results)
                            -> SSE stream back to client

Research mode: ResearchAgent runs multi-round iterative search with sub-query generation,
               claim verification, and provenance tracking (pythia_research table)
```

## Project Layout

- `src/pythia/server/app.py` — FastAPI app factory, all route definitions
- `src/pythia/server/search.py` — `SearchOrchestrator`: coordinates SearXNG + Oracle + Ollama
- `src/pythia/server/research.py` — `ResearchAgent`: multi-round deep research with sub-queries
- `src/pythia/server/oracle_cache.py` — `OracleCache`: async Oracle connection, cache CRUD, embeddings
- `src/pythia/server/grounding.py` — grounding/claim-verification layer
- `src/pythia/server/searxng.py` — `SearxngClient`: HTTP client for SearXNG API
- `src/pythia/server/ollama.py` — `OllamaClient`: HTTP client for Ollama `/api/chat`
- `src/pythia/tui/` — Textual TUI app with screens, themes, widgets, commands
- `src/pythia/cli.py` — Typer CLI (see CLI Commands below)
- `src/pythia/cli_runner.py` — headless query/research/embed runners (used by CLI)
- `src/pythia/config.py` — YAML config loader into Pydantic models (`pythia.yaml`)
- `src/pythia/embeddings.py` — sentence-transformers embedding generation (Python-side, not in-DB)
- `src/pythia/scraper.py` — web page scraper via `scrapling` (deep research)

## CLI Commands

```bash
pythia serve [--host 0.0.0.0] [--port 8900] [--config pythia.yaml]
pythia search [--config pythia.yaml] [--host ...] [--port ...]   # Textual TUI
pythia query <text> [--embed] [--no-cache] [--deep] [--stream]   # headless JSON
pythia research <text> [--stream] [--max-rounds N]               # headless research
pythia embed <text> [--file <jsonl>] [--store]                   # generate embeddings
```

All commands require `pythia.yaml` in cwd or pass `--config`. Config path can also come from `PYTHIA_CONFIG` env var.

## API Endpoints (port 8900)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/search` | SSE stream: web search + LLM synthesis |
| POST | `/research` | SSE stream: multi-round deep research (max 10 rounds) |
| GET | `/health` | Health check (Oracle, SearXNG, Ollama status) |
| GET | `/history` | Query history (limit param, default 20) |
| GET | `/stats` | Cache stats view |
| DELETE | `/cache` | Clear semantic cache |
| POST | `/embed` | Generate embedding for text |

## Configuration (`pythia.yaml`)

```yaml
server:   { host: "0.0.0.0", port: 8900 }
ollama:   { base_url: "http://localhost:11434", model: "qwen3.5:9b" }
searxng:  { base_url: "http://localhost:8889", max_results: 8, categories: [general] }
oracle:   { dsn: "localhost:1523/FREEPDB1", user: "pythia", password: "pythia",  # pragma: allowlist secret
            cache_similarity_threshold: 0.85, embedding_model: "ALL_MINILM_L6_V2" }
research: { max_rounds: 3, max_sub_queries: 5, deep_scrape: true, recall_threshold: 0.70 }
tui:      { theme: "dark" }
```

## Environment & Infrastructure

- **Conda env**: `pythia` (Python 3.12)
- **Docker services** (`docker compose up -d`):
  - `pythia-oracle` — Oracle 26ai Free Lite on port 1523
  - `pythia-searxng` — SearXNG on port 8889
- **Oracle DB**: `pythia/pythia@localhost:1523/FREEPDB1`
  - Tables: `pythia_cache` (VECTOR column + cosine index), `pythia_history`, `pythia_research`
  - View: `pythia_stats`
  - Embeddings are Python-generated via `sentence-transformers` (`ALL_MINILM_L6_V2`), stored with `TO_VECTOR(...)`. Oracle-side `VECTOR_EMBEDDING()` is optional/experimental.

## Setup

```bash
conda activate pythia
pip install -e ".[dev]"
docker compose up -d   # Oracle 26ai + SearXNG

# First-time DB schema (run as ADMIN, then PYTHIA user):
# See setup_schema.sql for step-by-step instructions
# Migration for research table (iterative investigations):
# migrations/002_iterative_investigations.sql

pythia serve           # API on :8900
pythia search          # TUI client
```

## Linting & Testing

```bash
ruff check src/          # linter
ruff format src/         # formatter
vulture src/             # dead code (test files excluded by config — run vulture tests/ manually)
pytest tests/ -v
pytest tests/ -v --cov=src/pythia   # with coverage
```

## Gotchas

- **Run from project root**: `pythia search` and `pythia serve` look for `pythia.yaml` and Docker assets in cwd. Use `--config` for absolute paths when running elsewhere.
- **Embeddings are Python-side**: `sentence-transformers` generates vectors in Python; they're stored via `TO_VECTOR(...)`. Don't confuse with Oracle's `VECTOR_EMBEDDING()` SQL function (supported but optional).
- **Oracle VECTOR index**: `pythia_cache` uses a `NEIGHBOR PARTITIONS` cosine index. Dropping and recreating the table requires re-running `setup_schema.sql`.
- **`scrapling` dep**: scraper uses `scrapling[fetchers]`; headless Playwright may be needed for JS-heavy pages.
- **`vulture` false positives**: Textual lifecycle hooks (`compose`, `on_*`, `action_*`) and Pydantic validators are excluded via `pyproject.toml` `ignore_names`. Don't add them to dead-code suppression in source.
