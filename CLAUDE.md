# Pythia

Self-hosted AI search engine combining SearXNG (meta-search), Ollama (LLM synthesis), and Oracle AI Vector Search (ONNX in-database embeddings for semantic caching).

## Architecture

```
Query -> SearchOrchestrator -> SearXNG (web results)
                            -> OracleCache (semantic cache check via VECTOR_EMBEDDING)
                            -> Ollama (synthesize answer from results)
                            -> SSE stream back to client

Research mode: ResearchAgent runs multi-round iterative search with sub-query generation
```

## Project Layout

- `src/pythia/server/app.py` — FastAPI app factory, all route definitions
- `src/pythia/server/search.py` — `SearchOrchestrator`: coordinates SearXNG + Oracle + Ollama
- `src/pythia/server/research.py` — `ResearchAgent`: multi-round deep research with sub-queries
- `src/pythia/server/oracle_cache.py` — `OracleCache`: async Oracle connection, cache CRUD, embeddings
- `src/pythia/server/searxng.py` — `SearxngClient`: HTTP client for SearXNG API
- `src/pythia/server/ollama.py` — `OllamaClient`: HTTP client for Ollama `/api/chat`
- `src/pythia/tui/` — Textual TUI app with screens, themes, widgets
- `src/pythia/cli.py` — Typer CLI: `pythia serve`, `pythia search`
- `src/pythia/config.py` — YAML config loader into Pydantic models (`pythia.yaml`)
- `src/pythia/embeddings.py` — Embedding utilities
- `src/pythia/scraper.py` — Web page content scraper (for deep research)

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

## Configuration

All config via `pythia.yaml` (Pydantic defaults if missing):

| Section | Key defaults |
|---------|-------------|
| `server` | host: 0.0.0.0, port: 8900 |
| `ollama` | base_url: localhost:11434, model: qwen3.5:9b |
| `searxng` | base_url: localhost:8889, max_results: 8 |
| `oracle` | dsn: localhost:1523/FREEPDB1, user/pass: pythia/pythia, cache_similarity: 0.85 |
| `research` | max_rounds: 3, max_sub_queries: 5, deep_scrape: true |

## Environment

- **Conda env**: `pythia` (Python 3.12)
- **Oracle DB**: `pythia/pythia@localhost:1523/FREEPDB1` (Oracle 26ai ADB-Free via docker-compose)
  - Tables: `pythia_cache` (with VECTOR column), `pythia_history`; View: `pythia_stats`
  - Embeddings: ONNX `ALL_MINILM_L6_V2` in-database (Ollama is LLM-only, no embeddings)

## Setup & Running

```bash
conda activate pythia

# Infrastructure (Oracle 26ai + SearXNG)
docker compose up -d

# API server
pythia serve

# TUI client
pythia search
```

## Testing

```bash
pytest tests/ -v
```
