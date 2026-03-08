# Pythia

**Self-hosted AI search engine** — a local Perplexity replacement combining SearXNG (free, unlimited web search), Ollama (local LLM inference), and Oracle AI Vector Search with ONNX in-database embeddings (semantic search cache).

Named after the priestess at the Oracle of Delphi who answered questions — a double meaning with Oracle Database as the backend.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Pythia TUI (Textual)                           │
│  Search results + AI answer + citations         │
│  Status bar: model | Oracle | SearXNG           │
│  Search input                                   │
└──────────────┬──────────────────────────────────┘
               │ HTTP (localhost:8900)
┌──────────────▼──────────────────────────────────┐
│  Pythia API Server (FastAPI)                    │
│  POST /search → SSE stream                      │
│  GET  /history, /health, /stats                 │
│  DELETE /cache                                  │
└───┬──────────┬──────────────┬───────────────────┘
    │          │              │
    ▼          ▼              ▼
 SearXNG    Ollama      Oracle DB 26ai
 :8888      :11434     :1521/FREEPDB1
 (Docker)   (local)    (ONNX embeddings + Vector Search)
```

### Search Flow

1. **Check cache** — Oracle generates an embedding for your query in-database using the ONNX model (`VECTOR_EMBEDDING()`), then performs cosine similarity search against cached queries
2. **Cache HIT** (similarity >= 0.85) — Return cached answer instantly with `[from cache]` badge
3. **Cache MISS** — Query SearXNG for top 8 web results, synthesize answer via Ollama with cited sources `[1]`, `[2]`, store result + embedding in Oracle for future hits
4. **Stream response** — Tokens appear in real-time via SSE

### Key Design: ONNX In-Database Embeddings

Pythia uses Oracle's `VECTOR_EMBEDDING()` SQL function to generate embeddings **inside the database** using an ONNX model (default: `ALL_MINILM_L6_V2`). No external embedding service needed — the database handles both embedding generation and vector similarity search.

## Prerequisites

- **Python 3.12+**
- **Docker** (for Oracle 26ai Free and SearXNG containers)
- **Ollama** installed and running locally (for LLM inference only)
- **Conda** (recommended for environment isolation)

## Quick Start

### 1. Create conda environment

```bash
conda create -n pythia python=3.12 -y
conda activate pythia
```

### 2. Install Pythia

```bash
cd ~/git/pythia
pip install -e ".[dev]"
```

### 3. Start infrastructure

```bash
# Start Oracle 26ai Free + SearXNG containers
docker compose up -d

# Wait for Oracle to be ready (~2 minutes on first start)
docker compose logs -f oracle  # wait for "DATABASE IS READY TO USE"

# Pull Ollama model for answer synthesis
ollama pull qwen3.5:9b
```

### 4. Set up Oracle schema

Connect to Oracle and create the Pythia user and tables:

```bash
# Connect as ADMIN (password from docker-compose.yml: Pythia2026#)
# For ADB-Free container, use the wallet or direct connection
sql admin/Pythia2026#@localhost:1521/FREEPDB1
```

```sql
-- Create user
CREATE USER pythia IDENTIFIED BY pythia;
GRANT CONNECT, RESOURCE, UNLIMITED TABLESPACE, DB_DEVELOPER_ROLE TO pythia;
GRANT CREATE MINING MODEL TO pythia;
```

Then connect as the `pythia` user and run the schema:

```bash
sql pythia/pythia@localhost:1521/FREEPDB1 @setup_schema.sql
```

#### Load ONNX Embedding Model

The ONNX model must be loaded once before Pythia can generate embeddings. For Oracle 26ai ADB-Free, models may be pre-loaded. Otherwise:

```sql
-- Connect as PYTHIA user
-- Download all_MiniLM_L6_v2.onnx to an Oracle directory object first
BEGIN
  DBMS_VECTOR.LOAD_ONNX_MODEL(
    'DM_DUMP',
    'all_MiniLM_L6_v2.onnx',
    'ALL_MINILM_L6_V2',
    JSON('{"function":"embedding","embeddingOutput":"embedding","input":{"input":["DATA"]}}')
  );
END;
/
```

Verify the model works:

```sql
SELECT VECTOR_EMBEDDING(ALL_MINILM_L6_V2 USING 'hello world' AS data) FROM DUAL;
```

### 5. Run Pythia

```bash
# Terminal 1: Start the API server
pythia serve

# Terminal 2: Launch the TUI
pythia search
```

## Usage

### TUI

The TUI is a full-screen Textual application with an agent-harness dark theme:

- Type your question in the search bar and press Enter
- Watch the answer stream in real-time with source citations
- Cache hits show a green badge with similarity score
- Web searches show a cyan badge with response time

### Slash Commands

| Command | Action |
|---------|--------|
| `/history` | Show recent searches |
| `/stats` | Show cache hit rate, total searches, avg response time |
| `/model <name>` | Switch Ollama model (e.g., `/model llama3.3:70b`) |
| `/cache clear` | Purge Oracle cache |
| `/clear` | Clear screen |
| `/help` | Show available commands |

### API Endpoints

The FastAPI server runs on `http://localhost:8900` by default.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/search` | POST | Search with SSE streaming. Body: `{"query": "...", "model": "optional"}` |
| `/health` | GET | Status of Oracle, SearXNG, Ollama + cache size |
| `/history` | GET | Recent search history (default limit: 20) |
| `/stats` | GET | Cache hit rate, total searches, avg response time |
| `/cache` | DELETE | Clear all cached entries |

#### SSE Stream Format

```
event: status
data: {"message": "Checking cache..."}

event: source
data: {"index": 1, "title": "...", "url": "...", "snippet": "..."}

event: token
data: {"content": "RLHF is a technique"}

event: done
data: {"cache_hit": false, "response_time_ms": 3200, "sources_count": 8}
```

## Configuration

Edit `pythia.yaml` to customize:

```yaml
server:
  host: "0.0.0.0"
  port: 8900

ollama:
  base_url: "http://localhost:11434"
  model: "qwen3.5:9b"          # LLM for answer synthesis

searxng:
  base_url: "http://localhost:8888"
  max_results: 8
  categories:
    - general
    - science
    - it

oracle:
  dsn: "localhost:1521/FREEPDB1"
  user: "pythia"
  password: "pythia"
  cache_similarity_threshold: 0.85   # cosine similarity for cache hit
  embedding_model: "ALL_MINILM_L6_V2"  # ONNX model loaded in Oracle

tui:
  theme: "dark"
```

### Environment-Specific Overrides

CLI flags override config values:

```bash
# Custom config file
pythia serve --config /path/to/custom.yaml

# Override host/port
pythia serve --host 127.0.0.1 --port 9000

# Override LLM model for TUI session
pythia search --model llama3.3:70b
```

## Development

### Run Tests

```bash
conda activate pythia
pytest tests/ -v
```

### Project Structure

```
pythia/
├── pyproject.toml
├── pythia.yaml              # Default config
├── docker-compose.yml       # Oracle 26ai Free + SearXNG
├── setup_schema.sql         # Oracle DDL + ONNX model setup
├── searxng/
│   └── settings.yml         # SearXNG config
├── src/pythia/
│   ├── cli.py               # Typer entry point
│   ├── config.py            # YAML config loader
│   ├── server/
│   │   ├── app.py           # FastAPI app factory
│   │   ├── search.py        # Search orchestrator
│   │   ├── oracle_cache.py  # Oracle ONNX embeddings + vector cache
│   │   ├── searxng.py       # SearXNG client
│   │   └── ollama.py        # Ollama LLM client
│   └── tui/
│       ├── app.py           # PythiaApp(App)
│       ├── screens/
│       │   └── search.py    # Main search screen
│       ├── widgets/         # Logo, ResultCard, SourceList, etc.
│       └── themes/
│           └── dark.tcss    # Agent-harness dark theme
└── tests/
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| TUI | Python 3.12+, Textual, Rich |
| API Server | FastAPI, uvicorn, SSE |
| Search | SearXNG (Docker) |
| LLM | Ollama (qwen3.5:9b default) |
| Embeddings | Oracle ONNX in-database (`VECTOR_EMBEDDING()`) |
| Database | Oracle Database 26ai Free (Docker) |
| Vector Search | Oracle AI Vector Search (COSINE distance) |

## License

MIT
