# pythia-tui — Go terminal UI

An **additional** way to run pythia: a Go front-end in the Charm v2 stack
(`charm.land/bubbletea/v2`, `lipgloss/v2`, `bubbles/v2`, `huh/v2`), alongside the
Python CLI and the Textual TUI.

It never reimplements retrieval or research. Every answer, search result and
research step comes from the same service the Python CLI and the Textual TUI
use, so a Go user and a Python user get identical results.

## Build and run

```bash
cd gotui
go build ./cmd/pythia-tui

./pythia-tui                          # full-screen: search, research tree, history, dashboard
./pythia-tui --start-service          # launch pythia's own service first, then attach
./pythia-tui --base-url http://127.0.0.1:8900
```

With no terminal on stdin it never prompts, and `ACCESSIBLE=1` routes to a plain
renderer for screen readers.

## Scriptable, non-interactive

| Flag | Purpose |
|---|---|
| `--query "..."` | run a search / research query |
| `--json` | emit machine-readable output |
| `--health` | report whether the service is reachable |
| `--stats` | service statistics |
| `--history` | stored research history |
| `--skills` | available skills |
| `--deep` | use the deep-research mode |
| `--max-rounds N` | bound deep research rounds |
| `--limit N` | bound result counts |
| `--model M` | override the model |
| `--clear-cache` | clear the retrieval cache |

## Configuration

| Flag | Purpose |
|---|---|
| `--host`, `--port` | service bind address for `--start-service` |
| `--base-url` | attach to an already-running service |
| `--config` | config file path |
| `--project-root` | checkout used when launching the service |
| `--setup` | run the connection prompt form |
| `--no-input` | never prompt (fails instead) |
| `--yes` | assume yes for confirmations |

Environment: `PYTHIA_CONFIG`, `PYTHIA_TUI_PYTHON` (interpreter used to start the
service).

## Design notes

- **Palette.** Colours come from the workspace design tokens
  (`docs/tui-design-tokens.md`); `internal/huhstyle` is a byte-identical copy of
  the canonical `docs/huh/huhstyle.go`. `scripts/tui-shot/check_palette.py`
  passes on this module.
- **Escape.** No `huh` form is *embedded* in the Bubble Tea model, so the classic
dead end cannot occur here: huh's default keymap binds `Quit` to `ctrl+c` only,
so a form that receives a delegated Escape silently does nothing and then
swallows every later keystroke. The one `huh` form in this module
(`ConnectionForm`) is run *standalone* via `RunConnectionPrompts` for the
`--setup` / accessible path, where huh's own `Run` drives it: Escape is inert
there, nothing advertises it, and `ctrl+c` always works.
  `TestTheTUIOwnsEscape` is a reflection tripwire that fails if a form is ever
  embedded in the model without intercepting Escape first.
- **No timer.** The UI schedules no refresh tick, so nothing can stall a test
  drain.
