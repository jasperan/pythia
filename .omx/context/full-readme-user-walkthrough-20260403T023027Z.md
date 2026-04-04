# Task Statement
Run a real end-to-end validation of the Pythia app from a user perspective by following the README and exercising the shipped surfaces instead of limiting work to unit tests.

# Desired Outcome
- Reproduce the setup and run flows a user would actually follow.
- Execute the app components that the README promises: install flow, infrastructure startup, schema setup path, CLI, API server, and TUI-adjacent startup where feasible.
- Fix defects discovered during the walkthrough.
- Verify fixes with direct execution and targeted tests.
- Commit each code-changing iteration without disturbing unrelated repo state.

# Known Facts / Evidence
- Repo is a Python package with `pyproject.toml` and editable install instructions in `README.md`.
- Core surfaces include CLI (`pythia`), FastAPI server, Textual TUI, Oracle cache layer, SearXNG integration, Ollama integration, and deep research flow.
- Infrastructure dependencies in the README are Docker, Oracle 26ai Free, SearXNG, Ollama, and Conda.
- Current git status shows existing untracked files: `.codex`, `.omc/`, `.pre-commit-config.yaml`.

# Constraints
- Must preserve unrelated worktree changes.
- Must prefer real walkthrough execution over mocked-only validation.
- Network access may be restricted; external pulls may require escalation or may be unavailable.
- Cannot literally "run 24/7" inside one turn; need to drive the current iteration to a grounded stopping point.
- Commit after each iteration only when there is a real code change and verification evidence for that iteration.

# Unknowns / Open Questions
- Which README steps already work in this environment versus fail due to code defects or environment prerequisites.
- Whether Docker daemon access, Oracle container boot, Ollama service, and model availability are currently usable.
- Whether the TUI can be exercised non-interactively enough to validate startup paths in this session.
- Whether install and integration flows require additional missing dependencies not declared in the README.

# Likely Codebase Touchpoints
- `README.md`
- `install.sh`
- `pyproject.toml`
- `src/pythia/cli.py`
- `src/pythia/cli_runner.py`
- `src/pythia/server/app.py`
- `src/pythia/server/search.py`
- `src/pythia/server/research.py`
- `src/pythia/services.py`
- `src/pythia/config.py`
- `src/pythia/tui/`
- `tests/`
