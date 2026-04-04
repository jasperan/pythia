# Autopilot Spec: Full README Walkthrough Validation

## Task Statement
Validate Pythia end to end from a user perspective by following the README and shipped install flow, running the documented setup and usage paths, identifying defects or documentation gaps, fixing code issues that block the journey, and verifying the resulting behavior with direct execution.

## Desired Outcomes
- A fresh user can install the project in a local Python environment.
- The documented infrastructure and service startup path is executable or fails with a clearly attributable environment prerequisite.
- The documented CLI surfaces (`serve`, `search`, `query`, `research`, `embed`) behave as described or are corrected.
- API server startup and at least representative endpoints are exercised.
- TUI startup is validated to the extent possible in a non-interactive terminal session.
- Any code defects uncovered during the walkthrough are fixed and verified.
- README mismatches discovered during execution are captured and, if appropriate, corrected.

## In-Scope Surfaces
- `install.sh` one-command installer behavior as far as the local environment permits
- Manual install path from `README.md`
- Editable package install from `pyproject.toml`
- Config loading from `pythia.yaml`
- Docker compose startup path for Oracle + SearXNG
- Oracle schema setup path and connection assumptions
- Ollama connectivity and model assumptions
- CLI commands:
  - `pythia serve`
  - `pythia search`
  - `pythia query`
  - `pythia research`
  - `pythia embed`
- FastAPI app startup plus representative endpoints (`/health`, `/search`, `/research`, `/embed`, `/history`, `/stats`, `/cache`)
- Textual app import/startup path and any non-interactive smoke coverage

## Out of Scope / Environment-Dependent Items
- Long-running "24/7" soak operation beyond the current execution window
- External network availability, model downloads, and third-party service uptime
- Manual SQL interaction that depends on unavailable Oracle tooling if the toolchain is absent from the environment
- Fully interactive human-driven TUI exploration beyond launch/smoke behavior in this terminal session

## Acceptance Criteria
- Installation path completes or reveals a concrete actionable defect with reproducible evidence.
- At least one real execution attempt is made for each documented major surface.
- Failures are classified as one of:
  - code defect
  - documentation defect
  - missing local prerequisite
  - sandbox or network restriction
- Any code defect fixed in this iteration is followed by direct re-execution of the previously failing path and targeted regression checks.
- Commits are created only for iterations that include verified code changes.

## Risks and Assumptions
- Oracle container startup and DB tooling may be heavy and time-sensitive.
- Ollama may be installed but not running, or the required model may be absent.
- Some flows may require escalated permissions for Docker daemon access or network downloads.
- TUI verification may need to rely on startup/smoke coverage rather than complete manual interaction.
- Existing untracked files in the worktree are unrelated and must remain untouched.

## Likely Touchpoints
- `README.md`
- `install.sh`
- `pyproject.toml`
- `pythia.yaml`
- `docker-compose.yml`
- `setup_schema.sql`
- `src/pythia/cli.py`
- `src/pythia/cli_runner.py`
- `src/pythia/config.py`
- `src/pythia/services.py`
- `src/pythia/server/`
- `src/pythia/tui/`
- `tests/`
