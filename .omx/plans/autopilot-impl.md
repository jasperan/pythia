# RALPLAN Implementation Plan: Full README Walkthrough Validation

## Principles
- Prefer real user flows over isolated mocks.
- Distinguish environment failures from code failures with evidence.
- Fix the narrowest root cause that unblocks the documented journey.
- Re-run the failing path after each fix before widening scope.
- Commit only verified code-changing iterations.

## Decision Drivers
- Maximize user-perspective coverage across install, startup, CLI, API, and TUI surfaces.
- Preserve momentum by front-loading prerequisite and environment checks.
- Keep the worktree safe by avoiding unrelated changes and destructive commands.

## Viable Options
- Option A: README-first walkthrough, fixing breakage as encountered.
  - Pros: matches user intent exactly, reveals real integration and docs issues, produces credible validation evidence.
  - Cons: slower, sensitive to environment availability, may require escalations.
- Option B: test-suite-first stabilization, then limited manual smoke.
  - Pros: faster signal on obvious regressions, easier automation.
  - Cons: misses installation and integration gaps, does not satisfy the requested user-perspective walkthrough.

## Decision
Choose Option A. Use the README and installer as the primary contract, while using targeted tests only as regression support after reproducing failures in real flows.

## Alternatives Considered
- Pure test-driven validation was rejected because the user explicitly asked for actual walkthrough coverage beyond pytest and mocks.
- Full long-duration soak testing was rejected for this turn because it exceeds the current execution window; instead, run the current iteration to a grounded stopping point and preserve evidence.

## Consequences
- More setup and orchestration work up front.
- Some steps may be blocked by local prerequisites rather than code; those blocks still need to be documented precisely.
- Fixes may span both code and README if the walkthrough uncovers contract drift.

## Execution Steps
1. Establish baseline environment evidence.
   - Verify Python, package manager, Conda, Docker, Ollama, and current git state.
   - Create a disposable virtual environment for the walkthrough.
2. Run the documented installation flows.
   - Attempt editable install inside the repo.
   - Attempt the installer script in a disposable target if needed.
   - Record any dependency or packaging failures.
3. Exercise the documented runtime prerequisites.
   - Start Docker services.
   - Check Oracle and SearXNG readiness.
   - Check Ollama service/model availability.
   - If a command fails due to sandbox or daemon restrictions, escalate appropriately.
4. Exercise shipped user surfaces in increasing dependency order.
   - `pythia embed` for local embedding path.
   - `pythia serve` plus `/health`.
   - `pythia query` and `pythia research`.
   - `pythia search` startup smoke for the TUI path.
5. On each failure:
   - Reproduce and isolate.
   - Inspect the implicated code or docs.
   - Implement the minimal fix.
   - Re-run the failing command and targeted regression checks.
   - Commit the verified iteration if files changed.
6. Finish with broader QA and review.
   - Run targeted tests and any relevant integration tests.
   - Perform a code-review-style pass on modified areas.
   - Summarize verified coverage, fixes, remaining blockers, and exact commands run.

## Verification Plan
- Installation:
  - `python3 -m venv .venv-readme-walkthrough`
  - `.venv-readme-walkthrough/bin/pip install -e '.[dev]'`
- Infrastructure:
  - `docker compose up -d`
  - `docker compose ps`
  - `docker compose logs oracle --tail 200`
- CLI/API/TUI:
  - `.venv-readme-walkthrough/bin/pythia embed 'hello world'`
  - `.venv-readme-walkthrough/bin/pythia serve --host 127.0.0.1 --port 8900`
  - `curl http://127.0.0.1:8900/health`
  - `.venv-readme-walkthrough/bin/pythia query 'What is RLHF?'`
  - `.venv-readme-walkthrough/bin/pythia research 'RISC-V vs ARM for edge AI' --max-rounds 1`
  - `.venv-readme-walkthrough/bin/pythia search --no-auto-start`

## Pre-Mortem
- Scenario 1: Packaging/install succeeds, but runtime imports fail due to undeclared dependencies or bad entrypoint wiring.
  - Mitigation: run each documented command directly from the installed environment, not only from source.
- Scenario 2: Docker or Ollama availability blocks deep flows, hiding whether the app code is correct.
  - Mitigation: separate environment errors from code errors and still validate the locally executable parts.
- Scenario 3: Fixes for integration issues regress tests or non-targeted commands.
  - Mitigation: after each fix, re-run the failing command plus targeted tests for adjacent modules.

## Concrete Acceptance Criteria
- A reproducible execution log exists for installation, service startup, and each major CLI surface.
- Any defect fixed during the walkthrough is verified by rerunning the failing path.
- No unrelated tracked or untracked files are reverted.
- Each commit corresponds to a verified code-changing iteration.

## Review Notes
- Architectural tension: README-complete validation wants maximum realism, but the environment may not permit every external dependency. The plan resolves this by preserving the README-first order while explicitly classifying environment blocks instead of collapsing into mock-only tests.
- Critical quality check: do not declare success for a surface unless it was actually executed in this environment or explicitly blocked by a documented prerequisite.
