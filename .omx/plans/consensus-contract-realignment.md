# Consensus Plan: Contract Realignment for README Walkthrough

Supersedes `.omx/plans/autopilot-impl.md` for this execution branch.

## Scope
- Realign shipped behavior, config loading, TUI request propagation, embedding docs, and shutdown ownership before rerunning the README walkthrough.
- Treat the code path, not aspirational README language, as the default contract source unless this plan explicitly upgrades the code contract.

## RALPLAN-DR

### Principles
- Prefer executable contracts over marketing copy.
- Make outside-root behavior explicit: supported paths must pass, unsupported paths must fail fast with remediation.
- Validate state propagation at the request payload boundary, not only in local widget state.
- Separate resource ownership from resource reachability.

### Decision Drivers
- The current walkthrough is blocked by contract drift between README, installer, config loading, and TUI runtime behavior.
- The fastest safe path is to narrow or clarify claims unless there is a small, testable code change that makes the contract real.
- Walkthrough evidence is only credible after the embedding architecture and shutdown semantics are made explicit.

### Options
- Option A: Upgrade code to match current README promises.
  - Pros: stronger UX if completed.
  - Cons: larger scope, drags installer/orchestration work into this fix set, higher regression risk.
- Option B: Make shipped code the source of truth, then selectively harden the smallest missing code paths that block installed usage.
  - Pros: bounded scope, matches observed behavior, gives a testable walkthrough contract.
  - Cons: README becomes less ambitious and some "one-command" marketing must be reduced.

### Decision
Choose Option B. Harden only the missing behavior that materially affects installed use outside repo root and TUI correctness; otherwise align docs to the shipped runtime.

## ADR

### Decision
- Installer contract: `install.sh` remains a dependency/bootstrap installer, not a full configure-and-run workflow.
- Installed-command contract: outside repo root is supported only when config is supplied via `--config` or `PYTHIA_CONFIG`; bare implicit config lookup outside the project root must fail clearly.
- Embedding architecture: current runtime source of truth is Python `sentence-transformers` embeddings stored via Oracle `TO_VECTOR`; Oracle ONNX loading is optional/non-required documentation, not a required runtime dependency.
- TUI ownership: the TUI may stop only resources it created in the current session; it must never unconditionally stop shared Docker services.

### Why Chosen
- `install.sh` only clones and installs dependencies today ([install.sh](/home/ubuntu/git/personal/pythia/install.sh#L44), [install.sh](/home/ubuntu/git/personal/pythia/install.sh#L91)).
- README still advertises a broader one-command flow and manual setup wording that exceeds that script ([README.md](/home/ubuntu/git/personal/pythia/README.md#L158), [README.md](/home/ubuntu/git/personal/pythia/README.md#L181)).
- `load_config()` ignores `PYTHIA_CONFIG`, while `ServiceManager` sets it only for the spawned API subprocess ([src/pythia/config.py](/home/ubuntu/git/personal/pythia/src/pythia/config.py#L71), [src/pythia/services.py](/home/ubuntu/git/personal/pythia/src/pythia/services.py#L234)).
- TUI search and research requests currently omit `model` in outbound JSON even after local model changes ([src/pythia/tui/screens/search.py](/home/ubuntu/git/personal/pythia/src/pythia/tui/screens/search.py#L164), [src/pythia/tui/screens/research.py](/home/ubuntu/git/personal/pythia/src/pythia/tui/screens/research.py#L90)).
- Runtime embeddings are Python-generated today, despite Oracle-ONNX-first wording in docs ([src/pythia/server/oracle_cache.py](/home/ubuntu/git/personal/pythia/src/pythia/server/oracle_cache.py#L31), [src/pythia/embeddings.py](/home/ubuntu/git/personal/pythia/src/pythia/embeddings.py#L1), [README.md](/home/ubuntu/git/personal/pythia/README.md#L227)).
- TUI shutdown currently tears down services through `stop_all()` whenever a manager exists, and `stop_all()` always stops Docker services ([src/pythia/tui/app.py](/home/ubuntu/git/personal/pythia/src/pythia/tui/app.py#L186), [src/pythia/services.py](/home/ubuntu/git/personal/pythia/src/pythia/services.py#L96)).

### Consequences
- README/install wording becomes narrower but truthful.
- Installed usage becomes predictable across working directories.
- Walkthrough B will validate the actual runtime architecture rather than a stale ONNX-based story.

### Follow-Ups
- If a later release wants a true one-command experience, treat that as new feature work with installer/service orchestration expansion.
- If Oracle-native embedding becomes the target architecture later, ship code changes first and only then promote ONNX loading back into the primary setup path.

## Contract Matrix

| Contract family | Evidence | Source of truth choice | Fix direction | Acceptance rule | Likely files |
|---|---|---|---|---|---|
| Install scope | README promises "clone, configure, and run in a single step", but `install.sh` only clones, creates `.venv`, and installs deps | `install.sh` behavior is authoritative for this release | both | README quick start and installer completion text describe bootstrap-only scope and point to manual infra/schema/runtime steps; no text implies services or schema are auto-configured | `README.md`, `install.sh`, `docs/slides/presentation.html` |
| Manual setup credentials | README now uses `admin/Welcome12345*` for `docker exec`; script still only installs deps | Manual setup docs own DB bootstrap contract; installer remains out of scope | docs | Manual setup section is internally consistent, and installer output does not imply DB user/schema setup occurred | `README.md`, `install.sh`, `docker-compose.yml` |
| Installed command config resolution | CLI defaults use `pythia.yaml` from `cwd`; `PYTHIA_CONFIG` is ignored by outer CLI/TUI load path | Explicit config contract: `--config` first, `PYTHIA_CONFIG` second, implicit repo-root file last | code | Outside repo root, commands honor `PYTHIA_CONFIG`/`--config`; missing implicit config fails fast with actionable message instead of silently using defaults | `src/pythia/config.py`, `src/pythia/cli.py`, `src/pythia/tui/app.py`, `src/pythia/services.py`, `README.md`, `tests/test_config.py`, `tests/test_services.py` |
| TUI model propagation | Search/research request bodies omit `model` after local model changes | API request payload is authoritative | code | Any local model change is present in outbound `/search` and `/research` JSON bodies without mutating shared client defaults | `src/pythia/tui/screens/search.py`, `src/pythia/tui/screens/research.py`, `src/pythia/tui/screens/dashboard.py`, `src/pythia/tui/widgets/settings_panel.py`, `tests/test_tui_integration.py` |
| Embedding architecture | Runtime uses Python `sentence-transformers` + `TO_VECTOR`; docs present Oracle ONNX loading as required-looking | Current runtime code path is authoritative | both | Docs clearly state Python-generated embeddings are the active path; ONNX loading is optional or removed from primary walkthrough; code comments/docstrings match | `src/pythia/embeddings.py`, `src/pythia/server/oracle_cache.py`, `src/pythia/server/ollama.py`, `README.md`, `setup_schema.sql`, `docs/slides/presentation.html`, `tests/test_embeddings.py`, `tests/test_cli_runner.py` |
| TUI shutdown ownership | App unmount calls shutdown; manager stop path always stops Docker services | Ownership tracking in code is authoritative | code | TUI stops API/docker only when this session started them; attaching to existing services never triggers compose shutdown | `src/pythia/tui/app.py`, `src/pythia/services.py`, `tests/test_services.py`, `tests/test_tui_integration.py` |

## Explicit Pass/Fail Rules: Installed Commands Outside Repo Root

### Must Pass
1. `cd /tmp && PYTHIA_CONFIG=/home/ubuntu/git/personal/pythia/pythia.yaml pythia query "What is RLHF?"`
2. `cd /tmp && PYTHIA_CONFIG=/home/ubuntu/git/personal/pythia/pythia.yaml pythia research "RISC-V vs ARM" --max-rounds 1`
3. `cd /tmp && pythia serve --config /home/ubuntu/git/personal/pythia/pythia.yaml --host 127.0.0.1 --port 8900`
4. `cd /tmp && pythia search --config /home/ubuntu/git/personal/pythia/pythia.yaml --no-auto-start`

### Must Fail
1. `cd /tmp && pythia query "What is RLHF?"`
   - Expected: non-zero exit with explicit missing-config guidance referencing `--config` and `PYTHIA_CONFIG`.
2. `cd /tmp && pythia search`
   - Expected: non-zero exit with explicit message that auto-start requires a reachable config plus project Docker assets, and suggests `--config`, `PYTHIA_CONFIG`, `--no-auto-start`, or launching from the project root.

## Required Payload-Level Validation

### TUI model propagation
1. Search command path:
   - Change model via `/model llama3.3:70b`.
   - Submit a search.
   - Assert outbound `/search` JSON includes `"model": "llama3.3:70b"`.
2. Dashboard/settings path:
   - Change model through the picker.
   - Submit one search and one research request.
   - Assert both outbound payloads carry the updated `model`.
3. Multi-turn preservation:
   - After a model change, submit a follow-up search.
   - Assert payload includes both the updated `model` and existing `conversation_history`.

### Shutdown ownership
1. If the TUI did not start Docker services in this session, unmount/Ctrl+C must not invoke compose shutdown.
2. If the TUI started only the API subprocess, shutdown must terminate only that subprocess.
3. If the TUI started Docker services in this session, shutdown may stop only those owned services once.

## Embedding Decision Gate Before Walkthrough B

Walkthrough B is blocked until this decision is implemented and documented:
- Primary architecture: Python `sentence-transformers` generates embeddings; Oracle stores/query-compares vectors via `TO_VECTOR`.
- Documentation action: remove or demote Oracle ONNX loading from the primary setup path.
- Walkthrough B must use the updated docs and must not require ONNX loading to declare success.

## Execution Sequence

1. Repair contract ownership and config resolution.
   - Implement `PYTHIA_CONFIG` support in the outer config loader and make CLI/TUI missing-config failures explicit outside project-root defaults.
   - Add focused tests for config precedence and outside-root failure messaging.
2. Repair TUI request propagation and shutdown ownership.
   - Pass the selected model in `/search` and `/research` payloads.
   - Track service ownership separately from service connectivity and gate shutdown accordingly.
3. Align installer/manual/embedding documentation.
   - Reduce one-command installer claims to bootstrap-only.
   - Keep manual DB bootstrap in README, including the documented `admin/Welcome12345*` entrypoint.
   - Rewrite embedding/setup docs to match the Python-embedding runtime.
4. Run Walkthrough B against the corrected contract.
   - Re-run install/bootstrap, outside-root command checks, TUI smoke, and README setup steps in the new documented order.

## Acceptance-to-Evidence Matrix

| Acceptance criterion | Exact evidence required |
|---|---|
| Installer scope is truthful | `rg -n "One-command install|configure, and run|Installation complete|Oracle Database connection required" README.md install.sh docs/slides/presentation.html` |
| `PYTHIA_CONFIG` is honored by outer CLI load | `pytest tests/test_config.py -q` plus a targeted test covering env precedence and missing-config messaging |
| Outside-root installed-command contract is enforced | `cd /tmp && PYTHIA_CONFIG=/home/ubuntu/git/personal/pythia/pythia.yaml pythia query "What is RLHF?"`; `cd /tmp && pythia query "What is RLHF?"`; capture exit code and stderr for both |
| TUI search/research payloads propagate the selected model | `pytest tests/test_tui_integration.py -q -k "model or research"` with mocked `httpx.AsyncClient` assertions on `/search` and `/research` request JSON |
| Shared model state is not mutated by overrides | `pytest tests/test_http_clients.py tests/test_battle_hardening.py tests/test_research.py -q` |
| Shutdown only affects owned resources | `pytest tests/test_services.py tests/test_tui_integration.py -q -k "shutdown or stop"` with mocked subprocess assertions proving `docker compose` stop/down is skipped when services were pre-existing or not owned |
| Embedding docs match runtime | `pytest tests/test_embeddings.py tests/test_cli_runner.py -q` plus `rg -n "ONNX|VECTOR_EMBEDDING|sentence-transformers|TO_VECTOR" README.md setup_schema.sql src/pythia/server/oracle_cache.py src/pythia/server/ollama.py docs/slides/presentation.html` |
| Walkthrough B matches corrected contract | `python3 -m venv .venv-readme-walkthrough && .venv-readme-walkthrough/bin/pip install -e '.[dev]'`; `cd /tmp && PYTHIA_CONFIG=/home/ubuntu/git/personal/pythia/pythia.yaml /home/ubuntu/git/personal/pythia/.venv-readme-walkthrough/bin/pythia search --no-auto-start`; capture command output plus any README step logs used during rerun |

## Handoff Guidance

- Available agent types for this follow-up: `planner`, `architect`, `critic`, `executor`, `test-engineer`, `verifier`, `writer`, `code-reviewer`.
- Recommended path: `ralph` for the code-first repair sequence, then a single verification pass.
- `ralph` launch hint: `$ralph "Execute /home/ubuntu/git/personal/pythia/.omx/plans/consensus-contract-realignment.md in order, stopping only after the acceptance-to-evidence matrix is satisfied."`
- `team` split if parallelism is needed:
  - Lane 1, high reasoning: config/outside-root contract plus tests.
  - Lane 2, high reasoning: TUI payload propagation and shutdown ownership plus tests.
  - Lane 3, medium reasoning: README/install/embedding docs sync after Lane 1 and Lane 2 contracts settle.
- `team` launch hint: `$team "Lane 1: config/outside-root contract and tests" "Lane 2: TUI model propagation and shutdown ownership" "Lane 3: README/install/embedding docs alignment after contract fixes"`
- Team verification path: run the acceptance-to-evidence matrix in order; docs lane does not close until Walkthrough B is rerun on the updated contract.
