"""Autoresearch agent — iterative experiment loop that optimizes a metric through measure-and-improve cycles."""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pythia.server.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AutoresearchEventType(StrEnum):
    STATUS = "status"
    PLAN = "plan"
    BASELINE = "baseline"
    METRIC = "metric"
    REVERT = "revert"
    DONE = "done"


@dataclass
class AutoresearchEvent:
    event_type: AutoresearchEventType
    data: dict


@dataclass
class ExperimentRecord:
    iteration: int = 0
    metric_value: float = 0.0
    metric_name: str = ""
    metric_direction: str = "higher"
    change_description: str = ""
    kept: bool = True
    benchmark_output: str = ""
    elapsed_ms: int = 0
    changed_files: list[str] | None = None


@dataclass
class AppliedChange:
    """A transactional file change that can be reverted after benchmarking."""

    originals: dict[Path, str]
    changed_files: list[str]


_AUTORESEARCH_PLAN_SYSTEM = """You are an optimization planner. Given a target metric and benchmark command, propose a concrete change to improve it.

Analyze the current state, the benchmark output, and propose a specific, testable modification.
Return JSON:
{{
  "change_description": "What to change and why",
  "edits": [
    {{
      "file": "path/to/file",
      "find": "exact text currently in the file",
      "replace": "replacement text"
    }}
  ],
  "confidence": 0.0-1.0
}}

Rules:
- Only edit files listed in Files in scope.
- Use exact text replacements from the provided file excerpts.
- Keep edits minimal and benchmark-focused.
- If no safe exact edit is possible, return an empty edits list."""

_AUTORESEARCH_PLAN_USER = """Goal: {target}
Optimization metric: {metric_name} ({metric_direction} is better)
Benchmark command: {benchmark_cmd}
Files in scope: {files_in_scope}

Current best {metric_name}: {best_metric}

Previous iteration benchmark output:
{benchmark_output}

Previous change: {previous_change}

File excerpts:
{file_context}

Propose the next change to improve {metric_name}. Return JSON."""

_METRIC_EXTRACT_SYSTEM = """You are a metric extractor. Given benchmark output, extract the numeric value of the target metric.

Return JSON: {{"value": <number>, "unit": "<string>"}}
If the metric cannot be extracted, return {{"value": null, "unit": ""}}."""

_METRIC_EXTRACT_USER = """Benchmark output:
{output}

Metric to extract: {metric_name}

Return JSON with the numeric value."""


class AutoresearchAgent:
    """Iterative experiment optimization loop."""

    def __init__(
        self,
        ollama: LLMClient,
        workspace_dir: Path | str | None = None,
    ):
        self.ollama = ollama
        self.workspace_dir = (Path(workspace_dir) if workspace_dir else Path.cwd()).resolve()
        self.session_dir = self.workspace_dir / ".autoresearch"
        self.records: list[ExperimentRecord] = []

    async def run(
        self,
        metric_name: str,
        benchmark_cmd: str,
        files_in_scope: list[str],
        metric_direction: str = "higher",
        max_iterations: int = 10,
        model: str | None = None,
        target: str | None = None,
    ) -> AsyncIterator[AutoresearchEvent]:
        model = model or self.ollama.model
        start = time.monotonic()

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.records = []

        yield AutoresearchEvent(AutoresearchEventType.STATUS, {
            "message": f"Starting autoresearch: optimize {metric_name} ({metric_direction} is better)",
        })

        yield AutoresearchEvent(AutoresearchEventType.STATUS, {"message": "Running baseline benchmark..."})
        baseline_output = self._run_benchmark(benchmark_cmd)
        baseline_metric = await self._extract_metric(
            baseline_output, metric_name, model,
        )

        if baseline_metric is None:
            yield AutoresearchEvent(AutoresearchEventType.STATUS, {
                "message": f"Could not extract metric '{metric_name}' from baseline output. Aborting.",
            })
            return

        best_record = ExperimentRecord(
            iteration=0, metric_value=baseline_metric,
            metric_name=metric_name, metric_direction=metric_direction,
            change_description="baseline", kept=True,
            benchmark_output=baseline_output[:2000],
        )
        self.records.append(best_record)

        yield AutoresearchEvent(AutoresearchEventType.BASELINE, {
            "metric_name": metric_name,
            "metric_value": baseline_metric,
            "message": f"Baseline {metric_name}: {baseline_metric}",
        })

        self._save_session()

        for iteration in range(1, max_iterations + 1):
            yield AutoresearchEvent(AutoresearchEventType.STATUS, {
                "message": f"Iteration {iteration}/{max_iterations} — planning improvement...",
            })

            plan = await self._propose_change(
                metric_name, benchmark_cmd, files_in_scope,
                metric_direction, best_record, model, target=target,
            )
            if not plan:
                yield AutoresearchEvent(AutoresearchEventType.STATUS, {
                    "message": "Could not propose a change. Stopping.",
                })
                break

            yield AutoresearchEvent(AutoresearchEventType.PLAN, {
                "iteration": iteration,
                "change_description": plan.get("change_description", ""),
                "file_to_modify": plan.get("file_to_modify", ""),
                "confidence": plan.get("confidence", 0),
            })

            yield AutoresearchEvent(AutoresearchEventType.STATUS, {
                "message": f"Applying change: {plan.get('change_description', '')}",
            })

            change_applied = self._apply_change(plan, files_in_scope)
            if change_applied is None:
                yield AutoresearchEvent(AutoresearchEventType.REVERT, {
                    "message": "Change could not be applied. Skipping iteration.",
                })
                continue

            yield AutoresearchEvent(AutoresearchEventType.STATUS, {
                "message": "Running benchmark...",
            })

            iteration_output = self._run_benchmark(benchmark_cmd)
            iteration_metric = await self._extract_metric(
                iteration_output, metric_name, model,
            )

            if iteration_metric is None:
                yield AutoresearchEvent(AutoresearchEventType.STATUS, {
                    "message": "Could not extract metric from iteration output. Reverting.",
                })
                self._revert_change(change_applied)
                continue

            improved = self._is_improved(
                iteration_metric, best_record.metric_value, metric_direction,
            )

            record = ExperimentRecord(
                iteration=iteration, metric_value=iteration_metric,
                metric_name=metric_name, metric_direction=metric_direction,
                change_description=plan.get("change_description", ""),
                kept=improved,
                benchmark_output=iteration_output[:2000],
                changed_files=change_applied.changed_files,
            )
            self.records.append(record)

            if improved:
                best_record = record
                yield AutoresearchEvent(AutoresearchEventType.METRIC, {
                    "iteration": iteration,
                    "metric_name": metric_name,
                    "metric_value": iteration_metric,
                    "improved": True,
                    "message": f"{metric_name}: {best_record.metric_value} -> {iteration_metric} (improved)",
                })
            else:
                yield AutoresearchEvent(AutoresearchEventType.METRIC, {
                    "iteration": iteration,
                    "metric_name": metric_name,
                    "metric_value": iteration_metric,
                    "improved": False,
                    "best_so_far": best_record.metric_value,
                    "message": f"{metric_name}: {iteration_metric} (no improvement, best: {best_record.metric_value})",
                })
                self._revert_change(change_applied)

            self._save_session()

            if iteration >= max_iterations:
                break

        elapsed_ms = int((time.monotonic() - start) * 1000)

        yield AutoresearchEvent(AutoresearchEventType.DONE, {
            "best_metric": best_record.metric_value,
            "best_iteration": best_record.iteration,
            "total_iterations": len(self.records) - 1,
            "improvement": self._improvement_pct(
                best_record.metric_value, self.records[0].metric_value, metric_direction,
            ),
            "elapsed_ms": elapsed_ms,
        })

    def _run_benchmark(self, cmd: str) -> str:
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=300, cwd=self.workspace_dir,
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "Benchmark timed out (300s)"
        except Exception as e:
            return f"Benchmark error: {e}"

    async def _extract_metric(self, output: str, metric_name: str, model: str) -> float | None:
        try:
            response = await self.ollama.generate(
                _METRIC_EXTRACT_SYSTEM,
                _METRIC_EXTRACT_USER.format(output=output[:4000], metric_name=metric_name),
                json_mode=True, model=model,
            )
            data = json.loads(response)
            return data.get("value")
        except Exception as e:
            logger.warning(f"LLM metric extraction failed: {e}")

        try:
            data = json.loads(output.strip())
            return data.get(metric_name)
        except (json.JSONDecodeError, AttributeError):
            pass

        match = re.search(rf'"{metric_name}"\s*:\s*([\d.]+)', output)
        if match:
            return float(match.group(1))

        match = re.search(rf'{metric_name}:\s*([\d.]+)', output)
        if match:
            return float(match.group(1))

        return None

    async def _propose_change(
        self, metric_name: str, benchmark_cmd: str, files_in_scope: list[str],
        metric_direction: str, best_record: ExperimentRecord, model: str,
        target: str | None = None,
    ) -> dict | None:
        response = ""
        file_context = self._collect_file_context(files_in_scope)
        try:
            response = await self.ollama.generate(
                _AUTORESEARCH_PLAN_SYSTEM,
                _AUTORESEARCH_PLAN_USER.format(
                    target=target or metric_name,
                    metric_name=metric_name,
                    benchmark_cmd=benchmark_cmd,
                    files_in_scope=", ".join(files_in_scope),
                    metric_direction=metric_direction,
                    best_metric=best_record.metric_value,
                    benchmark_output=best_record.benchmark_output[:2000],
                    previous_change=best_record.change_description,
                    file_context=file_context,
                ),
                json_mode=True, model=model,
            )
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Change proposal JSON parse error: {e}")
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.warning(f"Change proposal failed: {e}")
        return None

    def _apply_change(self, plan: dict, files_in_scope: list[str]) -> AppliedChange | None:
        edits = self._normalize_edits(plan)
        if not edits:
            return None

        if not files_in_scope:
            logger.warning("Autoresearch refused edits because no files were in scope")
            return None

        allowed = {self._relative_path(p) for p in files_in_scope}
        originals: dict[Path, str] = {}
        updates: dict[Path, str] = {}

        try:
            for edit in edits:
                rel_file = str(edit.get("file", "")).strip()
                find_text = str(edit.get("find", ""))
                replace_text = str(edit.get("replace", ""))
                if not rel_file or not find_text:
                    return None

                rel_path = self._relative_path(rel_file)
                if allowed and rel_path not in allowed:
                    logger.warning("Autoresearch refused out-of-scope edit: %s", rel_file)
                    return None

                path = self._resolve_workspace_path(rel_path)
                if path is None or not path.exists() or not path.is_file():
                    return None

                original = originals.setdefault(path, path.read_text())
                current = updates.get(path, original)
                if find_text not in current:
                    logger.warning("Autoresearch exact edit text not found in %s", rel_path)
                    return None
                updates[path] = current.replace(find_text, replace_text, 1)

            changed_files = []
            for path, updated in updates.items():
                if updated == originals[path]:
                    continue
                path.write_text(updated)
                changed_files.append(str(path.relative_to(self.workspace_dir)))

            if not changed_files:
                return None
            return AppliedChange(originals=originals, changed_files=changed_files)
        except Exception as e:
            logger.warning(f"Failed to apply change: {e}")
            for path, content in originals.items():
                with contextlib.suppress(Exception):
                    path.write_text(content)
            return None

    def _revert_change(self, change: AppliedChange) -> None:
        for target, content in change.originals.items():
            try:
                target.write_text(content)
            except Exception as e:
                logger.warning(f"Failed to revert change in {target}: {e}")

    def _is_improved(self, new: float, old: float, direction: str) -> bool:
        if direction == "higher":
            return new > old
        return new < old

    def _improvement_pct(self, best: float, baseline: float, direction: str) -> float:
        if direction == "higher":
            delta = best - baseline
        else:
            delta = baseline - best
        return round(delta / max(abs(baseline), 1e-9) * 100, 1)

    def _normalize_edits(self, plan: dict) -> list[dict]:
        edits = plan.get("edits")
        if isinstance(edits, list):
            return [e for e in edits if isinstance(e, dict)]

        file_to_modify = plan.get("file_to_modify") or plan.get("file")
        find_text = plan.get("find_text") or plan.get("find")
        replace_text = plan.get("replace_text") or plan.get("replace")
        if file_to_modify and find_text is not None and replace_text is not None:
            return [{"file": file_to_modify, "find": find_text, "replace": replace_text}]
        return []

    def _collect_file_context(self, files_in_scope: list[str]) -> str:
        if not files_in_scope:
            return "(No editable files were provided.)"

        parts = []
        remaining = 12000
        for rel in files_in_scope:
            rel_path = self._relative_path(rel)
            path = self._resolve_workspace_path(rel_path)
            if path is None or not path.exists() or not path.is_file():
                continue
            text = path.read_text(errors="replace")
            excerpt = text[: min(len(text), 3000, remaining)]
            parts.append(f"--- {rel_path} ---\n{excerpt}")
            remaining -= len(excerpt)
            if remaining <= 0:
                break
        return "\n\n".join(parts) if parts else "(No readable scoped files.)"

    def _relative_path(self, path: str | Path) -> str:
        raw = Path(path)
        if raw.is_absolute():
            try:
                raw = raw.resolve().relative_to(self.workspace_dir.resolve())
            except ValueError:
                return ""
        return raw.as_posix()

    def _resolve_workspace_path(self, rel_path: str) -> Path | None:
        if not rel_path:
            return None
        candidate = (self.workspace_dir / rel_path).resolve()
        try:
            candidate.relative_to(self.workspace_dir.resolve())
        except ValueError:
            return None
        return candidate

    def _save_session(self) -> None:
        session_file = self.session_dir / "session.jsonl"
        with open(session_file, "w") as f:
            for record in self.records:
                f.write(json.dumps({
                    "iteration": record.iteration,
                    "metric_value": record.metric_value,
                    "metric_name": record.metric_name,
                    "metric_direction": record.metric_direction,
                    "change_description": record.change_description,
                    "kept": record.kept,
                    "elapsed_ms": record.elapsed_ms,
                    "changed_files": record.changed_files or [],
                }) + "\n")
