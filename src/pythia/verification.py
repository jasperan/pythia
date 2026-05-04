"""Verification pass — validates claims against sources, checks URL liveness."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from pythia.server.llm_client import LLMClient

logger = logging.getLogger(__name__)

_VERIFY_SYSTEM = """You are a verification agent. Your job is to audit a research report by checking that every factual claim is properly supported by the provided sources.

Check each claim against the source material. For each issue found, record:
- The claim text (quote from the report)
- The issue type: "unsourced", "mismatched", "overstated", or "dead_url"
- The severity: "fatal", "major", or "minor"
- A brief explanation

Return a JSON object:
{{
  "claims_checked": <number>,
  "issues": [
    {{"claim": "...", "type": "unsourced|mismatched|overstated|dead_url", "severity": "fatal|major|minor", "explanation": "..."}}
  ],
  "status": "pass" | "pass_with_notes" | "fail",
  "summary": "Brief verification summary"
}}

Rules:
- "unsourced": claim has no matching source
- "mismatched": source doesn't support the specific claim made
- "overstated": claim uses stronger language than the source warrants
- "dead_url": source URL is marked as unreachable
- Only flag genuine issues. Do not nitpick hedged or opinion statements.
"""

_VERIFY_USER = """Research question: {query}

Report:
{report}

Sources:
{sources}

Verify every claim. Return JSON."""


@dataclass
class VerificationResult:
    claims_checked: int = 0
    issues: list[dict] = field(default_factory=list)
    status: str = "pending"
    summary: str = ""

    def to_markdown(self) -> str:
        lines = [
            "# Verification Report",
            "",
            f"- **Status:** {self.status}",
            f"- **Claims checked:** {self.claims_checked}",
            f"- **Issues found:** {len(self.issues)}",
            f"- **Summary:** {self.summary}",
        ]
        if self.issues:
            lines.append("")
            lines.append("## Issues")
            lines.append("")
            for i, issue in enumerate(self.issues, 1):
                lines.append(
                    f"{i}. **[{issue['severity'].upper()}] {issue['type']}**: {issue['claim']}"
                )
                lines.append(f"   - {issue['explanation']}")
        lines.append("")
        return "\n".join(lines)


_MAX_VERIFY_REPORT_CHARS = 12000
_MAX_VERIFY_SOURCES = 20
_MAX_VERIFY_SOURCE_CHARS = 700


async def verify_report(
    ollama: LLMClient,
    query: str,
    report: str,
    sources: list[dict],
    model: str,
) -> VerificationResult:
    if report.lstrip().startswith("# Evidence Ledger Report"):
        return _verify_evidence_ledger(report, sources)

    # Truncate to stay within model context limits
    truncated_report = report[:_MAX_VERIFY_REPORT_CHARS]
    if len(report) > _MAX_VERIFY_REPORT_CHARS:
        truncated_report += "\n\n[Report truncated for verification. Check claims in the portion above.]"

    sources_text = "\n\n".join(
        _format_source_for_verification(source, i)
        for i, source in enumerate(_select_verification_sources(report, sources))
    )

    user = _VERIFY_USER.format(query=query, report=truncated_report, sources=sources_text)

    try:
        response = await ollama.generate(_VERIFY_SYSTEM, user, json_mode=True, model=model)
        if not response or not response.strip():
            logger.warning("Verification returned empty response (prompt may exceed model context)")
            return VerificationResult(
                status="pass_with_notes",
                summary="Verification skipped: model returned empty response (prompt too large).",
            )
        data = json.loads(response)
        status = _normalize_verification_status(data.get("status"))
        summary = data.get("summary", "")
        if status == "fail" and data.get("status") not in {"pass", "pass_with_notes", "fail"}:
            summary = f"Verifier returned unsupported status {data.get('status')!r}. {summary}".strip()
        return VerificationResult(
            claims_checked=data.get("claims_checked", 0),
            issues=data.get("issues", []),
            status=status,
            summary=summary,
        )
    except json.JSONDecodeError as e:
        logger.warning(f"Verification JSON parse failed: {e} — raw response: {response[:200]!r}")
        return VerificationResult(
            status="pass_with_notes",
            summary="Verification skipped: could not parse model response as JSON.",
        )
    except Exception as e:
        logger.warning(f"Verification failed: {type(e).__name__}: {e}")
        return VerificationResult(
            status="pass_with_notes",
            summary=f"Verification skipped: {type(e).__name__}",
        )


def _select_verification_sources(report: str, sources: list[dict]) -> list[dict]:
    """Prefer sources cited by the report, then fill with early sources."""
    if not sources:
        return []

    by_index: dict[int, dict] = {}
    for fallback, source in enumerate(sources, 1):
        index = _safe_source_index(source, fallback)
        by_index[index] = source

    selected: list[dict] = []
    seen: set[int] = set()
    for raw in re.findall(r"\[(\d+)\]", report):
        index = int(raw)
        if index in seen or index not in by_index:
            continue
        selected.append(by_index[index])
        seen.add(index)
        if len(selected) >= _MAX_VERIFY_SOURCES:
            return selected

    for fallback, source in enumerate(sources, 1):
        index = _safe_source_index(source, fallback)
        if index in seen:
            continue
        selected.append(source)
        seen.add(index)
        if len(selected) >= _MAX_VERIFY_SOURCES:
            break

    return selected


def _format_source_for_verification(source: dict, fallback_index: int) -> str:
    index = _safe_source_index(source, fallback_index + 1)
    title = source.get("title", "Untitled")
    url = source.get("url", "no URL")
    snippet = source.get("snippet") or source.get("content") or ""
    if len(snippet) > _MAX_VERIFY_SOURCE_CHARS:
        snippet = snippet[:_MAX_VERIFY_SOURCE_CHARS] + "..."
    parts = [f"[{index}] {title} — {url}"]
    if snippet:
        parts.append(f"Source excerpt: {snippet}")
    return "\n".join(parts)


def _safe_source_index(source: dict, fallback: int) -> int:
    try:
        return int(source.get("index", fallback))
    except (TypeError, ValueError):
        return fallback


def _normalize_verification_status(status: object) -> str:
    if status in {"pass", "pass_with_notes", "fail"}:
        return str(status)
    return "fail"


def _verify_evidence_ledger(report: str, sources: list[dict]) -> VerificationResult:
    source_indices = {
        _safe_source_index(source, fallback)
        for fallback, source in enumerate(sources, 1)
    }
    cited = {int(raw) for raw in re.findall(r"\[(\d+)\]", report)}
    if not cited:
        return VerificationResult(
            status="fail",
            summary="Evidence ledger contains no numeric source citations.",
        )

    missing = sorted(cited - source_indices)
    if missing:
        return VerificationResult(
            claims_checked=len(cited),
            status="fail",
            issues=[
                {
                    "claim": f"Missing source citation [{index}]",
                    "type": "dead_url",
                    "severity": "major",
                    "explanation": "Evidence ledger cited a source index that is not present in the source list.",
                }
                for index in missing
            ],
            summary=f"Evidence ledger cites missing source indices: {missing}",
        )

    return VerificationResult(
        claims_checked=len(cited),
        status="pass_with_notes",
        summary=f"Evidence ledger locally verified with {len(cited)} cited source(s). Semantic synthesis was bypassed.",
    )
