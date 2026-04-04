"""Verification pass — validates claims against sources, checks URL liveness."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pythia.server.ollama import OllamaClient

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

    @property
    def has_fatal(self) -> bool:
        return any(i.get("severity") == "fatal" for i in self.issues)

    @property
    def has_major(self) -> bool:
        return any(i.get("severity") == "major" for i in self.issues)

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


async def verify_report(
    ollama: OllamaClient,
    query: str,
    report: str,
    sources: list[dict],
    model: str,
) -> VerificationResult:
    sources_text = "\n".join(
        f"[{s.get('index', i+1)}] {s.get('title', 'Untitled')} — {s.get('url', 'no URL')}"
        for i, s in enumerate(sources)
    )

    user = _VERIFY_USER.format(query=query, report=report, sources=sources_text)

    try:
        response = await ollama.generate(_VERIFY_SYSTEM, user, json_mode=True, model=model)
        data = json.loads(response)
        return VerificationResult(
            claims_checked=data.get("claims_checked", 0),
            issues=data.get("issues", []),
            status=data.get("status", "fail"),
            summary=data.get("summary", ""),
        )
    except (json.JSONDecodeError, KeyError, Exception) as e:
        logger.warning(f"Verification failed: {e}")
        return VerificationResult(
            status="fail",
            summary=f"Verification process encountered an error: {e}",
        )
