"""Provenance tracking for research sessions — source accounting and verification status."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ProvenanceRecord:
    """Tracks source accounting for a single research session."""

    topic: str
    slug: str
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    rounds: int = 0
    sources_consulted: int = 0
    sources_accepted: int = 0
    sources_rejected: int = 0
    verification_status: str = "pending"
    verification_summary: str = ""
    plan_path: str = ""
    research_files: list[str] = field(default_factory=list)
    model_used: str = ""
    elapsed_ms: int = 0

    def to_markdown(self) -> str:
        lines = [
            f"# Provenance: {self.topic}",
            "",
            f"- **Date:** {self.date}",
            f"- **Slug:** {self.slug}",
            f"- **Rounds:** {self.rounds}",
            f"- **Model:** {self.model_used}",
            f"- **Elapsed:** {self.elapsed_ms / 1000:.1f}s",
            f"- **Sources consulted:** {self.sources_consulted}",
            f"- **Sources accepted:** {self.sources_accepted}",
            f"- **Sources rejected:** {self.sources_rejected}",
            f"- **Verification:** {self.verification_status}",
        ]
        if self.verification_summary:
            lines.append(f"- **Verification notes:** {self.verification_summary}")
        if self.plan_path:
            lines.append(f"- **Plan:** {self.plan_path}")
        if self.research_files:
            lines.append(f"- **Research files:** {', '.join(self.research_files)}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "slug": self.slug,
            "date": self.date,
            "rounds": self.rounds,
            "sources_consulted": self.sources_consulted,
            "sources_accepted": self.sources_accepted,
            "sources_rejected": self.sources_rejected,
            "verification_status": self.verification_status,
            "verification_summary": self.verification_summary,
            "plan_path": self.plan_path,
            "research_files": self.research_files,
            "model_used": self.model_used,
            "elapsed_ms": self.elapsed_ms,
        }
