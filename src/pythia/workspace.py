"""Slug generation and workspace changelog utilities."""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


def generate_slug(text: str, max_words: int = 5) -> str:
    """Generate a URL-safe slug from arbitrary text.

    Rules (matching Feynman conventions):
    - Lowercase, hyphens as separators
    - No filler words
    - Max `max_words` words
    - Only alphanumeric and hyphens
    - Max 60 characters total

    Examples:
        >>> generate_slug("What are the tradeoffs between RISC-V and ARM for edge AI?")
        'tradeoffs-risc-v-arm-edge'
        >>> generate_slug("Cloud sandbox pricing comparison 2024")
        'cloud-sandbox-pricing-comparison-2024'
    """
    text = unicodedata.normalize("NFKD", text).lower()

    fillers = {"the", "a", "an", "is", "of", "for", "to", "in", "on", "at", "and", "or", "but", "with", "between", "what", "are", "how", "why", "do", "does", "did", "can", "could", "would", "should", "will", "shall", "from", "by", "about", "into", "through", "during", "before", "after", "above", "below", "up", "down", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "which", "who", "whom", "this", "that", "these", "those", "its", "it"}

    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
    meaningful = [w for w in words if w not in fillers and len(w) > 1]

    slug_words = meaningful[:max_words]

    slug = "-".join(slug_words)

    if len(slug) > 60:
        slug = slug[:60].rsplit("-", 1)[0]

    return slug or "research"


class WorkspaceChangelog:
    """Manages CHANGELOG.md as a chronological lab notebook for research sessions.

    Inspired by Feynman's workspace changelog convention — tracks what changed,
    what failed, what was verified, and next steps across research sessions.
    """

    def __init__(self, workspace_dir: Path | str | None = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.changelog_path = self.workspace_dir / "CHANGELOG.md"

    def _ensure_header(self) -> str:
        if self.changelog_path.exists():
            return self.changelog_path.read_text()
        return "# Research Workspace Changelog\n\n"

    def append_entry(
        self,
        slug: str,
        action: str,
        details: str = "",
        status: str = "in_progress",
        next_step: str = "",
    ) -> str:
        """Append a changelog entry and return the full updated content.

        Args:
            slug: Research topic slug
            action: What was done (e.g., "Completed round 1", "Gap analysis found 2 missing areas")
            details: Additional context
            status: One of in_progress, verified, blocked, completed, failed
            next_step: Recommended next action
        """
        existing = self._ensure_header()
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        entry = f"## [{timestamp}] `{slug}` — {action}\n"
        if details:
            entry += f"\n{details}\n"
        entry += f"\n**Status:** {status}\n"
        if next_step:
            entry += f"**Next:** {next_step}\n"
        entry += "\n"

        new_content = existing.rstrip() + "\n" + entry
        self.changelog_path.write_text(new_content)
        return new_content

    def read_recent(self, n: int = 5) -> str:
        if not self.changelog_path.exists():
            return ""
        content = self.changelog_path.read_text()
        sections = re.split(r"\n## ", content)
        return "\n## ".join(sections[-n:]) if len(sections) > n else content

    def read_relevant(self, slug: str) -> str:
        if not self.changelog_path.exists():
            return ""
        content = self.changelog_path.read_text()
        sections = re.split(r"\n## ", content)
        relevant = [s for s in sections if slug in s.lower() or slug.replace("-", " ") in s.lower()]
        return "\n## ".join(relevant) if relevant else ""
