"""Grounding badge — shows answer verification status."""
from __future__ import annotations

from textual.widget import Widget
from textual.reactive import reactive


class GroundingBadge(Widget):
    """Shows how well the answer is grounded in its cited sources."""

    DEFAULT_CSS = """
    GroundingBadge {
        height: 1;
        margin: 0 1;
        padding: 0 1;
    }
    """

    label = reactive("")

    def render(self) -> str:
        return self.label

    def show_grounding(self, score: float, grounded: int, total: int, label: str) -> None:
        if total == 0:
            self.label = ""
            return
        pct = int(score * 100)
        if label == "well-grounded":
            icon = "[green]\u2713[/green]"
        elif label == "partially-grounded":
            icon = "[yellow]\u25cf[/yellow]"
        else:
            icon = "[red]\u25cb[/red]"
        self.label = f"{icon} Grounding: {pct}% ({grounded}/{total} claims verified) \u2014 {label}"
