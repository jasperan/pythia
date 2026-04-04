"""Skills panel — displays available research skills and active skill indicator."""
from __future__ import annotations

import httpx
from io import StringIO
from rich.console import Console
from rich.table import Table
from textual.reactive import reactive
from textual.widgets import Static


class SkillsPanel(Static):
    DEFAULT_CSS = """
    SkillsPanel {
        width: 1fr;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
        margin: 1 0;
    }
    """

    active_skill: reactive[str | None] = reactive(None)

    def __init__(self) -> None:
        super().__init__()
        self._skills: list[dict] = []

    def compose(self):
        yield Static("[bold cyan]Research Skills[/bold cyan]")

    async def on_mount(self) -> None:
        await self.load_skills()

    async def load_skills(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://127.0.0.1:8900/skills")
                if resp.status_code == 200:
                    self._skills = resp.json()
                    self._render()
        except Exception:
            self._skills = [
                {"name": "deep-research", "description": "Multi-round iterative research", "triggers": ["deepresearch"]},
                {"name": "compare", "description": "Structured comparison matrix", "triggers": ["compare", "vs"]},
                {"name": "lit-review", "description": "Literature review with consensus", "triggers": ["literature review"]},
                {"name": "quick-answer", "description": "Single-shot search", "triggers": ["quick", "brief"]},
            ]
            self._render()

    def _render(self) -> None:
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Skill", style="bold cyan")
        table.add_column("Description", style="dim")
        table.add_column("Triggers", style="yellow")

        for s in self._skills:
            active_marker = " \u25b6" if s["name"] == self.active_skill else ""
            triggers = ", ".join(s.get("triggers", [])[:3])
            table.add_row(
                s["name"] + active_marker,
                s.get("description", ""),
                triggers,
            )

        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(table)
        table_text = console.file.getvalue()

        self.update(self._title_line() + "\n" + table_text)

    def _title_line(self) -> str:
        active = f" [green](active: {self.active_skill})[/green]" if self.active_skill else ""
        return f"[bold cyan]Research Skills[/bold cyan]{active}"

    def watch_active_skill(self) -> None:
        if self._skills:
            self._render()
