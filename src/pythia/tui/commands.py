"""Command palette provider for Pythia TUI."""
from __future__ import annotations

from textual.command import Provider, Hit, Hits


class PythiaCommands(Provider):
    """Provides all Pythia commands to the command palette."""

    async def search(self, query: str) -> Hits:
        app = self.app

        commands = [
            ("Switch to Search", "Navigate to Search screen", "switch_to_search"),
            ("Switch to Research", "Navigate to Research screen", "switch_to_research"),
            ("Switch to History", "Navigate to History screen", "switch_to_history"),
            ("Switch to Dashboard", "Navigate to Dashboard screen", "switch_to_dashboard"),
            ("Clear Results", "Clear current screen results", "clear_results"),
            ("Export Results", "Export last result as markdown", "export_results"),
            ("Toggle Deep Search", "Toggle deep scraping mode", "toggle_deep"),
            ("Cycle Theme", "Switch to next theme", "cycle_theme"),
        ]

        for name, help_text, action_name in commands:
            if query.lower() in name.lower():
                yield Hit(
                    score=len(query) / len(name) if query else 0,
                    match_display=name,
                    command=getattr(app, f"action_{action_name}", None),
                    help=help_text,
                )
