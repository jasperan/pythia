"""Session divider — separates search results with query text and timestamp."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static
from pythia.tui import colors


class SessionDivider(Static):
    DEFAULT_CSS = """
    SessionDivider {
        height: 1;
        margin: 1 0;
        color: $primary;
    }
    """

    def __init__(self, query: str, timestamp: str, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._query = query[:60] + "..." if len(query) > 60 else query
        self._timestamp = timestamp

    def on_mount(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        line = Text()
        line.append("─── ", style=f"{colors.PRIMARY}")
        line.append(f'"{self._query}"', style=f"bold {colors.PRIMARY}")
        line.append(f" ─── {self._timestamp} ", style=f"{colors.PRIMARY}")
        line.append("─" * 20, style=f"{colors.PRIMARY}")
        self.update(line)
