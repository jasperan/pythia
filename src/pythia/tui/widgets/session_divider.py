"""Session divider — separates search results with query text and timestamp."""
from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class SessionDivider(Static):
    DEFAULT_CSS = """
    SessionDivider {
        height: 1;
        margin: 1 0;
        color: #5f87ff;
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
        line.append("─── ", style="#5f87ff")
        line.append(f'"{self._query}"', style="bold #5f87ff")
        line.append(f" ─── {self._timestamp} ", style="#5f87ff")
        line.append("─" * 20, style="#5f87ff")
        self.update(line)
