"""Source list — numbered citations below the answer."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class SourceList(Static):
    DEFAULT_CSS = """
    SourceList {
        height: auto;
        padding: 0 1;
        margin: 1 0 0 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._sources: list[dict] = []

    def add_source(self, source: dict) -> None:
        self._sources.append(source)
        self._rebuild()

    def clear_sources(self) -> None:
        self._sources = []
        self.update("")

    def _rebuild(self) -> None:
        if not self._sources:
            self.update("")
            return
        text = Text()
        text.append("  \u2500\u2500\u2500 Sources \u2500\u2500\u2500\n", style="bold #89b4fa")
        for s in self._sources:
            idx = s.get("index", "?")
            title = s.get("title", "")
            url = s.get("url", "")
            text.append(f"  [{idx}] ", style="bold #89dceb")
            text.append(f"{title}\n", style="#cdd6f4")
            text.append(f"      {url}\n", style="#585b70")
        self.update(text)
