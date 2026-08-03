"""History list — filterable query history with keyboard navigation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from rich.text import Text
from textual.widgets import Static
from pythia.tui import colors


class HistoryFilter(Enum):
    ALL = auto()
    CACHE_HITS = auto()
    WEB_SEARCHES = auto()
    RESEARCH = auto()


@dataclass
class HistoryEntry:
    query: str
    cache_hit: bool
    response_time_ms: int
    model: str
    is_research: bool
    timestamp: str = ""


class HistoryList(Static):
    DEFAULT_CSS = """
    HistoryList {
        height: 1fr;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._entries: list[HistoryEntry] = []
        self._filter = HistoryFilter.ALL
        self._text_filter = ""
        self._selected_index = 0

    def load_entries(self, entries: list[HistoryEntry]) -> None:
        self._entries = entries
        self._selected_index = 0
        if self.is_attached:
            self._rebuild()

    def set_text_filter(self, text: str) -> None:
        self._text_filter = text.lower()
        self._selected_index = 0
        if self.is_attached:
            self._rebuild()

    def _get_visible(self) -> list[HistoryEntry]:
        entries = self._entries
        if self._filter == HistoryFilter.CACHE_HITS:
            entries = [e for e in entries if e.cache_hit]
        elif self._filter == HistoryFilter.WEB_SEARCHES:
            entries = [e for e in entries if not e.cache_hit and not e.is_research]
        elif self._filter == HistoryFilter.RESEARCH:
            entries = [e for e in entries if e.is_research]
        if self._text_filter:
            entries = [e for e in entries if self._text_filter in e.query.lower()]
        return entries

    def move_selection(self, delta: int) -> None:
        visible = self._get_visible()
        if not visible:
            return
        self._selected_index = max(0, min(len(visible) - 1, self._selected_index + delta))
        self._rebuild()

    def get_selected(self) -> HistoryEntry | None:
        visible = self._get_visible()
        if 0 <= self._selected_index < len(visible):
            return visible[self._selected_index]
        return None

    def _rebuild(self) -> None:
        visible = self._get_visible()
        if not visible:
            self.update("  No matching queries.")
            return

        text = Text()
        for i, entry in enumerate(visible):
            is_selected = i == self._selected_index

            if entry.is_research:
                text.append("  \u25c8 ", style=f"bold {colors.SECONDARY}")
            elif entry.cache_hit:
                text.append("  \u25cf ", style=f"bold {colors.SUCCESS}")
            else:
                text.append("  \u25cb ", style=f"bold {colors.INFO}")

            if entry.timestamp:
                text.append(f"{entry.timestamp}  ", style=f"{colors.DIM}")

            q = entry.query
            if len(q) > 55:
                q = q[:52] + "..."
            style = f"bold {colors.TEXT}" if is_selected else f"{colors.TEXT}"
            if is_selected:
                text.append(f"\u25b8 {q}\n", style=style)
            else:
                text.append(f"  {q}\n", style=style)

            if entry.is_research:
                text.append("              \U0001f52c research", style=f"{colors.SECONDARY}")
            elif entry.cache_hit:
                text.append("              \u26a1 cache hit", style=f"{colors.SUCCESS}")
            else:
                text.append("              \U0001f50d web search", style=f"{colors.INFO}")

            time_str = (
                f"{entry.response_time_ms}ms"
                if entry.response_time_ms < 1000
                else f"{entry.response_time_ms / 1000:.1f}s"
            )
            text.append(f" \u00b7 {time_str}", style=f"{colors.DIM}")
            text.append(f" \u00b7 {entry.model}\n\n", style=f"{colors.DIM}")

        self.update(text)
