"""Cache badge — shows cache hit/miss status after search."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class CacheBadge(Static):
    DEFAULT_CSS = """
    CacheBadge {
        height: 1;
        padding: 0 1;
        margin: 1 0 0 0;
    }
    """

    def show_cache_hit(self, similarity: float, time_ms: int) -> None:
        text = Text()
        text.append("  \u25cf ", style="bold #a6e3a1")
        text.append(f"from cache ({similarity:.2f} similarity)", style="#a6e3a1")
        text.append(f" \u00b7 {time_ms}ms", style="#585b70")
        self.update(text)

    def show_web_search(self, time_ms: int, sources_count: int) -> None:
        text = Text()
        text.append("  \u25cf ", style="bold #89dceb")
        text.append(f"web search ({sources_count} sources)", style="#89dceb")
        time_str = f"{time_ms}ms" if time_ms < 1000 else f"{time_ms / 1000:.1f}s"
        text.append(f" \u00b7 {time_str}", style="#585b70")
        self.update(text)
