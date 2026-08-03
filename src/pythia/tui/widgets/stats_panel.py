"""Stats panel — key metrics display for dashboard."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class StatsPanel(Static):
    DEFAULT_CSS = """
    StatsPanel {
        height: auto;
        padding: 1 2;
        border: solid #89b4fa;
    }
    """

    def update_stats(self, stats: dict) -> None:
        text = Text()
        text.append("  Cache Stats\n\n", style="bold #89dceb")
        text.append("  Total searches:  ", style="#6c7086")
        text.append(f"{stats.get('total_searches', 0)}\n", style="bold #cdd6f4")
        text.append("  Cache hits:      ", style="#6c7086")
        text.append(f"{stats.get('cache_hits', 0)}\n", style="bold #a6e3a1")
        text.append("  Hit rate:        ", style="#6c7086")
        text.append(f"{stats.get('cache_hit_rate', 0)}%\n", style="bold #a6e3a1")
        text.append("  Cache entries:   ", style="#6c7086")
        text.append(f"{stats.get('cache_size', 0)}\n", style="bold #89dceb")
        text.append("  Avg response:    ", style="#6c7086")
        avg = stats.get("avg_response_ms", 0)
        avg_str = f"{avg}ms" if avg < 1000 else f"{avg / 1000:.1f}s"
        text.append(f"{avg_str}\n", style="bold #cdd6f4")
        text.append("  Active days:     ", style="#6c7086")
        text.append(f"{stats.get('active_days', 0)}\n", style="bold #cdd6f4")
        self.update(text)
