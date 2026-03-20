"""Stats panel — key metrics display for dashboard."""
from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class StatsPanel(Static):
    DEFAULT_CSS = """
    StatsPanel {
        height: auto;
        padding: 1 2;
        border: solid #5f87ff;
    }
    """

    def update_stats(self, stats: dict) -> None:
        text = Text()
        text.append("  Cache Stats\n\n", style="bold #00d7ff")
        text.append("  Total searches:  ", style="#808080")
        text.append(f"{stats.get('total_searches', 0)}\n", style="bold #e0e0e0")
        text.append("  Cache hits:      ", style="#808080")
        text.append(f"{stats.get('cache_hits', 0)}\n", style="bold #b5bd68")
        text.append("  Hit rate:        ", style="#808080")
        text.append(f"{stats.get('cache_hit_rate', 0)}%\n", style="bold #b5bd68")
        text.append("  Cache entries:   ", style="#808080")
        text.append(f"{stats.get('cache_size', 0)}\n", style="bold #8abeb7")
        text.append("  Avg response:    ", style="#808080")
        avg = stats.get("avg_response_ms", 0)
        avg_str = f"{avg}ms" if avg < 1000 else f"{avg / 1000:.1f}s"
        text.append(f"{avg_str}\n", style="bold #e0e0e0")
        text.append("  Active days:     ", style="#808080")
        text.append(f"{stats.get('active_days', 0)}\n", style="bold #e0e0e0")
        self.update(text)
