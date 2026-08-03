"""Stats panel — key metrics display for dashboard."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static
from pythia.tui import colors


class StatsPanel(Static):
    DEFAULT_CSS = """
    StatsPanel {
        height: auto;
        padding: 1 2;
        border: solid $primary;
    }
    """

    def update_stats(self, stats: dict) -> None:
        text = Text()
        text.append("  Cache Stats\n\n", style=f"bold {colors.INFO}")
        text.append("  Total searches:  ", style=f"{colors.MUTED}")
        text.append(f"{stats.get('total_searches', 0)}\n", style=f"bold {colors.TEXT}")
        text.append("  Cache hits:      ", style=f"{colors.MUTED}")
        text.append(f"{stats.get('cache_hits', 0)}\n", style=f"bold {colors.SUCCESS}")
        text.append("  Hit rate:        ", style=f"{colors.MUTED}")
        text.append(f"{stats.get('cache_hit_rate', 0)}%\n", style=f"bold {colors.SUCCESS}")
        text.append("  Cache entries:   ", style=f"{colors.MUTED}")
        text.append(f"{stats.get('cache_size', 0)}\n", style=f"bold {colors.INFO}")
        text.append("  Avg response:    ", style=f"{colors.MUTED}")
        avg = stats.get("avg_response_ms", 0)
        avg_str = f"{avg}ms" if avg < 1000 else f"{avg / 1000:.1f}s"
        text.append(f"{avg_str}\n", style=f"bold {colors.TEXT}")
        text.append("  Active days:     ", style=f"{colors.MUTED}")
        text.append(f"{stats.get('active_days', 0)}\n", style=f"bold {colors.TEXT}")
        self.update(text)
