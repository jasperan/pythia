"""Sparkline panel — response time and cache hit visualization."""
from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

_BARS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587"


class SparklinePanel(Static):
    DEFAULT_CSS = """
    SparklinePanel {
        height: 5;
        padding: 1 2;
        border: solid #5f87ff;
    }
    """

    def update_data(self, history: list[dict]) -> None:
        text = Text()

        # Response times sparkline
        times = [h.get("response_time_ms", 0) for h in history]
        text.append("  Response Times\n  ", style="bold #00d7ff")
        if times:
            max_t = max(times) or 1
            for t in times:
                idx = min(int(t / max_t * (len(_BARS) - 1)), len(_BARS) - 1)
                text.append(_BARS[idx], style="#00d7ff")
            avg = sum(times) // len(times)
            p50 = sorted(times)[len(times) // 2]
            text.append(f"  avg: {avg}ms  p50: {p50}ms\n\n", style="#666666")
        else:
            text.append("  No data\n\n", style="#666666")

        # Cache hit/miss sparkline
        text.append("  Cache Hits\n  ", style="bold #b5bd68")
        for h in history:
            if h.get("cache_hit"):
                text.append("\u2587", style="#b5bd68")
            else:
                text.append("\u2581", style="#cc6666")
        hits = sum(1 for h in history if h.get("cache_hit"))
        text.append(f"  {hits}/{len(history)} hits\n", style="#666666")

        self.update(text)
