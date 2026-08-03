"""Research progress bar — round N/M with live counters."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static
from pythia.tui import colors


class ResearchProgressBar(Static):
    DEFAULT_CSS = """
    ResearchProgressBar {
        height: 1;
        padding: 0 1;
        background: $panel;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._current_round = 0
        self._max_rounds = 0
        self._findings = 0
        self._sources = 0
        self._elapsed_ms = 0

    def update_progress(
        self,
        round_num: int = 0,
        max_rounds: int = 0,
        findings: int = 0,
        sources: int = 0,
        elapsed_ms: int = 0,
    ) -> None:
        if round_num:
            self._current_round = round_num
        if max_rounds:
            self._max_rounds = max_rounds
        if findings:
            self._findings = findings
        if sources:
            self._sources = sources
        if elapsed_ms:
            self._elapsed_ms = elapsed_ms
        self._rebuild()

    def _rebuild(self) -> None:
        if self._max_rounds == 0:
            if self.is_attached:
                self.update("")
            return

        bar = Text()
        bar.append("  ")

        filled = self._current_round
        total = self._max_rounds
        for i in range(total):
            if i < filled:
                bar.append("▰", style=f"bold {colors.INFO}")
            else:
                bar.append("▱", style=f"{colors.DIM}")

        bar.append(f"  Round {self._current_round}/{self._max_rounds}", style=f"{colors.TEXT}")
        bar.append(f" · {self._findings} findings", style=f"{colors.SUCCESS}")
        bar.append(f" · {self._sources} sources", style=f"{colors.INFO}")

        elapsed_str = (
            f"{self._elapsed_ms}ms"
            if self._elapsed_ms < 1000
            else f"{self._elapsed_ms / 1000:.1f}s"
        )
        bar.append(f" · {elapsed_str}", style=f"{colors.DIM}")

        if self.is_attached:
            self.update(bar)
