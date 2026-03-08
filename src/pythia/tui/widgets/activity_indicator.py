"""Activity indicator with spinning dot — Pythia search phases."""
from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

_BRAILLE = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"


class ActivityIndicator(Static):
    DEFAULT_CSS = """
    ActivityIndicator {
        height: 1;
        padding: 0 1;
        margin: 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._frame = 0
        self._timer = None
        self._label = ""

    def on_mount(self) -> None:
        self._timer = self.set_interval(1 / 10, self._tick)

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_BRAILLE)
        self._rebuild()

    def set_label(self, label: str) -> None:
        self._label = label
        self._rebuild()

    def _rebuild(self) -> None:
        if not self._label:
            self.update("")
            return
        line = Text()
        spinner = _BRAILLE[self._frame]
        line.append(f"  {spinner} ", style="bold #00d7ff")
        line.append(self._label, style="#00d7ff")
        self.update(line)

    def stop(self) -> None:
        if self._timer:
            self._timer.stop()
            self._timer = None
        self.update("")

    def on_unmount(self) -> None:
        if self._timer:
            self._timer.stop()
