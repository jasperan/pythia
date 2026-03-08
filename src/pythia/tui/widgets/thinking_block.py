"""Collapsible thinking indicator — agent-harness / Pi style."""
from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static

_BRAILLE = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"


class ThinkingBlock(Static):
    DEFAULT_CSS = """
    ThinkingBlock {
        height: auto;
        min-height: 1;
        padding: 0 1;
        margin: 0;
    }
    """

    is_done: reactive[bool] = reactive(False)

    def __init__(self, label: str = "Thinking", **kwargs) -> None:
        super().__init__("", **kwargs)
        self._frame = 0
        self._timer = None
        self._label = label

    def on_mount(self) -> None:
        self._timer = self.set_interval(1 / 10, self._tick)
        self._rebuild()

    def _tick(self) -> None:
        if not self.is_done:
            self._frame = (self._frame + 1) % len(_BRAILLE)
            self._rebuild()

    def finish(self) -> None:
        self.is_done = True
        if self._timer:
            self._timer.stop()
            self._timer = None

    def watch_is_done(self, value: bool) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        line = Text()
        if self.is_done:
            line.append("\u25cf ", style="bold #b5bd68")
            line.append(self._label, style="#666666")
        else:
            spinner = _BRAILLE[self._frame]
            line.append(f"{spinner} ", style="bold #b294bb")
            line.append(self._label, style="#b294bb")
        self.update(line)
