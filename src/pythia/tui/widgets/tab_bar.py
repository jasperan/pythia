"""Tab bar — screen navigation across top."""
from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text
from textual.message import Message
from textual.widgets import Static


@dataclass
class TabDef:
    key: str
    label: str
    screen_name: str


class TabBar(Static):
    DEFAULT_CSS = """
    TabBar {
        height: 1;
        dock: top;
        background: #333345;
        padding: 0 0;
    }
    """

    class TabSelected(Message):
        def __init__(self, screen_name: str) -> None:
            super().__init__()
            self.screen_name = screen_name

    def __init__(self, tabs: list[TabDef], **kwargs) -> None:
        super().__init__("", **kwargs)
        self._tabs = tabs
        self._active = tabs[0].screen_name if tabs else ""

    def set_active(self, screen_name: str) -> None:
        valid = {t.screen_name for t in self._tabs}
        if screen_name in valid:
            self._active = screen_name
            if self.is_attached:
                self._rebuild()

    def on_mount(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        bar = Text()
        bar.append(" ")
        for tab in self._tabs:
            if tab.screen_name == self._active:
                bar.append(f" [{tab.key}] {tab.label} ", style="bold #00d7ff on #1e1e2e")
            else:
                bar.append(f" [{tab.key}] {tab.label} ", style="#808080")
            bar.append(" ")
        self.update(bar)

    def on_click(self, event) -> None:
        x = event.x
        offset = 1
        for tab in self._tabs:
            label_len = len(f" [{tab.key}] {tab.label} ") + 1
            if offset <= x < offset + label_len:
                self.set_active(tab.screen_name)
                self.post_message(self.TabSelected(tab.screen_name))
                return
            offset += label_len
