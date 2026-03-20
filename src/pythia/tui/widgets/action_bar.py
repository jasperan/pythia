"""Action bar — dashboard action buttons."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button


class ActionBar(Horizontal):
    DEFAULT_CSS = """
    ActionBar {
        height: 3;
        padding: 0 2;
        align: center middle;
    }
    ActionBar Button { margin: 0 1; }
    """

    class ActionRequested(Message):
        def __init__(self, action: str) -> None:
            super().__init__()
            self.action = action

    def compose(self) -> ComposeResult:
        yield Button("Clear Cache", id="btn-clear-cache", variant="error")
        yield Button("Export History", id="btn-export", variant="default")
        yield Button("Refresh", id="btn-refresh", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        action_map = {
            "btn-clear-cache": "clear_cache",
            "btn-export": "export_history",
            "btn-refresh": "refresh",
        }
        action = action_map.get(event.button.id, "")
        if action:
            self.post_message(self.ActionRequested(action))
