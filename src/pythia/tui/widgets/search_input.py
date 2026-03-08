"""Search input — bottom text area with slash command support."""
from __future__ import annotations

from textual.containers import Vertical
from textual.message import Message
from textual.widgets import TextArea


class SearchInput(Vertical):
    DEFAULT_CSS = """
    SearchInput {
        height: auto;
        min-height: 3;
        max-height: 5;
        border-top: solid #5f87ff;
        padding: 0 1;
        background: #1a2535;
    }
    """

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def compose(self):
        yield TextArea(id="search-textarea")

    def on_mount(self) -> None:
        ta = self.query_one("#search-textarea", TextArea)
        ta.focus()

    def on_key(self, event) -> None:
        if event.key == "enter" and not event.shift:
            event.prevent_default()
            ta = self.query_one("#search-textarea", TextArea)
            text = ta.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                ta.clear()
