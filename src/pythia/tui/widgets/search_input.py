"""Search input — bottom text area with slash command support."""
from __future__ import annotations

from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Input


class SearchInput(Vertical):
    DEFAULT_CSS = """
    SearchInput {
        height: 3;
        border-top: solid #5f87ff;
        padding: 0 1;
        background: #1a2535;
    }
    
    SearchInput > Input {
        width: 100%;
    }
    """

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def compose(self):
        yield Input(id="search-input", placeholder="Ask anything...")

    def on_mount(self) -> None:
        inp = self.query_one("#search-input", Input)
        inp.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        text = event.value.strip()
        if text:
            self.post_message(self.Submitted(text))
            self.query_one("#search-input", Input).value = ""
