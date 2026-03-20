"""Search input — bottom text area with slash command support."""
from __future__ import annotations

from rich.text import Text
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Input, Static


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

    SearchInput > Input {
        width: 100%;
    }

    SearchInput > #mode-label {
        height: 1;
    }
    """

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def compose(self):
        yield Static("", id="mode-label")
        yield Input(id="search-input", placeholder="Ask anything... (!! deep, ?? research)")

    def on_mount(self) -> None:
        inp = self.query_one("#search-input", Input)
        inp.focus()

    def set_mode(self, deep: bool = False) -> None:
        label = self.query_one("#mode-label", Static)
        if deep:
            label.update(Text("  Search [DEEP]", style="bold #ffab40"))
        else:
            label.update(Text("  Search", style="#666666"))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if text:
            self.post_message(self.Submitted(text))
            self.query_one("#search-input", Input).value = ""
