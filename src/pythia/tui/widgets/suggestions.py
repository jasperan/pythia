"""Follow-up suggestions widget — shows clickable related queries."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static


class Suggestions(Widget):
    """Displays follow-up question suggestions the user can click to search."""

    DEFAULT_CSS = """
    Suggestions {
        height: auto;
        margin: 1 0 0 0;
        padding: 0 1;
    }
    Suggestions .suggestions-label {
        color: $text-muted;
        margin-bottom: 0;
    }
    Suggestions .suggestion-btn {
        margin: 0 1 0 0;
        min-width: 10;
        height: 1;
        background: $surface;
        color: $text;
        border: none;
    }
    Suggestions .suggestion-btn:hover {
        background: $accent;
        color: $text;
    }
    """

    class Selected(Message):
        """Emitted when a suggestion is clicked."""
        def __init__(self, query: str) -> None:
            super().__init__()
            self.query = query

    def __init__(self, suggestions: list[str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._suggestions = suggestions or []

    def compose(self) -> ComposeResult:
        if self._suggestions:
            yield Static("Follow-up:", classes="suggestions-label")
            with Horizontal():
                for i, s in enumerate(self._suggestions, 1):
                    label = f"{i}. {s[:60]}" if len(s) > 60 else f"{i}. {s}"
                    yield Button(label, classes="suggestion-btn", id=f"suggestion-{i}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id.startswith("suggestion-"):
            idx = int(btn_id.split("-")[1]) - 1
            if 0 <= idx < len(self._suggestions):
                self.post_message(self.Selected(self._suggestions[idx]))
