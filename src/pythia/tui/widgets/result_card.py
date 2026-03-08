"""Result card — streaming markdown answer area."""
from __future__ import annotations

from rich.markdown import Markdown
from textual.widgets import Static


class ResultCard(Static):
    DEFAULT_CSS = """
    ResultCard {
        padding: 0 1;
        margin: 0;
        height: auto;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._tokens: list[str] = []

    def append_token(self, token: str) -> None:
        self._tokens.append(token)
        raw = "".join(self._tokens)
        self.update(Markdown(raw))

    def set_content(self, content: str) -> None:
        self._tokens = [content]
        self.update(Markdown(content))

    def clear_content(self) -> None:
        self._tokens = []
        self.update("")
