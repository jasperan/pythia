"""Result card — streaming markdown answer area."""

from __future__ import annotations

from rich.markdown import Markdown
from rich.theme import Theme
from textual.widgets import Static

from pythia.tui import colors


class ResearchMarkdown(Markdown):
    """Keep Rich's markdown defaults inside the research desk palette."""

    def __rich_console__(self, console, options):
        theme = Theme({
            "markdown.h1": f"bold {colors.PRIMARY}",
            "markdown.h2": f"bold {colors.PRIMARY}",
            "markdown.h3": f"bold {colors.SECONDARY}",
            "markdown.link": colors.INFO,
            "markdown.link_url": f"underline {colors.INFO}",
            "markdown.code": colors.SECONDARY,
        })
        with console.use_theme(theme):
            yield from super().__rich_console__(console, options)


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
        self.update(ResearchMarkdown(raw))

    def set_content(self, content: str) -> None:
        self._tokens = [content]
        self.update(ResearchMarkdown(content))

    def clear_content(self) -> None:
        self._tokens = []
        self.update("")
