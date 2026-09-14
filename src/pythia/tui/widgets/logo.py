"""ASCII logo banner — Pythia, Pi/agent-harness style."""

from rich.text import Text
from textual.widgets import Static
from pythia.tui import colors

_SUBTITLE = "THE RESEARCH DESK  /  Search. Connect. Understand."


def build_logo_text() -> Text:
    logo = Text()
    logo.append("◈  P Y T H I A\n", style=f"bold {colors.PRIMARY}")
    logo.append(f"{_SUBTITLE}\n", style=colors.MUTED)
    logo.append("\n01  SEARCH    02  RESEARCH    03  HISTORY    04  SYSTEM", style=colors.SUBTEXT)
    logo.append("\nType a question below · /help for commands", style=colors.MUTED)
    return logo


class LogoBanner(Static):
    DEFAULT_CSS = """
    LogoBanner {
        padding: 1 0;
        margin: 0 0 1 0;
        content-align: center middle;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(build_logo_text(), **kwargs)
