"""ASCII logo banner — Pythia, Pi/agent-harness style."""

from rich.text import Text
from textual.widgets import Static
from pythia.tui import colors

_PYTHIA_LINES = [
    "██████╗  ██╗   ██╗████████╗██╗  ██╗██╗ █████╗ ",
    "██╔══██╗ ╚██╗ ██╔╝╚══██╔══╝██║  ██║██║██╔══██╗",
    "██████╔╝  ╚████╔╝    ██║   ███████║██║███████║",
    "██╔═══╝    ╚██╔╝     ██║   ██╔══██║██║██╔══██║",
    "██║         ██║      ██║   ██║  ██║██║██║  ██║",
    "╚═╝         ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝",
]

_SUBTITLE = "The Oracle Answers"


def build_logo_text() -> Text:
    logo = Text()
    for line in _PYTHIA_LINES:
        logo.append(f"  {line}\n", style=f"bold {colors.INFO}")
    logo.append(f"\n  {_SUBTITLE}\n", style=f"{colors.PRIMARY}")
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
