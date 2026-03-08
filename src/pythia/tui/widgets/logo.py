"""ASCII logo banner — Pythia, Pi/agent-harness style."""
from rich.text import Text
from textual.widgets import Static

_PYTHIA_LINES = [
    "██████╗  ██╗   ██╗████████╗██╗  ██╗██╗ █████╗ ",
    "██╔══██╗ ╚██╗ ██╔╝╚══██╔══╝██║  ██║██║██╔══██╗",
    "██████╔╝  ╚████╔╝    ██║   ███████║██║███████║",
    "██╔═══╝    ╚██╔╝     ██║   ██╔══██║██║██╔══██║",
    "██║         ██║      ██║   ██║  ██║██║██║  ██║",
    "╚═╝         ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝",
]

_SUBTITLE = "The Oracle Answers"

_CYAN_GRADIENT = [
    "#00d7ff",
    "#00c4ee",
    "#00b1dd",
    "#009ecc",
    "#008bbb",
    "#0078aa",
]


def build_logo_text() -> Text:
    logo = Text()
    for i, line in enumerate(_PYTHIA_LINES):
        color = _CYAN_GRADIENT[i % len(_CYAN_GRADIENT)]
        logo.append(f"  {line}\n", style=f"bold {color}")
    logo.append(f"\n  {_SUBTITLE}\n", style="#5f87ff")
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
