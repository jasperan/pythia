"""History screen — filterable query history with re-run."""
from __future__ import annotations
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static
from pythia.config import PythiaConfig


class HistoryScreen(Screen):
    def __init__(self, config: PythiaConfig) -> None:
        super().__init__()
        self.config = config
        self._api_base = f"http://{config.server.host}:{config.server.port}"
        if config.server.host == "0.0.0.0":
            self._api_base = f"http://127.0.0.1:{config.server.port}"

    def compose(self) -> ComposeResult:
        yield Static("  History screen — press [1] to go back to Search", id="history-placeholder")
