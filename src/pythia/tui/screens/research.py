"""Research theater screen — live visualization of multi-round research."""
from __future__ import annotations
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static
from pythia.config import PythiaConfig
from pythia.tui.widgets.search_input import SearchInput


class ResearchScreen(Screen):
    def __init__(self, config: PythiaConfig) -> None:
        super().__init__()
        self.config = config
        self._api_base = f"http://{config.server.host}:{config.server.port}"
        if config.server.host == "0.0.0.0":
            self._api_base = f"http://127.0.0.1:{config.server.port}"

    def compose(self) -> ComposeResult:
        yield Static("  Research screen — press [1] to go back to Search", id="research-placeholder")
        yield SearchInput()
