"""Textual TUI app for Pythia."""
from __future__ import annotations

import time

from textual.app import App

from pythia.config import PythiaConfig
from pythia.tui.screens.search import SearchScreen


class PythiaApp(App):
    TITLE = "Pythia"
    CSS_PATH = "themes/dark.tcss"

    _CTRL_C_WINDOW = 1.0

    def __init__(self, config: PythiaConfig) -> None:
        super().__init__()
        self.config = config
        self._last_ctrl_c: float = 0.0

    def on_mount(self) -> None:
        self.push_screen(SearchScreen(self.config))

    def _on_key(self, event) -> None:
        if getattr(event, "key", None) == "ctrl+c":
            now = time.monotonic()
            if now - self._last_ctrl_c < self._CTRL_C_WINDOW:
                self.exit()
            else:
                self._last_ctrl_c = now
                self.notify("Press Ctrl+C again to quit", severity="warning", timeout=2)


def run_tui(config: PythiaConfig) -> None:
    app = PythiaApp(config)
    app.run()
