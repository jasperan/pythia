"""Status bar — model, Oracle status, SearXNG status, cache size."""
from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class PythiaStatusBar(Static):
    DEFAULT_CSS = """
    PythiaStatusBar {
        height: 1;
        dock: bottom;
        background: #333345;
        color: #808080;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._model = ""
        self._oracle_ok = False
        self._searxng_ok = False
        self._cache_size = 0

    def update_status(self, model: str | None = None, oracle_ok: bool | None = None, searxng_ok: bool | None = None, cache_size: int | None = None) -> None:
        if model is not None:
            self._model = model
        if oracle_ok is not None:
            self._oracle_ok = oracle_ok
        if searxng_ok is not None:
            self._searxng_ok = searxng_ok
        if cache_size is not None:
            self._cache_size = cache_size
        self._rebuild()

    def _rebuild(self) -> None:
        bar = Text()
        bar.append(" ")
        bar.append("Model: ", style="#666666")
        bar.append(self._model, style="bold #00d7ff")
        bar.append(" \u2502 ", style="#666666")
        oracle_dot = "\u25cf" if self._oracle_ok else "\u25cb"
        oracle_style = "#b5bd68" if self._oracle_ok else "#cc6666"
        bar.append("Oracle: ", style="#666666")
        bar.append(f"{oracle_dot} ", style=oracle_style)
        bar.append(" \u2502 ", style="#666666")
        searxng_dot = "\u25cf" if self._searxng_ok else "\u25cb"
        searxng_style = "#b5bd68" if self._searxng_ok else "#cc6666"
        bar.append("SearXNG: ", style="#666666")
        bar.append(f"{searxng_dot} ", style=searxng_style)
        bar.append(" \u2502 ", style="#666666")
        bar.append("Cache: ", style="#666666")
        bar.append(str(self._cache_size), style="#8abeb7")
        self.update(bar)
