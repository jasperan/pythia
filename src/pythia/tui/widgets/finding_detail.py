"""Finding detail — expandable view of a single research finding."""
from __future__ import annotations

from rich.markdown import Markdown
from textual.widgets import Static


class FindingDetail(Static):
    DEFAULT_CSS = """
    FindingDetail {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        border: solid #333345;
    }
    """

    def __init__(self, sub_query: str, summary: str, sources: list[dict], round_num: int = 1, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._sub_query = sub_query
        self._summary = summary
        self._sources = sources
        self._round_num = round_num

    def on_mount(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        self.update(Markdown(f"**{self._sub_query}** *(Round {self._round_num})*\n\n{self._summary}"))
