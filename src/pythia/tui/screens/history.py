"""History screen — filterable query history with re-run."""
from __future__ import annotations

import httpx
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Input, Static

from pythia.config import PythiaConfig
from pythia.tui.widgets.history_list import HistoryList, HistoryEntry


class HistoryScreen(Screen):
    DEFAULT_CSS = """
    HistoryScreen { layout: vertical; }
    #history-filter-bar { height: 1; padding: 0 2; background: #2a2a3a; }
    #history-filter-input { height: 3; padding: 0 1; border-top: solid #5f87ff; background: #1a2535; }
    #history-footer { height: 2; dock: bottom; background: #333345; padding: 0 2; }
    """

    BINDINGS = [
        ("j", "move_down", "Down"),
        ("k", "move_up", "Up"),
        ("down", "move_down", "Down"),
        ("up", "move_up", "Up"),
        ("enter", "rerun", "Re-run"),
        ("r", "research", "Research"),
        ("d", "delete_entry", "Delete"),
    ]

    def __init__(self, config: PythiaConfig, host: str | None = None, port: int | None = None) -> None:
        super().__init__()
        self.config = config
        api_host = host or config.server.host
        api_port = port or config.server.port
        if api_host == "0.0.0.0":
            api_host = "127.0.0.1"
        self._api_base = f"http://{api_host}:{api_port}"

    def compose(self) -> ComposeResult:
        yield Static("", id="history-filter-bar")
        yield HistoryList()
        with Vertical(id="history-filter-input"):
            yield Input(id="history-filter", placeholder="Type to filter queries...")
        yield Static("", id="history-footer")

    def on_mount(self) -> None:
        self.call_later(self._load_history)

    async def _load_history(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._api_base}/history", params={"limit": 100})
                history = resp.json()

            entries = []
            for h in history:
                query = h.get("query", "")
                is_research = query.startswith("[research]")
                entries.append(HistoryEntry(
                    query=query.replace("[research] ", "") if is_research else query,
                    cache_hit=h.get("cache_hit", False),
                    response_time_ms=h.get("response_time_ms", 0),
                    model=h.get("model_used", ""),
                    is_research=is_research,
                    timestamp=h.get("timestamp", ""),
                ))

            hl = self.query_one(HistoryList)
            hl.load_entries(entries)
            self._update_footer(entries)

        except Exception as e:
            self.query_one(HistoryList).update(f"  Error loading history: {e}")

    def _update_footer(self, entries: list[HistoryEntry]) -> None:
        total = len(entries)
        hits = sum(1 for e in entries if e.cache_hit)
        rate = f"{hits / total * 100:.0f}%" if total > 0 else "0%"
        avg_ms = sum(e.response_time_ms for e in entries) // max(total, 1)
        avg_str = f"{avg_ms}ms" if avg_ms < 1000 else f"{avg_ms / 1000:.1f}s"

        footer = Text()
        footer.append(f"  {total} queries", style="#e0e0e0")
        footer.append(f" \u00b7 {hits} cache hits ({rate})", style="#b5bd68")
        footer.append(f" \u00b7 avg {avg_str}", style="#666666")
        footer.append("\n  \u2191\u2193/jk Navigate  Enter Re-run  r Research  / Filter", style="#808080")
        self.query_one("#history-footer", Static).update(footer)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "history-filter":
            self.query_one(HistoryList).set_text_filter(event.value)

    def action_move_down(self) -> None:
        self.query_one(HistoryList).move_selection(1)

    def action_move_up(self) -> None:
        self.query_one(HistoryList).move_selection(-1)

    def action_rerun(self) -> None:
        selected = self.query_one(HistoryList).get_selected()
        if selected:
            from pythia.tui.app import PythiaApp
            app = self.app
            if isinstance(app, PythiaApp):
                app._pending_search_query = selected.query
                app._switch_to("search")

    def action_research(self) -> None:
        selected = self.query_one(HistoryList).get_selected()
        if selected:
            from pythia.tui.app import PythiaApp
            app = self.app
            if isinstance(app, PythiaApp):
                app._pending_research_query = selected.query
                app._switch_to("research")

    def action_delete_entry(self) -> None:
        selected = self.query_one(HistoryList).get_selected()
        if selected:
            self.notify("Delete not yet supported by API", timeout=2)
