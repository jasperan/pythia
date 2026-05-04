"""Research theater screen — live visualization of multi-round research."""
from __future__ import annotations

import json
import time

import httpx
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen

from pythia.config import PythiaConfig
from pythia.tui.widgets.activity_indicator import ActivityIndicator
from pythia.tui.widgets.research_progress import ResearchProgressBar
from pythia.tui.widgets.research_tree import ResearchTree
from pythia.tui.widgets.result_card import ResultCard
from pythia.tui.widgets.search_input import SearchInput
from pythia.tui.widgets.source_list import SourceList


class ResearchScreen(Screen):
    DEFAULT_CSS = """
    ResearchScreen { layout: vertical; }
    #research-split { height: 1fr; }
    #research-tree-pane { width: 30; border-right: solid #5f87ff; overflow-y: auto; }
    #research-main-pane { width: 1fr; overflow-y: auto; padding: 1 2; }
    """

    def __init__(self, config: PythiaConfig, host: str | None = None, port: int | None = None) -> None:
        super().__init__()
        self.config = config
        api_host = host or config.server.host
        api_port = port or config.server.port
        if api_host == "0.0.0.0":
            api_host = "127.0.0.1"
        self._api_base = f"http://{api_host}:{api_port}"
        self._findings_count = 0
        self._sources_count = 0
        self._start_time = 0.0

    def compose(self) -> ComposeResult:
        with Horizontal(id="research-split"):
            with VerticalScroll(id="research-tree-pane"):
                yield ResearchTree()
            with VerticalScroll(id="research-main-pane"):
                yield ResultCard()
                yield SourceList()
        yield ResearchProgressBar()
        yield ActivityIndicator()
        yield SearchInput()

    def on_mount(self) -> None:
        # Check for pending research query from search screen or history
        from pythia.tui.app import PythiaApp
        app = self.app
        if isinstance(app, PythiaApp) and app._pending_research_query:
            query = app._pending_research_query
            app._pending_research_query = None
            if query:
                self.call_later(lambda: self._run_research(query))

    async def on_search_input_submitted(self, event: SearchInput.Submitted) -> None:
        query = event.value
        if query.startswith("??"):
            query = query[2:].strip()
        if not query:
            return
        await self._run_research(query)

    async def _run_research(self, query: str) -> None:
        tree = self.query_one(ResearchTree)
        result_card = self.query_one(ResultCard)
        source_list = self.query_one(SourceList)
        progress = self.query_one(ResearchProgressBar)
        activity = self.query_one(ActivityIndicator)

        # Reset state
        tree.reset()
        result_card.clear_content()
        source_list.clear_sources()
        self._findings_count = 0
        self._sources_count = 0
        self._start_time = time.monotonic()

        activity.set_label("Starting research...")
        current_round = 0
        max_rounds = self.config.research.max_rounds

        event_type = ""
        try:
            async with httpx.AsyncClient(timeout=300.0) as client, client.stream(
                "POST",
                f"{self._api_base}/research",
                json={"query": query, "model": self.config.ollama.model},
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        elapsed = int((time.monotonic() - self._start_time) * 1000)

                        if event_type == "status":
                            activity.set_label(data.get("message", ""))

                        elif event_type == "recall":
                            findings = data.get("findings", [])
                            tree.set_recall(findings)

                        elif event_type == "plan":
                            sub_queries = data.get("sub_queries", [])
                            tree.add_plan(sub_queries)

                        elif event_type == "round_start":
                            current_round = data.get("round", 1)
                            max_rounds = data.get("max_rounds", max_rounds)
                            tree.start_round(current_round, max_rounds)
                            progress.update_progress(
                                round_num=current_round, max_rounds=max_rounds,
                                elapsed_ms=elapsed,
                            )

                        elif event_type == "finding":
                            sq = data.get("sub_query", "")
                            num_src = data.get("num_sources", 0)
                            preview = data.get("summary_preview", "")
                            tree.complete_finding(sq, num_sources=num_src, preview=preview)
                            self._findings_count += 1
                            self._sources_count += num_src
                            progress.update_progress(
                                findings=self._findings_count,
                                sources=self._sources_count,
                                elapsed_ms=elapsed,
                            )

                        elif event_type == "gap_analysis":
                            gaps = data.get("gaps", [])
                            reasoning = data.get("reasoning", "")
                            sufficient = data.get("sufficient", True)
                            if not sufficient and gaps:
                                tree.add_gaps(gaps, reasoning=reasoning)

                        elif event_type == "token":
                            result_card.append_token(data.get("content", ""))

                        elif event_type == "done":
                            activity.stop()
                            tree.mark_complete()
                            progress.update_progress(
                                round_num=data.get("rounds_used", current_round),
                                max_rounds=max_rounds,
                                findings=data.get("total_findings", self._findings_count),
                                sources=data.get("total_sources", self._sources_count),
                                elapsed_ms=data.get("elapsed_ms", elapsed),
                            )

        except Exception as e:
            activity.stop()
            result_card.set_content(f"**Error:** {e}")
