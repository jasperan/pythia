"""Main search screen — composes all widgets into the Pythia TUI."""
from __future__ import annotations

import json
from datetime import datetime

import httpx
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen

from pythia.config import PythiaConfig
from pythia.services import ServiceManager, ServiceInfo
from pythia.tui.widgets.activity_indicator import ActivityIndicator
from pythia.tui.widgets.cache_badge import CacheBadge
from pythia.tui.widgets.logo import LogoBanner
from pythia.tui.widgets.result_card import ResultCard
from pythia.tui.widgets.search_input import SearchInput
from pythia.tui.widgets.service_status import ServiceStatusIndicator
from pythia.tui.widgets.session_divider import SessionDivider
from pythia.tui.widgets.source_list import SourceList
from pythia.tui.widgets.status_bar import PythiaStatusBar


class SearchScreen(Screen):
    def __init__(self, config: PythiaConfig) -> None:
        super().__init__()
        self.config = config
        self._api_base = f"http://{config.server.host}:{config.server.port}"
        if config.server.host == "0.0.0.0":
            self._api_base = f"http://127.0.0.1:{config.server.port}"
        self._service_manager: ServiceManager | None = None
        self._health_check_interval = None

    def compose(self) -> ComposeResult:
        yield LogoBanner()
        yield VerticalScroll(id="results-area")
        yield ActivityIndicator()
        yield SearchInput()
        yield ServiceStatusIndicator(id="service-status")
        yield PythiaStatusBar()

    def on_mount(self) -> None:
        self._health_check_interval = self.set_interval(2.0, self._check_health)
        self._try_connect_service_manager()
        # Check for pending search query from history re-run
        from pythia.tui.app import PythiaApp
        app = self.app
        if isinstance(app, PythiaApp) and app._pending_search_query:
            query = app._pending_search_query
            app._pending_search_query = None
            self.call_later(lambda: self._run_search(query))

    _SERVICE_CONNECT_MAX_RETRIES = 60  # 30 seconds at 0.5s intervals

    def _try_connect_service_manager(self, _retries: int = 0) -> None:
        if not self.is_attached:
            return
        from pythia.tui.app import PythiaApp
        app = self.app
        if isinstance(app, PythiaApp):
            if app._service_manager:
                self._service_manager = app._service_manager
                self._service_manager.register_status_callback(self._on_service_status_update)
            elif app._auto_start and _retries < self._SERVICE_CONNECT_MAX_RETRIES:
                self.set_timer(0.5, lambda: self._try_connect_service_manager(_retries + 1))

    def on_unmount(self) -> None:
        if self._health_check_interval is not None:
            self._health_check_interval.stop()
            self._health_check_interval = None

    def _on_service_status_update(self, statuses: dict[str, ServiceInfo]) -> None:
        try:
            status_widget = self.query_one("#service-status", ServiceStatusIndicator)
            if status_widget:
                status_widget.update_services(statuses)
        except Exception:
            pass

    async def _check_health(self) -> None:
        try:
            status = self.query_one(PythiaStatusBar)
        except Exception:
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._api_base}/health")
                data = resp.json()
                status.update_status(
                    model=self.config.ollama.model,
                    oracle_ok=data.get("oracle", False),
                    searxng_ok=data.get("searxng", False),
                    cache_size=data.get("cache_size", 0),
                )
        except Exception:
            status.update_status(model=self.config.ollama.model, oracle_ok=False, searxng_ok=False, cache_size=0)

    async def on_search_input_submitted(self, event: SearchInput.Submitted) -> None:
        query = event.value
        if query.startswith("/"):
            await self._handle_command(query)
            return

        deep = False
        if query.startswith("!!"):
            query = query[2:].strip()
            deep = True
        elif query.startswith("??"):
            query = query[2:].strip()
            if query:
                from pythia.tui.app import PythiaApp
                app = self.app
                if isinstance(app, PythiaApp):
                    app._pending_research_query = query
                    app._switch_to("research")
            return

        if not query:
            return

        # Get deep mode from app if not explicitly set by prefix
        if not deep:
            from pythia.tui.app import PythiaApp
            app = self.app
            if isinstance(app, PythiaApp):
                deep = app._deep_mode

        await self._run_search(query, deep=deep)

    async def _run_search(self, query: str, deep: bool = False) -> None:
        results_area = self.query_one("#results-area", VerticalScroll)
        activity = self.query_one(ActivityIndicator)

        # Add session divider
        timestamp = datetime.now().strftime("%-I:%M %p")
        divider = SessionDivider(query=query, timestamp=timestamp)
        await results_area.mount(divider)

        # Create new result group
        result_card = ResultCard()
        source_list = SourceList()
        cache_badge = CacheBadge()
        await results_area.mount(result_card)
        await results_area.mount(source_list)
        await results_area.mount(cache_badge)

        results_area.scroll_end()
        activity.set_label("Searching...")

        event_type = ""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                req_body = {"query": query}
                if deep:
                    req_body["deep"] = True
                async with client.stream("POST", f"{self._api_base}/search", json=req_body) as resp:
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

                            if event_type == "status":
                                activity.set_label(data.get("message", ""))
                            elif event_type == "source":
                                source_list.add_source(data)
                            elif event_type == "token":
                                result_card.append_token(data.get("content", ""))
                            elif event_type == "done":
                                activity.stop()
                                if data.get("cache_hit"):
                                    cache_badge.show_cache_hit(data.get("similarity", 0), data.get("response_time_ms", 0))
                                else:
                                    cache_badge.show_web_search(data.get("response_time_ms", 0), data.get("sources_count", 0))
                                await self._check_health()
                                results_area.scroll_end()
        except Exception as e:
            activity.stop()
            result_card.set_content(f"**Error:** {e}")

    async def _handle_command(self, command: str) -> None:
        results_area = self.query_one("#results-area", VerticalScroll)
        # Mount a fresh ResultCard for command output
        result_card = ResultCard()
        await results_area.mount(result_card)
        results_area.scroll_end()

        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/clear":
            await results_area.remove_children()
        elif cmd == "/history":
            from pythia.tui.app import PythiaApp
            app = self.app
            if isinstance(app, PythiaApp):
                app._switch_to("history")
        elif cmd == "/stats":
            from pythia.tui.app import PythiaApp
            app = self.app
            if isinstance(app, PythiaApp):
                app._switch_to("dashboard")
        elif cmd == "/model":
            if len(parts) > 1:
                new_model = parts[1].strip()
                self.config.ollama.model = new_model
                try:
                    self.query_one(PythiaStatusBar).update_status(model=new_model)
                except Exception:
                    pass
                result_card.set_content(f"Model switched to **{new_model}**")
            else:
                result_card.set_content("Usage: `/model <model_name>`")
        elif cmd == "/cache":
            if len(parts) > 1 and parts[1].strip() == "clear":
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        resp = await client.delete(f"{self._api_base}/cache")
                        data = resp.json()
                    result_card.set_content(f"Cache cleared. **{data.get('deleted', 0)}** entries deleted.")
                    await self._check_health()
                except Exception as e:
                    result_card.set_content(f"**Error:** {e}")
            else:
                result_card.set_content("Usage: `/cache clear`")
        elif cmd == "/help":
            result_card.set_content(
                "## Commands\n\n"
                "- `/history` — Switch to History screen\n"
                "- `/stats` — Switch to Dashboard screen\n"
                "- `/model <name>` — Switch Ollama model\n"
                "- `/cache clear` — Purge Oracle cache\n"
                "- `/clear` — Clear screen\n"
                "- `/help` — Show this help\n\n"
                "## Search Prefixes\n\n"
                "- `!! <query>` — Deep search (scrapes pages)\n"
                "- `?? <query>` — Research mode (multi-round)\n"
            )
        else:
            result_card.set_content(f"Unknown command: `{cmd}`. Type `/help` for available commands.")
