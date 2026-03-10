"""Main search screen — composes all widgets into the Pythia TUI."""
from __future__ import annotations

import asyncio
import json

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
        self._health_check_interval: float | None = None

    def compose(self) -> ComposeResult:
        yield LogoBanner()
        with VerticalScroll(id="results-area"):
            yield ResultCard()
            yield SourceList()
            yield CacheBadge()
        yield ActivityIndicator()
        yield SearchInput()
        yield ServiceStatusIndicator(id="service-status")
        yield PythiaStatusBar()

    def on_mount(self) -> None:
        # Start periodic health checks immediately (they tolerate API being down)
        self._health_check_interval = self.set_interval(2.0, self._check_health)
        # Try to connect to service manager
        self._try_connect_service_manager()

    def _try_connect_service_manager(self) -> None:
        """Try to get reference to service manager from app, retrying if not ready."""
        from pythia.tui.app import PythiaApp
        app = self.app
        if isinstance(app, PythiaApp) and app._service_manager:
            self._service_manager = app._service_manager
            self._service_manager.register_status_callback(self._on_service_status_update)
        else:
            # Retry until service manager is available
            self.set_timer(0.5, self._try_connect_service_manager)

    def on_unmount(self) -> None:
        """Clean up health check interval when screen is unmounted."""
        if self._health_check_interval is not None:
            self.clear_interval(self._health_check_interval)
            self._health_check_interval = None

    def _on_service_status_update(self, statuses: dict[str, ServiceInfo]) -> None:
        """Update service status indicator."""
        status_widget = self.query_one("#service-status", ServiceStatusIndicator)
        if status_widget:
            status_widget.update_services(statuses)

    async def _check_health(self) -> None:
        status = self.query_one(PythiaStatusBar)
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
        await self._run_search(query)

    async def _run_search(self, query: str) -> None:
        result_card = self.query_one(ResultCard)
        source_list = self.query_one(SourceList)
        cache_badge = self.query_one(CacheBadge)
        activity = self.query_one(ActivityIndicator)

        result_card.clear_content()
        source_list.clear_sources()
        cache_badge.clear_badge()
        activity.set_label("Searching...")

        event_type = ""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", f"{self._api_base}/search", json={"query": query}) as resp:
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
        except Exception as e:
            activity.stop()
            result_card.set_content(f"**Error:** {e}")

    async def _handle_command(self, command: str) -> None:
        result_card = self.query_one(ResultCard)
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/clear":
            result_card.clear_content()
            self.query_one(SourceList).clear_sources()
            self.query_one(CacheBadge).clear_badge()
        elif cmd == "/history":
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self._api_base}/history", params={"limit": 20})
                    history = resp.json()
                lines = ["## Recent Searches\n"]
                for h in history:
                    badge = "\u25cf" if h.get("cache_hit") else "\u25cb"
                    lines.append(f"- {badge} **{h['query']}** ({h.get('response_time_ms', 0)}ms)")
                result_card.set_content("\n".join(lines))
            except Exception as e:
                result_card.set_content(f"**Error:** {e}")
        elif cmd == "/stats":
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self._api_base}/stats")
                    stats = resp.json()
                text = (
                    f"## Pythia Stats\n\n"
                    f"- **Total searches:** {stats.get('total_searches', 0)}\n"
                    f"- **Cache hits:** {stats.get('cache_hits', 0)}\n"
                    f"- **Cache hit rate:** {stats.get('cache_hit_rate', 0)}%\n"
                    f"- **Avg response:** {stats.get('avg_response_ms', 0)}ms\n"
                    f"- **Active days:** {stats.get('active_days', 0)}\n"
                )
                result_card.set_content(text)
            except Exception as e:
                result_card.set_content(f"**Error:** {e}")
        elif cmd == "/model":
            if len(parts) > 1:
                new_model = parts[1].strip()
                self.config.ollama.model = new_model
                self.query_one(PythiaStatusBar).update_status(model=new_model)
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
                "- `/history` — Show recent searches\n"
                "- `/stats` — Cache hit rate, total searches, avg response time\n"
                "- `/model <name>` — Switch Ollama model\n"
                "- `/cache clear` — Purge Oracle cache\n"
                "- `/clear` — Clear screen\n"
                "- `/help` — Show this help\n"
            )
        else:
            result_card.set_content(f"Unknown command: `{cmd}`. Type `/help` for available commands.")
