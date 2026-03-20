"""Dashboard screen — stats, cache management, settings."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import httpx
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen

from pythia.config import PythiaConfig
from pythia.tui.widgets.action_bar import ActionBar
from pythia.tui.widgets.settings_panel import SettingsPanel
from pythia.tui.widgets.sparkline_panel import SparklinePanel
from pythia.tui.widgets.stats_panel import StatsPanel


class DashboardScreen(Screen):
    DEFAULT_CSS = """
    DashboardScreen { layout: vertical; }
    #dashboard-top { height: auto; min-height: 8; }
    """

    def __init__(self, config: PythiaConfig) -> None:
        super().__init__()
        self.config = config
        self._api_base = f"http://{config.server.host}:{config.server.port}"
        if config.server.host == "0.0.0.0":
            self._api_base = f"http://127.0.0.1:{config.server.port}"
        self._refresh_interval = None

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            with Horizontal(id="dashboard-top"):
                yield StatsPanel()
                yield SparklinePanel()
            yield SettingsPanel(self.config)
            yield ActionBar()

    def on_mount(self) -> None:
        self._refresh_interval = self.set_interval(5.0, self._refresh_data)
        self.call_later(self._refresh_data)
        settings = self.query_one(SettingsPanel)
        self.call_later(settings.load_models)

    def on_unmount(self) -> None:
        if self._refresh_interval:
            self._refresh_interval.stop()

    async def _refresh_data(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                stats_resp = await client.get(f"{self._api_base}/stats")
                stats = stats_resp.json()

                health_resp = await client.get(f"{self._api_base}/health")
                health = health_resp.json()
                stats["cache_size"] = health.get("cache_size", 0)

                self.query_one(StatsPanel).update_stats(stats)

                history_resp = await client.get(f"{self._api_base}/history", params={"limit": 20})
                history = history_resp.json()
                self.query_one(SparklinePanel).update_data(history)

        except Exception:
            pass

    async def on_action_bar_action_requested(self, event: ActionBar.ActionRequested) -> None:
        if event.action == "clear_cache":
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.delete(f"{self._api_base}/cache")
                    data = resp.json()
                    self.notify(f"Cache cleared: {data.get('deleted', 0)} entries", timeout=3)
                    await self._refresh_data()
            except Exception as e:
                self.notify(f"Error: {e}", severity="error", timeout=3)

        elif event.action == "export_history":
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self._api_base}/history", params={"limit": 1000})
                    history = resp.json()

                ts = datetime.now().strftime("%Y-%m-%d")
                path = Path.home() / f"pythia-history-{ts}.md"
                lines = ["# Pythia Search History\n"]
                for h in history:
                    badge = "cache" if h.get("cache_hit") else "web"
                    lines.append(f"- **{h.get('query', '')}** ({badge}, {h.get('response_time_ms', 0)}ms)")
                path.write_text("\n".join(lines))
                self.notify(f"Exported to {path}", timeout=3)
            except Exception as e:
                self.notify(f"Error: {e}", severity="error", timeout=3)

        elif event.action == "refresh":
            await self._refresh_data()
            self.notify("Refreshed", timeout=1)

    def on_settings_panel_setting_changed(self, event: SettingsPanel.SettingChanged) -> None:
        if event.key == "model":
            self.config.ollama.model = event.value
            self.notify(f"Model: {event.value}", timeout=2)
        elif event.key == "deep":
            from pythia.tui.app import PythiaApp
            app = self.app
            if isinstance(app, PythiaApp):
                app._deep_mode = event.value
