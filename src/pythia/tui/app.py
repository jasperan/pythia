"""Textual TUI app for Pythia."""
from __future__ import annotations

import time
from pathlib import Path

from textual.app import App
from textual import work

from pythia.config import PythiaConfig
from pythia.tui.commands import PythiaCommands
from pythia.services import ServiceManager, ServiceInfo, ServiceStatus
from pythia.tui.screens.search import SearchScreen
from pythia.tui.screens.research import ResearchScreen
from pythia.tui.screens.history import HistoryScreen
from pythia.tui.screens.dashboard import DashboardScreen

AVAILABLE_THEMES = ["dark", "light", "catppuccin-mocha", "nord"]
_THEMES_DIR = Path(__file__).parent / "themes"


class PythiaApp(App):
    TITLE = "Pythia"
    CSS_PATH = "themes/dark.tcss"
    COMMANDS = {PythiaCommands}

    _CTRL_C_WINDOW = 1.0

    BINDINGS = [
        ("ctrl+l", "clear_results", "Clear"),
        ("ctrl+e", "export_results", "Export"),
        ("ctrl+t", "cycle_theme", "Theme"),
        ("ctrl+d", "toggle_deep", "Deep Search"),
    ]

    def __init__(
        self,
        config: PythiaConfig,
        auto_start: bool = True,
        host: str | None = None,
        port: int | None = None,
        config_path: str = "pythia.yaml",
    ) -> None:
        super().__init__()
        self.config = config
        self._last_ctrl_c: float = 0.0
        self._auto_start = auto_start
        self._host = host or config.server.host
        self._port = port or config.server.port
        self._api_base = self._build_api_base(self._host, self._port)
        self._config_path = config_path
        self._service_manager: ServiceManager | None = None
        self._current_theme: str = getattr(getattr(config, "tui", None), "theme", None) or "dark"
        self._deep_mode: bool = False
        self._pending_search_query: str | None = None
        self._pending_research_query: str | None = None
        self._current_screen_name: str = "search"

    @staticmethod
    def _build_api_base(host: str, port: int) -> str:
        """Normalize app-facing API URLs so 0.0.0.0 becomes reachable localhost."""
        api_host = "127.0.0.1" if host == "0.0.0.0" else host
        return f"http://{api_host}:{port}"

    def _cycle_theme(self) -> None:
        """Advance to the next theme in AVAILABLE_THEMES, wrapping around."""
        idx = AVAILABLE_THEMES.index(self._current_theme) if self._current_theme in AVAILABLE_THEMES else 0
        self._current_theme = AVAILABLE_THEMES[(idx + 1) % len(AVAILABLE_THEMES)]
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Load the current theme CSS file by updating CSS_PATH and refreshing."""
        css_path = _THEMES_DIR / f"{self._current_theme}.tcss"
        if not css_path.exists():
            self.notify(f"Theme file not found: {css_path.name}", severity="error", timeout=3)
            return
        try:
            self.CSS_PATH = f"themes/{self._current_theme}.tcss"
            self.refresh_css(animate=False)
        except Exception as e:
            self.notify(f"Theme error: {e}", severity="error", timeout=3)

    def on_mount(self) -> None:
        self.install_screen(SearchScreen(self.config, host=self._host, port=self._port), name="search")
        self.install_screen(ResearchScreen(self.config, host=self._host, port=self._port), name="research")
        self.install_screen(HistoryScreen(self.config, host=self._host, port=self._port), name="history")
        self.install_screen(DashboardScreen(self.config, host=self._host, port=self._port), name="dashboard")
        self.push_screen("search")
        if self._auto_start:
            self._start_services()

    def _switch_to(self, screen_name: str) -> None:
        if screen_name == self._current_screen_name:
            return
        self.switch_screen(screen_name)
        self._current_screen_name = screen_name

    def action_switch_to_search(self) -> None:
        self._switch_to("search")

    def action_switch_to_research(self) -> None:
        self._switch_to("research")

    def action_switch_to_history(self) -> None:
        self._switch_to("history")

    def action_switch_to_dashboard(self) -> None:
        self._switch_to("dashboard")

    def action_clear_results(self) -> None:
        screen = self.screen
        try:
            from textual.containers import VerticalScroll
            results_area = screen.query_one("#results-area", VerticalScroll)
            results_area.remove_children()
        except Exception:
            self.notify("Nothing to clear", timeout=1)

    async def action_export_results(self) -> None:
        from datetime import datetime
        from pathlib import Path
        import httpx
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
            self.notify(f"Export failed: {e}", severity="error", timeout=3)

    async def action_clear_cache(self) -> None:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.delete(f"{self._api_base}/cache")
                data = resp.json()
                self.notify(f"Cache cleared: {data.get('deleted', 0)} entries", timeout=3)
        except Exception as e:
            self.notify(f"Error: {e}", severity="error", timeout=3)

    def action_toggle_deep(self) -> None:
        self._deep_mode = not self._deep_mode
        self.notify(f"Deep search: {'ON' if self._deep_mode else 'OFF'}", timeout=2)

    def action_cycle_theme(self) -> None:
        self._cycle_theme()
        self.notify(f"Theme: {self._current_theme}", timeout=2)

    @work(exclusive=True)
    async def _start_services(self) -> None:
        """Start all Pythia services in background."""
        self._service_manager = ServiceManager(
            config_path=self._config_path,
            host=self._host,
            port=self._port,
        )
        self._service_manager.register_status_callback(self._on_service_status_update)
        await self._service_manager.start_all()

    def _on_service_status_update(self, statuses: dict[str, ServiceInfo]) -> None:
        for name, info in statuses.items():
            if info.status == ServiceStatus.RUNNING:
                self.notify(f"{info.name}: {info.message}", severity="information", timeout=2)
            elif info.status == ServiceStatus.ERROR:
                self.notify(f"{info.name}: {info.message}", severity="error", timeout=3)

    async def on_key(self, event) -> None:
        # Number key screen switching (only when search input isn't focused)
        screen_keys = {"1": "search", "2": "research", "3": "history", "4": "dashboard"}
        if event.key in screen_keys:
            from textual.widgets import Input
            focused = self.focused
            if focused is None or not isinstance(focused, Input):
                self._switch_to(screen_keys[event.key])
                return

        if event.key == "ctrl+c":
            now = time.monotonic()
            if now - self._last_ctrl_c < self._CTRL_C_WINDOW:
                await self._shutdown_services()
                self.exit()
            else:
                self._last_ctrl_c = now
                self.notify("Press Ctrl+C again to quit", severity="warning", timeout=2)

    async def on_unmount(self) -> None:
        await self._shutdown_services()

    async def _shutdown_services(self) -> None:
        if self._service_manager:
            await self._service_manager.stop_all()
            self._service_manager = None

def run_tui(config, auto_start=True, host=None, port=None, config_path="pythia.yaml"):
    app = PythiaApp(config, auto_start=auto_start, host=host, port=port, config_path=config_path)
    app.run()
