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
    ) -> None:
        super().__init__()
        self.config = config
        self._last_ctrl_c: float = 0.0
        self._auto_start = auto_start
        self._host = host or config.server.host
        self._port = port or config.server.port
        self._service_manager: ServiceManager | None = None
        self._current_theme: str = getattr(getattr(config, "tui", None), "theme", None) or "dark"
        self._deep_mode: bool = False
        self._pending_search_query: str | None = None
        self._pending_research_query: str | None = None
        self._current_screen_name: str = "search"

    def _cycle_theme(self) -> None:
        """Advance to the next theme in AVAILABLE_THEMES, wrapping around."""
        idx = AVAILABLE_THEMES.index(self._current_theme) if self._current_theme in AVAILABLE_THEMES else 0
        self._current_theme = AVAILABLE_THEMES[(idx + 1) % len(AVAILABLE_THEMES)]
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Load the current theme CSS file and reparse the stylesheet."""
        css_path = _THEMES_DIR / f"{self._current_theme}.tcss"
        self.__class__.CSS_PATH = f"themes/{self._current_theme}.tcss"
        try:
            self.stylesheet.read_all([str(css_path)])
            self.refresh_css(animate=False)
        except Exception:
            pass

    def on_mount(self) -> None:
        self.install_screen(SearchScreen(self.config), name="search")
        self.install_screen(ResearchScreen(self.config), name="research")
        self.install_screen(HistoryScreen(self.config), name="history")
        self.install_screen(DashboardScreen(self.config), name="dashboard")
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
        self.notify("Clear (not yet wired)", timeout=1)

    def action_export_results(self) -> None:
        self.notify("Export (not yet wired)", timeout=1)

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
            config_path="pythia.yaml",
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
            focused = self.focused
            if focused is None or not hasattr(focused, 'value'):
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


def run_tui(config, auto_start=True, host=None, port=None):
    app = PythiaApp(config, auto_start=auto_start, host=host, port=port)
    app.run()
