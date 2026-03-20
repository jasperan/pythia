"""Textual TUI app for Pythia."""
from __future__ import annotations

import time
from pathlib import Path

from textual.app import App
from textual import work

from pythia.config import PythiaConfig
from pythia.services import ServiceManager, ServiceInfo, ServiceStatus
from pythia.tui.screens.search import SearchScreen

AVAILABLE_THEMES = ["dark", "light", "catppuccin-mocha", "nord"]
_THEMES_DIR = Path(__file__).parent / "themes"


class PythiaApp(App):
    TITLE = "Pythia"
    CSS_PATH = "themes/dark.tcss"

    _CTRL_C_WINDOW = 1.0

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

    def _cycle_theme(self) -> None:
        """Advance to the next theme in AVAILABLE_THEMES, wrapping around."""
        idx = AVAILABLE_THEMES.index(self._current_theme) if self._current_theme in AVAILABLE_THEMES else 0
        self._current_theme = AVAILABLE_THEMES[(idx + 1) % len(AVAILABLE_THEMES)]
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Load the current theme CSS file and reparse the stylesheet."""
        css_path = _THEMES_DIR / f"{self._current_theme}.tcss"
        # Update the CSS_PATH so future reparsing picks up the right file
        self.__class__.CSS_PATH = f"themes/{self._current_theme}.tcss"
        try:
            self.stylesheet.read_all([str(css_path)])
            self.refresh_css(animate=False)
        except Exception:
            # App may not be mounted yet (e.g. in tests); state is still updated
            pass

    def on_mount(self) -> None:
        self.push_screen(SearchScreen(self.config))
        if self._auto_start:
            self._start_services()

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
        """Handle service status updates."""
        # Notify user of status changes
        for name, info in statuses.items():
            if info.status == ServiceStatus.RUNNING:
                self.notify(f"{info.name}: {info.message}", severity="information", timeout=2)
            elif info.status == ServiceStatus.ERROR:
                self.notify(f"{info.name}: {info.message}", severity="error", timeout=3)

    async def on_key(self, event) -> None:
        if event.key == "ctrl+c":
            now = time.monotonic()
            if now - self._last_ctrl_c < self._CTRL_C_WINDOW:
                await self._shutdown_services()
                self.exit()
            else:
                self._last_ctrl_c = now
                self.notify("Press Ctrl+C again to quit", severity="warning", timeout=2)

    async def on_unmount(self) -> None:
        """Clean up services when app exits."""
        await self._shutdown_services()

    async def _shutdown_services(self) -> None:
        """Shutdown all services gracefully."""
        if self._service_manager:
            await self._service_manager.stop_all()
            self._service_manager = None


def run_tui(
    config: PythiaConfig,
    auto_start: bool = True,
    host: str | None = None,
    port: int | None = None,
) -> None:
    app = PythiaApp(config, auto_start=auto_start, host=host, port=port)
    app.run()
