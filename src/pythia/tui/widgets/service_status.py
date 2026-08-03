"""Service status indicator — shows API, Oracle, and SearXNG status."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from pythia.services import ServiceInfo, ServiceStatus
from pythia.tui import colors


class ServiceStatusIndicator(Static):
    """Shows real-time status of Pythia services."""

    DEFAULT_CSS = """
    ServiceStatusIndicator {
        height: auto;
        max-height: 2;
        dock: bottom;
        background: $panel;
        color: $text-muted;
        padding: 0 2;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._services: dict[str, ServiceInfo] = {}

    def update_services(self, services: dict[str, ServiceInfo]) -> None:
        """Update status from service manager."""
        self._services = services
        self._rebuild()

    def _rebuild(self) -> None:
        """Rebuild the status display."""
        status_text = Text()
        status_text.append(" ", style="default")

        # API Server status
        api_info = self._services.get("api")
        if api_info:
            status_text.append("API: ", style=f"{colors.DIM}")
            dot, style = self._get_dot_style(api_info.status)
            status_text.append(f"{dot} ", style=style)
            status_text.append(f"{api_info.message} ", style=f"{colors.MUTED}")
        else:
            status_text.append("API: ", style=f"{colors.DIM}")
            status_text.append("○ ", style=f"{colors.DIM}")
            status_text.append("Initializing ", style=f"{colors.MUTED}")

        status_text.append(" │ ", style=f"{colors.DIM}")

        # Oracle status
        oracle_info = self._services.get("oracle")
        if oracle_info:
            status_text.append("Oracle: ", style=f"{colors.DIM}")
            dot, style = self._get_dot_style(oracle_info.status)
            status_text.append(f"{dot} ", style=style)
            status_text.append(f"{oracle_info.message} ", style=f"{colors.MUTED}")
        else:
            status_text.append("Oracle: ", style=f"{colors.DIM}")
            status_text.append("○ ", style=f"{colors.DIM}")
            status_text.append("Starting ", style=f"{colors.MUTED}")

        status_text.append(" │ ", style=f"{colors.DIM}")

        # SearXNG status
        searxng_info = self._services.get("searxng")
        if searxng_info:
            status_text.append("SearXNG: ", style=f"{colors.DIM}")
            dot, style = self._get_dot_style(searxng_info.status)
            status_text.append(f"{dot} ", style=style)
            status_text.append(f"{searxng_info.message}", style=f"{colors.MUTED}")
        else:
            status_text.append("SearXNG: ", style=f"{colors.DIM}")
            status_text.append("○ ", style=f"{colors.DIM}")
            status_text.append("Starting", style=f"{colors.MUTED}")

        self.update(status_text)

    def _get_dot_style(self, status: ServiceStatus) -> tuple[str, str]:
        """Get dot character and style for status."""
        if status == ServiceStatus.RUNNING:
            return "●", f"{colors.SUCCESS}"  # Green
        if status == ServiceStatus.STARTING:
            return "◐", f"{colors.WARNING}"  # Orange
        if status == ServiceStatus.ERROR:
            return "●", f"{colors.ERROR}"  # Red
        return "○", f"{colors.DIM}"  # Gray
