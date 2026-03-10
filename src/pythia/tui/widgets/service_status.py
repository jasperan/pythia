"""Service status indicator — shows API, Oracle, and SearXNG status."""
from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from pythia.services import ServiceInfo, ServiceStatus


class ServiceStatusIndicator(Static):
    """Shows real-time status of Pythia services."""

    DEFAULT_CSS = """
    ServiceStatusIndicator {
        height: auto;
        max-height: 2;
        dock: bottom;
        background: #2a2a3a;
        color: #808080;
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
            status_text.append("API: ", style="#666666")
            dot, style = self._get_dot_style(api_info.status)
            status_text.append(f"{dot} ", style=style)
            status_text.append(f"{api_info.message} ", style="#808080")
        else:
            status_text.append("API: ", style="#666666")
            status_text.append("○ ", style="#666666")
            status_text.append("Initializing ", style="#808080")
        
        status_text.append(" │ ", style="#666666")
        
        # Oracle status
        oracle_info = self._services.get("oracle")
        if oracle_info:
            status_text.append("Oracle: ", style="#666666")
            dot, style = self._get_dot_style(oracle_info.status)
            status_text.append(f"{dot} ", style=style)
            status_text.append(f"{oracle_info.message} ", style="#808080")
        else:
            status_text.append("Oracle: ", style="#666666")
            status_text.append("○ ", style="#666666")
            status_text.append("Starting ", style="#808080")
        
        status_text.append(" │ ", style="#666666")
        
        # SearXNG status
        searxng_info = self._services.get("searxng")
        if searxng_info:
            status_text.append("SearXNG: ", style="#666666")
            dot, style = self._get_dot_style(searxng_info.status)
            status_text.append(f"{dot} ", style=style)
            status_text.append(f"{searxng_info.message}", style="#808080")
        else:
            status_text.append("SearXNG: ", style="#666666")
            status_text.append("○ ", style="#666666")
            status_text.append("Starting", style="#808080")
        
        self.update(status_text)

    def _get_dot_style(self, status: ServiceStatus) -> tuple[str, str]:
        """Get dot character and style for status."""
        if status == ServiceStatus.RUNNING:
            return "●", "#b5bd68"  # Green
        elif status == ServiceStatus.STARTING:
            return "◐", "#ffab40"  # Orange
        elif status == ServiceStatus.ERROR:
            return "●", "#cc6666"  # Red
        else:
            return "○", "#666666"  # Gray
