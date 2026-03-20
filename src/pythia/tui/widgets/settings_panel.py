"""Settings panel — model picker, toggles."""
from __future__ import annotations

import httpx
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Label, Select, Switch

from pythia.config import PythiaConfig


class SettingsPanel(Vertical):
    DEFAULT_CSS = """
    SettingsPanel {
        height: auto;
        padding: 1 2;
        border: solid #5f87ff;
    }
    SettingsPanel Label { margin: 0 0 0 1; }
    SettingsPanel Select { margin: 0 0 1 1; width: 40; }
    """

    class SettingChanged(Message):
        def __init__(self, key: str, value) -> None:
            super().__init__()
            self.key = key
            self.value = value

    def __init__(self, config: PythiaConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = config

    def compose(self) -> ComposeResult:
        yield Label("Settings", id="settings-title")
        yield Label("Model:")
        yield Select(
            [(self._config.ollama.model, self._config.ollama.model)],
            value=self._config.ollama.model,
            id="model-select",
        )
        yield Label("Deep search:")
        yield Switch(value=False, id="deep-switch")

    async def load_models(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._config.ollama.base_url}/api/tags")
                data = resp.json()
                models = [(m["name"], m["name"]) for m in data.get("models", [])]
                if models:
                    select = self.query_one("#model-select", Select)
                    select.set_options(models)
        except Exception:
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model-select" and event.value and event.value != Select.BLANK:
            self.post_message(self.SettingChanged("model", event.value))

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "deep-switch":
            self.post_message(self.SettingChanged("deep", event.value))
