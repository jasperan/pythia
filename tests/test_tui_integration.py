"""End-to-end integration tests for Pythia TUI using Textual's test harness."""
from __future__ import annotations

from pathlib import Path
import pytest
from unittest.mock import AsyncMock
import httpx
from textual.widgets import Input

from pythia.config import PythiaConfig
from pythia.tui.app import PythiaApp, AVAILABLE_THEMES
from pythia.tui import app as app_module
from pythia.tui.screens import search as search_screen_module
from pythia.tui.screens import research as research_screen_module
from pythia.tui.screens.search import SearchScreen
from pythia.tui.screens.research import ResearchScreen


@pytest.fixture
def config():
    return PythiaConfig()


@pytest.fixture
def app(config):
    return PythiaApp(config, auto_start=False)


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _make_streaming_client(recorded: dict[str, object], lines: list[str]):
    class _RecordingAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, json):
            recorded["method"] = method
            recorded["url"] = url
            recorded["json"] = json
            return _FakeStreamResponse(lines)

    return _RecordingAsyncClient


# --- App lifecycle & screen installation ---


@pytest.mark.asyncio
async def test_app_mounts_and_shows_search_screen(app):
    async with app.run_test():
        assert app._current_screen_name == "search"
        # Search screen should have the logo and search input
        assert app.screen.query("LogoBanner")
        assert app.screen.query("SearchInput")


@pytest.mark.asyncio
async def test_all_four_screens_installed(app):
    async with app.run_test():
        assert app.is_screen_installed("search")
        assert app.is_screen_installed("research")
        assert app.is_screen_installed("history")
        assert app.is_screen_installed("dashboard")


# --- Screen switching via number keys ---


@pytest.mark.asyncio
async def test_switch_to_research_via_key(app):
    async with app.run_test() as pilot:
        # Focus must NOT be on an Input for number keys to work
        app.set_focus(None)
        await pilot.press("2")
        assert app._current_screen_name == "research"


@pytest.mark.asyncio
async def test_switch_to_history_via_key(app):
    async with app.run_test() as pilot:
        app.set_focus(None)
        await pilot.press("3")
        assert app._current_screen_name == "history"


@pytest.mark.asyncio
async def test_switch_to_dashboard_via_key(app):
    async with app.run_test() as pilot:
        app.set_focus(None)
        await pilot.press("4")
        assert app._current_screen_name == "dashboard"


@pytest.mark.asyncio
async def test_switch_back_to_search_via_key(app):
    async with app.run_test() as pilot:
        app.set_focus(None)
        await pilot.press("3")
        assert app._current_screen_name == "history"
        # History screen may auto-focus its filter Input; unfocus before pressing 1
        app.set_focus(None)
        await pilot.press("1")
        assert app._current_screen_name == "search"


@pytest.mark.asyncio
async def test_number_keys_ignored_when_input_focused(app):
    async with app.run_test() as pilot:
        # Focus the search input
        inp = app.screen.query_one("#search-input", Input)
        app.set_focus(inp)
        await pilot.press("2")
        # Should still be on search (key went to input, not screen switch)
        assert app._current_screen_name == "search"


# --- Screen switching via action methods ---


@pytest.mark.asyncio
async def test_switch_via_action_methods(app):
    async with app.run_test():
        app.action_switch_to_research()
        assert app._current_screen_name == "research"
        app.action_switch_to_dashboard()
        assert app._current_screen_name == "dashboard"
        app.action_switch_to_history()
        assert app._current_screen_name == "history"
        app.action_switch_to_search()
        assert app._current_screen_name == "search"


@pytest.mark.asyncio
async def test_switch_to_same_screen_is_noop(app):
    async with app.run_test():
        app.action_switch_to_search()
        assert app._current_screen_name == "search"


@pytest.mark.asyncio
async def test_app_passes_host_and_port_overrides_to_screens(config):
    search_screen = SearchScreen(config, host="127.0.0.1", port=9001)
    research_screen = ResearchScreen(config, host="127.0.0.1", port=9001)

    assert search_screen._api_base == "http://127.0.0.1:9001"
    assert research_screen._api_base == "http://127.0.0.1:9001"


@pytest.mark.asyncio
async def test_auto_start_passes_config_path_to_service_manager(config, monkeypatch):
    captured: dict[str, object] = {}

    class DummyServiceManager:
        def __init__(self, config_path: str, host: str, port: int) -> None:
            captured["config_path"] = config_path
            captured["host"] = host
            captured["port"] = port

        def register_status_callback(self, callback) -> None:
            self._callback = callback

        async def start_all(self) -> None:
            captured["started"] = True

        async def stop_all(self) -> None:
            captured["stopped"] = True

    monkeypatch.setattr(app_module, "ServiceManager", DummyServiceManager)

    app = PythiaApp(
        config,
        auto_start=True,
        host="127.0.0.1",
        port=9001,
        config_path="/tmp/custom.yaml",
    )

    async with app.run_test() as pilot:
        await pilot.pause()

    assert captured["config_path"] == "/tmp/custom.yaml"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9001
    assert captured["started"] is True


@pytest.mark.asyncio
async def test_app_actions_use_host_override(config, monkeypatch, tmp_path):
    calls: list[tuple[str, str]] = []

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class DummyClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None):
            calls.append(("get", url))
            return DummyResponse([])

        async def delete(self, url):
            calls.append(("delete", url))
            return DummyResponse({"deleted": 0})

    monkeypatch.setattr(httpx, "AsyncClient", DummyClient)
    monkeypatch.setattr(Path, "write_text", lambda self, text: len(text))

    app = PythiaApp(config, auto_start=False, host="api.example.com", port=9443)

    async with app.run_test():
        await app.action_export_results()
        await app.action_clear_cache()

    assert calls[0] == ("get", "http://api.example.com:9443/history")
    assert calls[1] == ("delete", "http://api.example.com:9443/cache")


# --- Theme cycling ---


@pytest.mark.asyncio
async def test_theme_cycling_via_action(app):
    async with app.run_test():
        assert app._current_theme == "dark"
        app.action_cycle_theme()
        assert app._current_theme == "light"
        app.action_cycle_theme()
        assert app._current_theme == "catppuccin-mocha"
        app.action_cycle_theme()
        assert app._current_theme == "nord"
        app.action_cycle_theme()
        assert app._current_theme == "dark"


# --- Deep search toggle ---


@pytest.mark.asyncio
async def test_deep_mode_toggle(app):
    async with app.run_test():
        assert app._deep_mode is False
        app.action_toggle_deep()
        assert app._deep_mode is True
        app.action_toggle_deep()
        assert app._deep_mode is False


# --- Search screen composition ---


@pytest.mark.asyncio
async def test_search_screen_has_required_widgets(app):
    async with app.run_test():
        screen = app.screen
        assert screen.query("LogoBanner"), "Missing LogoBanner"
        assert screen.query("#results-area"), "Missing results area"
        assert screen.query("ActivityIndicator"), "Missing ActivityIndicator"
        assert screen.query("SearchInput"), "Missing SearchInput"
        assert screen.query("PythiaStatusBar"), "Missing PythiaStatusBar"
        assert screen.query("ServiceStatusIndicator"), "Missing ServiceStatusIndicator"


@pytest.mark.asyncio
async def test_search_screen_results_area_starts_empty(app):
    async with app.run_test():
        from textual.containers import VerticalScroll
        results = app.screen.query_one("#results-area", VerticalScroll)
        assert len(results.children) == 0


@pytest.mark.asyncio
async def test_search_requests_include_selected_model(app, monkeypatch):
    recorded: dict[str, object] = {}
    client_cls = _make_streaming_client(
        recorded,
        ["event: done", 'data: {"cache_hit": false, "response_time_ms": 1, "sources_count": 0}'],
    )
    monkeypatch.setattr(search_screen_module.httpx, "AsyncClient", client_cls)

    async with app.run_test():
        screen = app.screen
        screen.config.ollama.model = "llama3.3:70b"
        screen._check_health = AsyncMock()
        await screen._run_search("test model")

    assert recorded["json"]["model"] == "llama3.3:70b"


@pytest.mark.asyncio
async def test_search_follow_up_requests_preserve_model_and_history(app, monkeypatch):
    recorded: dict[str, object] = {}
    client_cls = _make_streaming_client(
        recorded,
        ["event: done", 'data: {"cache_hit": false, "response_time_ms": 1, "sources_count": 0}'],
    )
    monkeypatch.setattr(search_screen_module.httpx, "AsyncClient", client_cls)

    async with app.run_test():
        screen = app.screen
        screen.config.ollama.model = "llama3.3:70b"
        screen._check_health = AsyncMock()
        screen._conversation_history = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
        await screen._run_search("follow up")

    assert recorded["json"]["model"] == "llama3.3:70b"
    assert recorded["json"]["conversation_history"] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]


# --- Research screen composition ---


@pytest.mark.asyncio
async def test_research_screen_has_split_pane(app):
    async with app.run_test() as pilot:
        app.set_focus(None)
        await pilot.press("2")
        screen = app.screen
        assert screen.query("#research-split"), "Missing research split pane"
        assert screen.query("#research-tree-pane"), "Missing tree pane"
        assert screen.query("#research-main-pane"), "Missing main pane"
        assert screen.query("ResearchTree"), "Missing ResearchTree"
        assert screen.query("ResearchProgressBar"), "Missing ResearchProgressBar"
        assert screen.query("SearchInput"), "Missing SearchInput"


@pytest.mark.asyncio
async def test_research_requests_include_selected_model(app, monkeypatch):
    recorded: dict[str, object] = {}
    client_cls = _make_streaming_client(
        recorded,
        ["event: done", 'data: {"rounds_used": 1, "total_findings": 0, "total_sources": 0, "elapsed_ms": 1}'],
    )
    monkeypatch.setattr(research_screen_module.httpx, "AsyncClient", client_cls)

    async with app.run_test() as pilot:
        app.action_switch_to_research()
        await pilot.pause()
        screen = app.screen
        screen.config.ollama.model = "llama3.3:70b"
        await screen._run_research("edge ai")

    assert recorded["json"]["model"] == "llama3.3:70b"


# --- History screen composition ---


@pytest.mark.asyncio
async def test_history_screen_has_required_widgets(app):
    async with app.run_test() as pilot:
        app.set_focus(None)
        await pilot.press("3")
        screen = app.screen
        assert screen.query("HistoryList"), "Missing HistoryList"
        assert screen.query("#history-filter"), "Missing filter input"
        assert screen.query("#history-footer"), "Missing footer"


# --- Dashboard screen composition ---


@pytest.mark.asyncio
async def test_dashboard_screen_has_required_widgets(app):
    async with app.run_test() as pilot:
        app.set_focus(None)
        await pilot.press("4")
        screen = app.screen
        assert screen.query("StatsPanel"), "Missing StatsPanel"
        assert screen.query("SparklinePanel"), "Missing SparklinePanel"
        assert screen.query("SettingsPanel"), "Missing SettingsPanel"
        assert screen.query("ActionBar"), "Missing ActionBar"


# --- Pending query handoff ---


@pytest.mark.asyncio
async def test_pending_research_query_attribute(app):
    async with app.run_test():
        app._pending_research_query = "test topic"
        assert app._pending_research_query == "test topic"
        # Clear it like the research screen would
        app._pending_research_query = None
        assert app._pending_research_query is None


@pytest.mark.asyncio
async def test_pending_search_query_attribute(app):
    async with app.run_test():
        app._pending_search_query = "test query"
        assert app._pending_search_query == "test query"


# --- Widget unit tests within app context ---


@pytest.mark.asyncio
async def test_research_tree_in_app_context(app):
    async with app.run_test() as pilot:
        app.set_focus(None)
        await pilot.press("2")
        from pythia.tui.widgets.research_tree import ResearchTree, NodeState
        tree = app.screen.query_one(ResearchTree)
        tree.add_plan(["Sub Q 1", "Sub Q 2", "Sub Q 3"])
        assert len(tree._rounds) == 1
        assert len(tree._rounds[0]["sub_queries"]) == 3
        tree.complete_finding("Sub Q 1", num_sources=5, preview="Found stuff")
        assert tree._rounds[0]["sub_queries"][0]["state"] == NodeState.COMPLETE
        tree.add_gaps(["Follow-up A"], reasoning="Missing info")
        assert len(tree._rounds) == 2


@pytest.mark.asyncio
async def test_research_progress_in_app_context(app):
    async with app.run_test() as pilot:
        app.set_focus(None)
        await pilot.press("2")
        from pythia.tui.widgets.research_progress import ResearchProgressBar
        bar = app.screen.query_one(ResearchProgressBar)
        bar.update_progress(round_num=1, max_rounds=3, findings=2, sources=8, elapsed_ms=1500)
        assert bar._current_round == 1
        assert bar._max_rounds == 3
        assert bar._findings == 2


@pytest.mark.asyncio
async def test_history_list_in_app_context(app):
    async with app.run_test() as pilot:
        app.set_focus(None)
        await pilot.press("3")
        from pythia.tui.widgets.history_list import HistoryList, HistoryEntry
        hl = app.screen.query_one(HistoryList)
        entries = [
            HistoryEntry(query="vector databases", cache_hit=True, response_time_ms=23, model="qwen3.5:9b", is_research=False),
            HistoryEntry(query="quantum computing", cache_hit=False, response_time_ms=2000, model="qwen3.5:9b", is_research=False),
            HistoryEntry(query="[research] AI safety", cache_hit=False, response_time_ms=8000, model="qwen3.5:9b", is_research=True),
        ]
        hl.load_entries(entries)
        assert len(hl._entries) == 3
        # Test selection navigation
        hl.move_selection(1)
        assert hl._selected_index == 1
        selected = hl.get_selected()
        assert selected.query == "quantum computing"
        # Test filtering
        hl.set_text_filter("vector")
        visible = hl._get_visible()
        assert len(visible) == 1


# --- Clear results action ---


@pytest.mark.asyncio
async def test_clear_results_action(app):
    async with app.run_test() as pilot:
        from textual.containers import VerticalScroll
        from pythia.tui.widgets.result_card import ResultCard
        # Mount a result card in the results area
        results = app.screen.query_one("#results-area", VerticalScroll)
        card = ResultCard()
        await results.mount(card)
        card.set_content("test content")
        assert len(results.children) == 1
        # Clear
        app.action_clear_results()
        await pilot.pause()
        assert len(results.children) == 0


# --- CSS theme files exist ---


def test_all_theme_files_exist():
    from pathlib import Path
    themes_dir = Path(__file__).parent.parent / "src" / "pythia" / "tui" / "themes"
    for theme in AVAILABLE_THEMES:
        path = themes_dir / f"{theme}.tcss"
        assert path.exists(), f"Missing theme file: {path}"
        content = path.read_text()
        assert len(content) > 100, f"Theme file too small: {path}"
        # Every theme should define Screen and SearchInput at minimum
        assert "Screen" in content, f"Theme {theme} missing Screen selector"
        assert "SearchInput" in content, f"Theme {theme} missing SearchInput selector"


# --- Command palette provider ---


@pytest.mark.asyncio
async def test_command_palette_registered(app):
    async with app.run_test():
        from pythia.tui.commands import PythiaCommands
        assert PythiaCommands in app.COMMANDS
