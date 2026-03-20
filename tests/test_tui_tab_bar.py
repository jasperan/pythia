"""Tests for TUI tab bar widget."""
from pythia.tui.widgets.tab_bar import TabBar, TabDef


def test_tab_bar_creation():
    tabs = [
        TabDef(key="1", label="Search", screen_name="search"),
        TabDef(key="2", label="Research", screen_name="research"),
    ]
    bar = TabBar(tabs)
    assert bar._tabs == tabs
    assert bar._active == "search"


def test_tab_bar_set_active():
    tabs = [
        TabDef(key="1", label="Search", screen_name="search"),
        TabDef(key="2", label="Research", screen_name="research"),
    ]
    bar = TabBar(tabs)
    bar.set_active("research")
    assert bar._active == "research"


def test_tab_bar_ignores_invalid():
    tabs = [TabDef(key="1", label="Search", screen_name="search")]
    bar = TabBar(tabs)
    bar.set_active("nonexistent")
    assert bar._active == "search"
