"""Tests for TUI theme cycling."""
import pytest
from pythia.tui.app import PythiaApp, AVAILABLE_THEMES
from pythia.config import PythiaConfig


def test_available_themes_list():
    assert "dark" in AVAILABLE_THEMES
    assert "light" in AVAILABLE_THEMES
    assert "catppuccin-mocha" in AVAILABLE_THEMES
    assert "nord" in AVAILABLE_THEMES


def test_theme_cycle_wraps():
    config = PythiaConfig()
    app = PythiaApp(config, auto_start=False)
    assert app._current_theme == "dark"
    app._cycle_theme()
    assert app._current_theme == "light"
    app._cycle_theme()
    assert app._current_theme == "catppuccin-mocha"
    app._cycle_theme()
    assert app._current_theme == "nord"
    app._cycle_theme()
    assert app._current_theme == "dark"
