"""Catppuccin Mocha design tokens — single source of truth for TUI colors.

Canonical palette (see docs/tui-design-tokens.md). Theme .tcss files define
the same tokens as TCSS variables; this module exposes them to Python code
(Rich markup style strings cannot reference TCSS variables).
"""

BG = "#1e1e2e"
SURFACE = "#181825"
ELEVATED = "#313244"
HIGHEST = "#45475a"
TEXT = "#cdd6f4"
SUBTEXT = "#a6adc8"
MUTED = "#6c7086"
DIM = "#585b70"
PRIMARY = "#89b4fa"
SECONDARY = "#cba6f7"
INFO = "#89dceb"
SUCCESS = "#a6e3a1"
WARNING = "#f9e2af"
ERROR = "#f38ba8"
