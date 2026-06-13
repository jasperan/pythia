"""Filesystem path helpers anchored to the Pythia project root.

Centralizes the "find the project root" logic that was previously copy-pasted
as fragile ``Path(__file__).parent.parent.parent.parent`` chains in several
modules. Resolving from a single, correctly-anchored location avoids the bug
where modules at different depths produced different (and sometimes wrong)
roots.
"""

from __future__ import annotations

from pathlib import Path

# This file lives at <root>/src/pythia/paths.py, so the project root is three
# parents up. Computed once at import time.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def project_root() -> Path:
    """Return the Pythia project root directory."""
    return PROJECT_ROOT


def skills_dir() -> Path:
    """Return the directory holding skill definition files (``<root>/skills``)."""
    return PROJECT_ROOT / "skills"


def docker_compose_file() -> Path:
    """Return the path to the project's ``docker-compose.yml``."""
    return PROJECT_ROOT / "docker-compose.yml"
