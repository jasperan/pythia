"""CLI contract tests around config discovery."""
from __future__ import annotations

from typer.testing import CliRunner

from pythia.cli import app

runner = CliRunner()


def test_query_requires_explicit_config_outside_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHIA_CONFIG", raising=False)

    result = runner.invoke(app, ["query", "What is RLHF?"])

    assert result.exit_code == 2
    assert "Use --config /abs/path/to/pythia.yaml or set PYTHIA_CONFIG." in result.output


def test_search_requires_explicit_config_outside_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHIA_CONFIG", raising=False)

    result = runner.invoke(app, ["search"])

    assert result.exit_code == 2
    assert "Auto-start also needs access to the project Docker assets" in result.output
