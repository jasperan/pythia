"""CLI contract tests around config discovery."""
from __future__ import annotations

from typer.testing import CliRunner

from pythia.cli import app
from pythia.tui import app as tui_app_module

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


def test_search_forwards_resolved_config_path_to_run_tui(monkeypatch, tmp_path):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text("server:\n  port: 9000\n")
    captured = {}

    def fake_run_tui(config, auto_start=True, host=None, port=None, config_path="pythia.yaml"):
        captured["config_path"] = config_path
        captured["auto_start"] = auto_start

    monkeypatch.setattr(tui_app_module, "run_tui", fake_run_tui)

    result = runner.invoke(app, ["search", "--config", str(config_path), "--no-auto-start"])

    assert result.exit_code == 0
    assert captured["config_path"] == str(config_path.resolve())
    assert captured["auto_start"] is False
