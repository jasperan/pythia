"""CLI contract tests around config discovery."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from pythia.cli import app
from pythia.cli_runner import _flat_research_events
from pythia.server.research import ResearchEvent, ResearchEventType
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


def test_research_continue_forwards_cli_options(monkeypatch, tmp_path):
    config_path = tmp_path / "pythia.yaml"
    config_path.write_text("research:\n  max_rounds: 3\n")
    captured = {}

    async def fake_continue(config, slug, **kwargs):
        captured["backend"] = config.backend
        captured["slug"] = slug
        captured.update(kwargs)

    monkeypatch.setattr("pythia.cli_runner.run_continue_research", fake_continue)

    result = runner.invoke(
        app,
        [
            "research-continue",
            "risc-v-vs-arm",
            "--config",
            str(config_path),
            "--focus",
            "new benchmarks",
            "--max-rounds",
            "10",
            "--model",
            "llama3",
            "--backend",
            "oci-genai",
            "--stream",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "backend": "oci-genai",
        "slug": "risc-v-vs-arm",
        "focus": "new benchmarks",
        "model_override": "llama3",
        "stream": True,
        "max_rounds": 10,
    }


def test_research_refine_forwards_cli_options(monkeypatch, tmp_path):
    config_path = tmp_path / "pythia.yaml"
    config_path.write_text("research:\n  max_rounds: 3\n")
    captured = {}

    async def fake_refine(config, slug, directive, **kwargs):
        captured["backend"] = config.backend
        captured["slug"] = slug
        captured["directive"] = directive
        captured.update(kwargs)

    monkeypatch.setattr("pythia.cli_runner.run_refine_research", fake_refine)

    result = runner.invoke(
        app,
        [
            "research-refine",
            "risc-v-vs-arm",
            "Focus on power consumption",
            "--config",
            str(config_path),
            "--max-rounds",
            "4",
            "--model",
            "llama3",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "backend": "ollama",
        "slug": "risc-v-vs-arm",
        "directive": "Focus on power consumption",
        "model_override": "llama3",
        "stream": False,
        "max_rounds": 4,
    }


@pytest.mark.asyncio
async def test_flat_research_output_includes_failed_findings(capsys):
    class FakeOllama:
        model = "test-model"

    class FakeAgent:
        ollama = FakeOllama()

    async def events():
        yield ResearchEvent(
            ResearchEventType.DONE,
            {
                "rounds_used": 1,
                "total_findings": 2,
                "total_sources": 1,
                "elapsed_ms": 10,
                "failed_findings": 1,
            },
        )

    await _flat_research_events(FakeAgent(), "query", events(), None)

    payload = json.loads(capsys.readouterr().out)
    assert payload["failed_findings"] == 1
