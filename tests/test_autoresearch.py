"""Tests for the autonomous metric-improvement loop."""
from __future__ import annotations

import json
import re
import sys

import pytest

from pythia.autoresearch import AutoresearchAgent, AutoresearchEventType


class FakePlannerLLM:
    model = "test-model"

    def __init__(self, plan: dict | list[dict]):
        self.plans = plan if isinstance(plan, list) else [plan]
        self.plan_calls = 0

    async def generate(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        model: str | None = None,
    ) -> str:
        if "metric extractor" in system.lower():
            match = re.search(r"score:\s*([-+]?\d+(?:\.\d+)?)", user)
            return json.dumps({"value": float(match.group(1)) if match else None})
        idx = min(self.plan_calls, len(self.plans) - 1)
        self.plan_calls += 1
        return json.dumps(self.plans[idx])


def _benchmark_script(tmp_path) -> str:
    script = tmp_path / "benchmark.py"
    script.write_text(
        "from pathlib import Path\n"
        "text = Path('target.py').read_text()\n"
        "print('score:', 3 if 'return 3' in text else 2 if 'return 2' in text else 1)\n"
    )
    return f"{sys.executable} benchmark.py"


@pytest.mark.asyncio
async def test_autoresearch_applies_scoped_exact_edit_when_metric_improves(tmp_path):
    target = tmp_path / "target.py"
    target.write_text("def value():\n    return 1\n")
    plan = {
        "change_description": "Increase the benchmarked return value.",
        "edits": [
            {
                "file": "target.py",
                "find": "def value():\n    return 1\n",
                "replace": "def value():\n    return 2\n",
            }
        ],
        "confidence": 0.9,
    }
    agent = AutoresearchAgent(FakePlannerLLM(plan), workspace_dir=tmp_path)

    events = [
        event
        async for event in agent.run(
            metric_name="score",
            benchmark_cmd=_benchmark_script(tmp_path),
            files_in_scope=["target.py"],
            max_iterations=1,
            target="make score higher",
        )
    ]

    metric = next(e for e in events if e.event_type == AutoresearchEventType.METRIC)
    assert metric.data["improved"] is True
    assert "return 2" in target.read_text()

    session = (tmp_path / ".autoresearch" / "session.jsonl").read_text()
    assert '"changed_files": ["target.py"]' in session


@pytest.mark.asyncio
async def test_autoresearch_keeps_multiple_improving_iterations(tmp_path):
    target = tmp_path / "target.py"
    target.write_text("def value():\n    return 1\n")
    plans = [
        {
            "change_description": "Improve score from one to two.",
            "edits": [
                {
                    "file": "target.py",
                    "find": "def value():\n    return 1\n",
                    "replace": "def value():\n    return 2\n",
                }
            ],
            "confidence": 0.9,
        },
        {
            "change_description": "Improve score from two to three.",
            "edits": [
                {
                    "file": "target.py",
                    "find": "def value():\n    return 2\n",
                    "replace": "def value():\n    return 3\n",
                }
            ],
            "confidence": 0.8,
        },
    ]
    agent = AutoresearchAgent(FakePlannerLLM(plans), workspace_dir=tmp_path)

    events = [
        event
        async for event in agent.run(
            metric_name="score",
            benchmark_cmd=_benchmark_script(tmp_path),
            files_in_scope=["target.py"],
            max_iterations=2,
            target="make score improve each iteration",
        )
    ]

    metrics = [e for e in events if e.event_type == AutoresearchEventType.METRIC]
    assert [m.data["improved"] for m in metrics] == [True, True]
    assert [m.data["metric_value"] for m in metrics] == [2.0, 3.0]
    assert "return 3" in target.read_text()

    done = next(e for e in events if e.event_type == AutoresearchEventType.DONE)
    assert done.data["best_metric"] == 3.0
    assert done.data["best_iteration"] == 2
    assert done.data["improvement"] == 200.0


@pytest.mark.asyncio
async def test_autoresearch_reverts_exact_edit_when_metric_does_not_improve(tmp_path):
    target = tmp_path / "target.py"
    original = "def value():\n    return 1\n"
    target.write_text(original)
    benchmark = tmp_path / "benchmark.py"
    benchmark.write_text("print('score:', 1)\n")
    plan = {
        "change_description": "A change that does not improve the benchmark.",
        "edits": [
            {
                "file": "target.py",
                "find": original,
                "replace": "def value():\n    return 2\n",
            }
        ],
        "confidence": 0.5,
    }
    agent = AutoresearchAgent(FakePlannerLLM(plan), workspace_dir=tmp_path)

    events = [
        event
        async for event in agent.run(
            metric_name="score",
            benchmark_cmd=f"{sys.executable} benchmark.py",
            files_in_scope=["target.py"],
            max_iterations=1,
        )
    ]

    metric = next(e for e in events if e.event_type == AutoresearchEventType.METRIC)
    assert metric.data["improved"] is False
    assert target.read_text() == original


@pytest.mark.asyncio
async def test_autoresearch_refuses_out_of_scope_edits(tmp_path):
    target = tmp_path / "target.py"
    other = tmp_path / "other.py"
    target.write_text("def value():\n    return 1\n")
    other.write_text("def value():\n    return 1\n")
    plan = {
        "change_description": "Try to edit a file outside the allowed scope.",
        "edits": [
            {
                "file": "other.py",
                "find": "def value():\n    return 1\n",
                "replace": "def value():\n    return 2\n",
            }
        ],
        "confidence": 0.2,
    }
    agent = AutoresearchAgent(FakePlannerLLM(plan), workspace_dir=tmp_path)

    events = [
        event
        async for event in agent.run(
            metric_name="score",
            benchmark_cmd=_benchmark_script(tmp_path),
            files_in_scope=["target.py"],
            max_iterations=1,
        )
    ]

    assert any(e.event_type == AutoresearchEventType.REVERT for e in events)
    assert "return 2" not in other.read_text()


@pytest.mark.asyncio
async def test_autoresearch_requires_explicit_edit_scope(tmp_path):
    target = tmp_path / "target.py"
    original = "def value():\n    return 1\n"
    target.write_text(original)
    plan = {
        "change_description": "Try to edit without an explicit scope.",
        "edits": [
            {
                "file": "target.py",
                "find": original,
                "replace": "def value():\n    return 2\n",
            }
        ],
        "confidence": 0.2,
    }
    agent = AutoresearchAgent(FakePlannerLLM(plan), workspace_dir=tmp_path)

    events = [
        event
        async for event in agent.run(
            metric_name="score",
            benchmark_cmd=_benchmark_script(tmp_path),
            files_in_scope=[],
            max_iterations=1,
        )
    ]

    assert any(e.event_type == AutoresearchEventType.REVERT for e in events)
    assert target.read_text() == original


def test_autoresearch_lower_is_better_reports_positive_improvement(tmp_path):
    agent = AutoresearchAgent(FakePlannerLLM({}), workspace_dir=tmp_path)
    assert agent._improvement_pct(best=5, baseline=10, direction="lower") == 50.0
