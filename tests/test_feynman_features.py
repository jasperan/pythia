"""Tests for new Feynman-inspired features: workspace, provenance, verification, skills, autoresearch."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pythia.provenance import ProvenanceRecord
from pythia.skills import SkillDef, SkillLoader
from pythia.verification import VerificationResult
from pythia.workspace import WorkspaceChangelog, generate_slug


class TestGenerateSlug:
    def test_basic_slug(self):
        assert generate_slug("tradeoffs between RISC-V and ARM for edge AI") == "tradeoffs-risc-v-arm-edge-ai"

    def test_removes_fillers(self):
        slug = generate_slug("what is the best way to learn python")
        assert "the" not in slug
        assert "is" not in slug
        assert "to" not in slug

    def test_max_words(self):
        slug = generate_slug("cloud sandbox pricing comparison tools and services 2024", max_words=3)
        parts = slug.split("-")
        assert len(parts) <= 3

    def test_max_length(self):
        slug = generate_slug("a" * 200)
        assert len(slug) <= 60

    def test_special_characters_removed(self):
        slug = generate_slug("What's new in Python 3.12?!")
        assert "'" not in slug
        assert "?" not in slug
        assert "!" not in slug

    def test_fallback_for_all_fillers(self):
        assert generate_slug("the a is of for to") == "research"


class TestWorkspaceChangelog:
    @pytest.fixture
    def tmp_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    def test_append_creates_file(self, tmp_workspace):
        cl = WorkspaceChangelog(tmp_workspace)
        cl.append_entry("test-topic", "Started research", "Initial query", "in_progress")
        assert (tmp_workspace / "CHANGELOG.md").exists()

    def test_append_content(self, tmp_workspace):
        cl = WorkspaceChangelog(tmp_workspace)
        cl.append_entry("test-topic", "Round 1 complete", "3 findings", "in_progress")
        content = (tmp_workspace / "CHANGELOG.md").read_text()
        assert "test-topic" in content
        assert "Round 1 complete" in content
        assert "in_progress" in content

    def test_multiple_entries(self, tmp_workspace):
        cl = WorkspaceChangelog(tmp_workspace)
        cl.append_entry("slug-a", "First", "", "in_progress")
        cl.append_entry("slug-b", "Second", "", "completed")
        content = (tmp_workspace / "CHANGELOG.md").read_text()
        assert "slug-a" in content
        assert "slug-b" in content

    def test_read_recent_empty(self, tmp_workspace):
        cl = WorkspaceChangelog(tmp_workspace)
        assert cl.read_recent() == ""

    def test_read_relevant(self, tmp_workspace):
        cl = WorkspaceChangelog(tmp_workspace)
        cl.append_entry("my-topic", "Entry 1", "", "in_progress")
        cl.append_entry("other-topic", "Entry 2", "", "completed")
        cl.append_entry("my-topic", "Entry 3", "", "completed")
        relevant = cl.read_relevant("my-topic")
        assert "Entry 1" in relevant
        assert "Entry 3" in relevant
        assert "Entry 2" not in relevant


class TestProvenanceRecord:
    def test_default_values(self):
        p = ProvenanceRecord(topic="test", slug="test-topic")
        assert p.rounds == 0
        assert p.sources_consulted == 0
        assert p.verification_status == "pending"

    def test_to_markdown(self):
        p = ProvenanceRecord(
            topic="test", slug="test-topic", rounds=2,
            sources_consulted=10, sources_accepted=8, sources_rejected=2,
            verification_status="pass", model_used="qwen3.5:9b", elapsed_ms=45000,
        )
        md = p.to_markdown()
        assert "test" in md
        assert "test-topic" in md
        assert "10" in md
        assert "pass" in md
        assert "qwen3.5:9b" in md

    def test_to_dict(self):
        p = ProvenanceRecord(topic="test", slug="test-topic", rounds=1)
        d = p.to_dict()
        assert d["topic"] == "test"
        assert d["slug"] == "test-topic"
        assert d["rounds"] == 1


class TestVerificationResult:
    def test_default(self):
        v = VerificationResult()
        assert v.status == "pending"
        assert not v.has_fatal
        assert not v.has_major

    def test_has_fatal(self):
        v = VerificationResult(issues=[{"severity": "fatal", "type": "unsourced", "claim": "x", "explanation": "y"}])
        assert v.has_fatal

    def test_has_major(self):
        v = VerificationResult(issues=[{"severity": "major", "type": "mismatched", "claim": "x", "explanation": "y"}])
        assert v.has_major

    def test_to_markdown(self):
        v = VerificationResult(
            claims_checked=10, status="pass_with_notes",
            summary="Minor issues found",
            issues=[{"severity": "minor", "type": "overstated", "claim": "X is best", "explanation": "Source says X is good, not best"}],
        )
        md = v.to_markdown()
        assert "pass_with_notes" in md
        assert "**Claims checked:** 10" in md
        assert "overstated" in md


class TestSkillLoader:
    @pytest.fixture
    def tmp_skills_dir(self):
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    def test_default_skills(self):
        loader = SkillLoader()
        names = [s.name for s in loader.list_skills()]
        assert "deep-research" in names
        assert "compare" in names
        assert "lit-review" in names
        assert "quick-answer" in names

    def test_match_by_trigger(self):
        loader = SkillLoader()
        skill = loader.match("Compare RISC-V vs ARM")
        assert skill is not None
        assert skill.name == "compare"

    def test_match_lit_review(self):
        loader = SkillLoader()
        skill = loader.match("Literature review on transformers")
        assert skill is not None
        assert skill.name == "lit-review"

    def test_no_match(self):
        loader = SkillLoader()
        skill = loader.match("hello world")
        assert skill is None

    def test_load_from_yaml(self, tmp_skills_dir):
        skill_file = tmp_skills_dir / "custom.yaml"
        skill_file.write_text(
            "name: custom-skill\n"
            "description: A custom skill\n"
            "triggers: [custom, special]\n"
            "max_rounds: 5\n"
        )
        loader = SkillLoader(tmp_skills_dir)
        skill = loader.get("custom-skill")
        assert skill is not None
        assert skill.max_rounds == 5
        assert "custom" in skill.triggers

    def test_get_by_name(self):
        loader = SkillLoader()
        skill = loader.get("deep-research")
        assert skill is not None
        assert skill.max_rounds == 3
