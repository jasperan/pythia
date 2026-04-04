"""Pluggable skill definitions for research workflows."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SkillDef:
    name: str = ""
    description: str = ""
    triggers: list[str] = field(default_factory=list)
    system_prompt: str = ""
    user_prompt_template: str = ""
    output_format: str = "markdown"
    requires_scrape: bool = False
    max_rounds: int | None = None
    max_sub_queries: int | None = None
    agents: list[str] = field(default_factory=list)


_DEFAULT_SKILLS: dict[str, SkillDef] = {
    "deep-research": SkillDef(
        name="deep-research",
        description="Multi-round iterative research with gap analysis and synthesis",
        triggers=["deepresearch", "deep research", "thorough investigation", "comprehensive analysis"],
        system_prompt="You are Pythia, an AI research engine. Conduct thorough multi-round research.",
        user_prompt_template="Research question: {query}\n\nConduct a thorough investigation with multiple rounds if needed.",
        output_format="markdown",
        requires_scrape=True,
        max_rounds=3,
        max_sub_queries=5,
        agents=["researcher"],
    ),
    "compare": SkillDef(
        name="compare",
        description="Produce a structured comparison matrix between two or more items",
        triggers=["compare", "vs", "versus", "difference between", "comparison"],
        system_prompt="You are Pythia, an AI comparison engine. Produce structured comparison matrices with clear dimensions, pros/cons, and evidence-backed assessments.",
        user_prompt_template="Compare: {query}\n\nProduce a structured comparison with clear dimensions, evidence for each point, and a summary matrix.",
        output_format="markdown",
        requires_scrape=True,
        max_rounds=2,
        max_sub_queries=4,
        agents=["researcher"],
    ),
    "lit-review": SkillDef(
        name="lit-review",
        description="Literature review with consensus, disagreements, and open questions",
        triggers=["literature review", "lit review", "state of the art", "survey of", "academic landscape"],
        system_prompt="You are Pythia, an AI research engine conducting a literature review. Identify consensus, disagreements, and open questions in the academic and technical literature.",
        user_prompt_template="Literature review topic: {query}\n\nSurvey the literature, identify consensus areas, disagreements, and open questions.",
        output_format="markdown",
        requires_scrape=True,
        max_rounds=3,
        max_sub_queries=5,
        agents=["researcher"],
    ),
    "quick-answer": SkillDef(
        name="quick-answer",
        description="Single-shot search with direct answer — no iterative rounds",
        triggers=["quick", "fast", "brief", "summary of"],
        system_prompt="You are Pythia, an AI search engine. Provide a concise, direct answer based on web search results.",
        user_prompt_template="Question: {query}\n\nProvide a concise answer based search results.",
        output_format="markdown",
        requires_scrape=False,
        max_rounds=1,
        max_sub_queries=3,
        agents=["researcher"],
    ),
}


class SkillLoader:
    def __init__(self, skills_dir: Path | str | None = None):
        self.skills_dir = Path(skills_dir) if skills_dir else None
        self.skills: dict[str, SkillDef] = dict(_DEFAULT_SKILLS)
        if self.skills_dir and self.skills_dir.exists():
            self._load_from_directory()

    def _load_from_directory(self):
        for skill_file in self.skills_dir.glob("*.yaml"):
            try:
                with open(skill_file) as f:
                    data = yaml.safe_load(f)
                if data and "name" in data:
                    skill = SkillDef(
                        name=data["name"],
                        description=data.get("description", ""),
                        triggers=data.get("triggers", []),
                        system_prompt=data.get("system_prompt", ""),
                        user_prompt_template=data.get("user_prompt_template", ""),
                        output_format=data.get("output_format", "markdown"),
                        requires_scrape=data.get("requires_scrape", False),
                        max_rounds=data.get("max_rounds"),
                        max_sub_queries=data.get("max_sub_queries"),
                        agents=data.get("agents", []),
                    )
                    self.skills[skill.name] = skill
            except Exception as e:
                logger.warning(f"Failed to load skill from {skill_file}: {e}")

    def match(self, query: str) -> SkillDef | None:
        query_lower = query.lower()
        for skill in self.skills.values():
            for trigger in skill.triggers:
                if trigger.lower() in query_lower:
                    return skill
        return None

    def get(self, name: str) -> SkillDef | None:
        return self.skills.get(name)

    def list_skills(self) -> list[SkillDef]:
        return list(self.skills.values())
