from __future__ import annotations

import re
from dataclasses import dataclass, field


_CITATION_RE = re.compile(r"\[(\d+)\]")


def canonicalize_claim(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extract_citation_refs(text: str) -> list[int]:
    return [int(m) for m in _CITATION_RE.findall(text)]


@dataclass
class NormalizedSource:
    source_index: int
    url: str
    title: str
    snippet: str
    content_text: str
    sub_query: str
    searxng_rank: int
    domain_name: str = ""
    rerank_score: float = 0.0
    trust_score: float = 0.0
    freshness_score: float = 0.0


@dataclass
class ExtractedClaim:
    claim_order: int
    claim_text: str
    canonical_text: str
    section_name: str
    claim_kind: str
    source_refs: list[int] = field(default_factory=list)
    strength_hint: str = ""


@dataclass
class ClaimEvidence:
    source_index: int
    source_url: str
    evidence_role: str
    excerpt_text: str
    support_score: float
    citation_label: str = ""
    start_char: int | None = None
    end_char: int | None = None
