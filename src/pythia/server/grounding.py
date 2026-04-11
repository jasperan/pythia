"""Answer grounding — verifies LLM claims against source text."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class GroundedClaim:
    """A single claim extracted from the answer with its grounding status."""
    text: str
    cited_sources: list[int]
    grounded: bool = False


@dataclass
class GroundingResult:
    """Overall grounding assessment for an answer."""
    score: float = 0.0  # 0.0 to 1.0
    total_claims: int = 0
    grounded_claims: int = 0
    claims: list[GroundedClaim] = field(default_factory=list)
    label: str = "unverified"


def _extract_claims(answer: str) -> list[GroundedClaim]:
    """Extract citation-bearing sentences from the answer."""
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    claims = []
    for sent in sentences:
        cited = [int(n) for n in re.findall(r'\[(\d+)\]', sent)]
        if cited:
            # Strip citation markers for comparison
            clean = re.sub(r'\[\d+\]', '', sent).strip()
            if len(clean) > 10:  # Skip trivially short fragments
                claims.append(GroundedClaim(text=clean, cited_sources=cited))
    return claims


def _word_overlap(claim_text: str, source_text: str) -> float:
    """Compute word-level recall: fraction of claim keywords found in source."""
    claim_words = set(claim_text.lower().split())
    source_words = set(source_text.lower().split())
    # Remove stop words for better signal
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
            "for", "of", "and", "or", "but", "not", "with", "by", "from", "as",
            "it", "its", "this", "that", "be", "has", "have", "had", "do", "does"}
    claim_words -= stop
    source_words -= stop
    if not claim_words:
        return 0.0
    intersection = claim_words & source_words
    return len(intersection) / len(claim_words)


def verify_grounding(answer: str, sources: list[dict]) -> GroundingResult:
    """Verify how well the answer's claims are grounded in source text.

    Uses word-overlap between cited claims and their referenced source snippets.
    No LLM call needed — this is fast, deterministic, and adds zero latency.
    """
    claims = _extract_claims(answer)
    if not claims:
        return GroundingResult(score=1.0, label="no-claims")

    # Build source index
    source_map: dict[int, str] = {}
    for s in sources:
        idx = s.get("index", 0)
        snippet = s.get("snippet", "")
        title = s.get("title", "")
        source_map[idx] = f"{title} {snippet}"

    grounded_count = 0
    threshold = 0.25  # 25% keyword overlap = grounded

    for claim in claims:
        best = 0.0
        for src_idx in claim.cited_sources:
            src_text = source_map.get(src_idx, "")
            if src_text:
                overlap = _word_overlap(claim.text, src_text)
                best = max(best, overlap)
        claim.grounded = best >= threshold
        if claim.grounded:
            grounded_count += 1

    score = grounded_count / len(claims)

    if score >= 0.8:
        label = "well-grounded"
    elif score >= 0.5:
        label = "partially-grounded"
    else:
        label = "weakly-grounded"

    return GroundingResult(
        score=round(score, 2),
        total_claims=len(claims),
        grounded_claims=grounded_count,
        claims=claims,
        label=label,
    )
