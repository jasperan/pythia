"""Tests for answer grounding verification."""
from pythia.server.grounding import (
    GroundingResult,
    GroundedClaim,
    _extract_claims,
    _word_overlap,
    verify_grounding,
)


def test_extract_claims_basic():
    text = "LLMs use RLHF for alignment [1]. DPO is an alternative [2]."
    claims = _extract_claims(text)
    assert len(claims) == 2
    assert claims[0].cited_sources == [1]
    assert claims[1].cited_sources == [2]


def test_extract_claims_multiple_citations():
    text = "Transformers power modern NLP [1][3]. They use attention mechanisms [2]."
    claims = _extract_claims(text)
    assert len(claims) == 2
    assert set(claims[0].cited_sources) == {1, 3}


def test_extract_claims_no_citations():
    text = "This has no citations at all."
    claims = _extract_claims(text)
    assert len(claims) == 0


def test_extract_claims_skips_short_fragments():
    text = "Yes [1]. This is a longer claim about neural networks [2]."
    claims = _extract_claims(text)
    # "Yes" is too short (< 10 chars after stripping citation)
    assert len(claims) == 1
    assert claims[0].cited_sources == [2]


def test_word_overlap_perfect():
    overlap = _word_overlap("neural networks training", "neural networks training data")
    assert overlap > 0.8


def test_word_overlap_zero():
    overlap = _word_overlap("quantum computing", "banana recipes cooking")
    assert overlap == 0.0


def test_word_overlap_partial():
    overlap = _word_overlap("machine learning algorithms", "machine learning models deep")
    assert 0.2 < overlap < 0.8


def test_word_overlap_ignores_stopwords():
    # "the", "is", "a" are stop words — should be excluded
    overlap = _word_overlap("the cat is on the mat", "a dog is on a rug")
    # Only non-stop words matter: {cat, mat} vs {dog, rug} = 0 overlap
    assert overlap == 0.0


def test_verify_grounding_well_grounded():
    answer = "Python is a popular programming language [1]. It supports object-oriented programming [2]."
    sources = [
        {"index": 1, "title": "Python", "snippet": "Python is a popular programming language used widely"},
        {"index": 2, "title": "OOP", "snippet": "Python supports object-oriented programming paradigm"},
    ]
    result = verify_grounding(answer, sources)
    assert result.score >= 0.5
    assert result.total_claims == 2
    assert result.grounded_claims >= 1
    assert result.label in ("well-grounded", "partially-grounded")


def test_verify_grounding_weakly_grounded():
    answer = "Quantum computing will revolutionize encryption [1]. Qubits enable parallel computation [2]."
    sources = [
        {"index": 1, "title": "Weather", "snippet": "Today's weather forecast shows sunny skies"},
        {"index": 2, "title": "Cooking", "snippet": "Best recipes for chocolate cake baking"},
    ]
    result = verify_grounding(answer, sources)
    assert result.score < 0.5
    assert result.label == "weakly-grounded"


def test_verify_grounding_no_claims():
    answer = "I don't have enough information to answer this question."
    sources = [{"index": 1, "title": "T", "snippet": "some snippet"}]
    result = verify_grounding(answer, sources)
    assert result.score == 1.0
    assert result.label == "no-claims"


def test_verify_grounding_empty_sources():
    answer = "Something with a citation [1]."
    sources = []
    result = verify_grounding(answer, sources)
    assert result.total_claims == 1
    assert result.grounded_claims == 0


def test_verify_grounding_missing_source_index():
    answer = "Referenced source that does not exist in our list [99]."
    sources = [{"index": 1, "title": "T", "snippet": "actual content here"}]
    result = verify_grounding(answer, sources)
    assert result.grounded_claims == 0


def test_grounding_result_dataclass():
    r = GroundingResult(score=0.75, total_claims=4, grounded_claims=3, label="well-grounded")
    assert r.score == 0.75
    assert r.claims == []


def test_grounded_claim_dataclass():
    c = GroundedClaim(text="test claim", cited_sources=[1, 2])
    assert c.grounded is False
    assert c.best_overlap == 0.0
