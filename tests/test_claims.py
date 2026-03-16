from pythia.server.claims import (
    ClaimEvidence,
    ExtractedClaim,
    NormalizedSource,
    canonicalize_claim,
    extract_citation_refs,
)


def test_canonicalize_claim_normalizes_whitespace_and_case():
    assert canonicalize_claim("  ARM Is More Efficient.  ") == "arm is more efficient."


def test_extract_citation_refs_reads_numeric_citations():
    assert extract_citation_refs("ARM used less power [1][3].") == [1, 3]


def test_normalized_source_requires_source_index_and_content():
    source = NormalizedSource(
        source_index=7,
        url="https://example.com/a",
        title="Example",
        snippet="short snippet",
        content_text="full scraped content",
        sub_query="power efficiency",
        searxng_rank=1,
    )
    claim = ExtractedClaim(
        claim_order=1,
        claim_text="ARM used less power in the benchmark.",
        canonical_text="arm used less power in the benchmark.",
        section_name="Executive Summary",
        claim_kind="fact",
        source_refs=[7],
    )
    evidence = ClaimEvidence(
        source_index=7,
        source_url="https://example.com/a",
        evidence_role="supports",
        excerpt_text="ARM used less power than x86 in the benchmark.",
        support_score=0.91,
        citation_label="[7]",
    )
    assert source.source_index == 7
    assert claim.source_refs == [7]
    assert evidence.citation_label == "[7]"
