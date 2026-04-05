"""Tests for deep research agent."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pythia.config import ResearchConfig
from pythia.server.research import ResearchAgent, ResearchEvent, ResearchEventType, Finding
from pythia.server.searxng import SearchResult


def _make_agent(
    ollama_generate_return='{"sub_queries": ["sub q1", "sub q2"]}',
    ollama_stream_tokens=None,
    search_results=None,
    recall_findings=None,
    gap_analysis_return=None,
    gap_analysis_sequence=None,
    completeness_return=None,
    config_overrides=None,
):
    """Build a ResearchAgent with mocked dependencies.

    gap_analysis_sequence: list of JSON strings for successive gap analysis calls.
    If provided, overrides gap_analysis_return.
    """
    mock_ollama = AsyncMock()
    mock_ollama.model = "qwen3.5:9b"

    # Track calls to generate to return different responses
    generate_responses = []
    # First call: decompose query
    generate_responses.append(ollama_generate_return)
    # Second+ calls: summarize findings (enough for multiple rounds)
    for _ in range(10):
        generate_responses.append("Summary of findings from search results [1].")

    # Gap analysis responses — either a sequence or a single repeated value
    if gap_analysis_sequence:
        gap_responses = list(gap_analysis_sequence)
    else:
        gap = gap_analysis_return or '{"sufficient": true, "gaps": [], "reasoning": "All covered."}'
        gap_responses = [gap]

    completeness_resp = completeness_return or '{"status": "COMPLETE", "reasoning": "Report is thorough.", "follow_up_queries": []}'

    call_count = {"n": 0}
    gap_call_count = {"n": 0}

    async def mock_generate(system, user, json_mode=False, model=None):
        if json_mode and "completeness" in system.lower():
            return completeness_resp
        if json_mode and "gaps" in system.lower():
            # Gap analysis call
            idx = min(gap_call_count["n"], len(gap_responses) - 1)
            gap_call_count["n"] += 1
            return gap_responses[idx]
        idx = min(call_count["n"], len(generate_responses) - 1)
        call_count["n"] += 1
        return generate_responses[idx]

    mock_ollama.generate = mock_generate

    # Streaming for report synthesis
    tokens = ollama_stream_tokens or ["# Report\n\n", "This is ", "the research ", "report."]

    async def mock_stream(system, user, model=None):
        for t in tokens:
            yield t

    mock_ollama.generate_stream = mock_stream

    mock_cache = AsyncMock()
    mock_cache.recall_findings = AsyncMock(return_value=recall_findings or [])
    mock_cache.store_research = AsyncMock(return_value="abc123def456")
    mock_cache.store_finding = AsyncMock()
    mock_cache.record_search = AsyncMock()

    mock_searxng = AsyncMock()
    results = search_results or [
        SearchResult(index=1, title="Result 1", url="https://example.com/1", snippet="Snippet 1"),
        SearchResult(index=2, title="Result 2", url="https://example.com/2", snippet="Snippet 2"),
    ]
    mock_searxng.search = AsyncMock(return_value=results)

    cfg = ResearchConfig(**(config_overrides or {"max_rounds": 1, "deep_scrape": False}))

    agent = ResearchAgent(
        ollama=mock_ollama, cache=mock_cache,
        searxng=mock_searxng, config=cfg,
    )
    return agent, mock_ollama, mock_cache, mock_searxng


@pytest.mark.asyncio
async def test_research_basic_flow():
    """Research should decompose, search, summarize, and produce a report."""
    agent, mock_ollama, mock_cache, mock_searxng = _make_agent()

    events = []
    async for event in agent.research("What are the tradeoffs of RISC-V vs ARM?"):
        events.append(event)

    types = [e.event_type for e in events]

    # Should have all key phases
    assert ResearchEventType.PLAN in types
    assert ResearchEventType.ROUND_START in types
    assert ResearchEventType.FINDING in types
    assert ResearchEventType.TOKEN in types
    assert ResearchEventType.DONE in types

    # Check plan event has sub_queries
    plan_event = next(e for e in events if e.event_type == ResearchEventType.PLAN)
    assert len(plan_event.data["sub_queries"]) == 2

    # Check done event has metrics
    done_event = next(e for e in events if e.event_type == ResearchEventType.DONE)
    assert done_event.data["rounds_used"] == 1
    assert done_event.data["total_findings"] > 0
    assert done_event.data["elapsed_ms"] >= 0

    # Should have stored research and findings (batch)
    mock_cache.store_research.assert_called_once()
    mock_cache.store_findings_batch.assert_called_once()


@pytest.mark.asyncio
async def test_research_with_recall():
    """Research should include recalled findings from past sessions."""
    recalled = [
        {
            "sub_query": "ARM power efficiency",
            "summary": "ARM is very power efficient.",
            "sources": [],
            "research_query": "ARM vs x86",
            "similarity": 0.82,
        }
    ]
    agent, _, mock_cache, _ = _make_agent(recall_findings=recalled)

    events = []
    async for event in agent.research("RISC-V vs ARM for edge AI"):
        events.append(event)

    types = [e.event_type for e in events]
    assert ResearchEventType.RECALL in types

    recall_event = next(e for e in events if e.event_type == ResearchEventType.RECALL)
    assert recall_event.data["count"] == 1


@pytest.mark.asyncio
async def test_research_multi_round():
    """Research should iterate when gap analysis says findings are insufficient."""
    gap_round1 = '{"sufficient": false, "gaps": ["What about cost?"], "reasoning": "Cost not covered."}'
    agent, _, mock_cache, mock_searxng = _make_agent(
        gap_analysis_return=gap_round1,
        config_overrides={"max_rounds": 2, "deep_scrape": False},
    )

    events = []
    async for event in agent.research("RISC-V vs ARM comprehensive"):
        events.append(event)

    round_starts = [e for e in events if e.event_type == ResearchEventType.ROUND_START]
    # Should have 2 rounds (gap analysis says insufficient, then max_rounds reached)
    assert len(round_starts) == 2
    assert round_starts[0].data["round"] == 1
    assert round_starts[1].data["round"] == 2


@pytest.mark.asyncio
async def test_research_gap_driven_early_stop():
    """Research should stop early when gap analysis says findings are sufficient."""
    agent, _, _, _ = _make_agent(
        gap_analysis_sequence=[
            '{"sufficient": false, "gaps": ["What about cost?"], "reasoning": "Cost not covered."}',
            '{"sufficient": true, "gaps": [], "reasoning": "All covered now."}',
        ],
        config_overrides={"max_rounds": 5, "deep_scrape": False},
    )

    events = []
    async for event in agent.research("understanding quantum computing implications"):
        events.append(event)

    round_starts = [e for e in events if e.event_type == ResearchEventType.ROUND_START]
    # Round 1: gap says insufficient → round 2. Round 2: gap says sufficient → stop.
    # Should NOT reach rounds 3-5.
    assert len(round_starts) == 2
    assert round_starts[0].data["round"] == 1
    assert round_starts[1].data["round"] == 2

    gap_events = [e for e in events if e.event_type == ResearchEventType.GAP_ANALYSIS]
    assert len(gap_events) == 2
    assert gap_events[1].data["sufficient"] is True

    done = next(e for e in events if e.event_type == ResearchEventType.DONE)
    assert done.data["rounds_used"] == 2


@pytest.mark.asyncio
async def test_research_model_override():
    """Model override should be passed through and restored."""
    agent, mock_ollama, _, _ = _make_agent()

    original_model = mock_ollama.model
    events = []
    async for event in agent.research("test query", model_override="llama3.3:70b"):
        events.append(event)

    # Model should be restored after research
    assert mock_ollama.model == original_model


@pytest.mark.asyncio
async def test_research_decompose_fallback():
    """If decomposition fails, should fallback to sensible defaults."""
    agent, _, _, _ = _make_agent(ollama_generate_return="not valid json at all")

    events = []
    async for event in agent.research("test query"):
        events.append(event)

    plan_event = next(e for e in events if e.event_type == ResearchEventType.PLAN)
    # Fallback should produce at least the original query
    assert len(plan_event.data["sub_queries"]) >= 1


@pytest.mark.asyncio
async def test_research_search_failure_graceful():
    """If SearXNG fails for a sub-query, research should continue with others."""
    agent, _, _, mock_searxng = _make_agent()

    call_count = {"n": 0}
    original_search = mock_searxng.search

    async def failing_search(query):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("SearXNG down")
        return [SearchResult(index=1, title="OK", url="https://ok.com", snippet="Works")]

    mock_searxng.search = failing_search

    events = []
    async for event in agent.research("test query"):
        events.append(event)

    # Should still produce a report despite one failed search
    assert ResearchEventType.DONE in [e.event_type for e in events]


@pytest.mark.asyncio
async def test_research_empty_search_results():
    """Research should handle empty search results gracefully."""
    agent, _, _, mock_searxng = _make_agent(search_results=[])

    events = []
    async for event in agent.research("obscure topic"):
        events.append(event)

    # Should complete without error
    done = next(e for e in events if e.event_type == ResearchEventType.DONE)
    assert done.data["rounds_used"] >= 1


@pytest.mark.asyncio
async def test_research_recall_failure_graceful():
    """If recall fails (e.g. table doesn't exist yet), research should continue."""
    agent, _, mock_cache, _ = _make_agent()
    mock_cache.recall_findings = AsyncMock(side_effect=Exception("ORA-00942: table does not exist"))

    events = []
    async for event in agent.research("test query"):
        events.append(event)

    # Should still complete
    assert ResearchEventType.DONE in [e.event_type for e in events]
    # No recall event since it failed
    assert ResearchEventType.RECALL not in [e.event_type for e in events]


@pytest.mark.asyncio
async def test_research_completeness_check_emits_event():
    """Research with completeness checks should emit COMPLETENESS_CHECK events."""
    agent, mock_ollama, mock_cache, _ = _make_agent(
        config_overrides={"max_rounds": 1, "deep_scrape": False, "max_completeness_checks": 1},
    )

    events = []
    async for event in agent.research("test query"):
        events.append(event)

    types = [e.event_type for e in events]
    assert ResearchEventType.COMPLETENESS_CHECK in types

    cc_event = next(e for e in events if e.event_type == ResearchEventType.COMPLETENESS_CHECK)
    assert cc_event.data["status"] in ("COMPLETE", "INCOMPLETE", "STUCK")


@pytest.mark.asyncio
async def test_research_completeness_triggers_extra_round():
    """When completeness check says INCOMPLETE, research should run follow-up queries."""
    agent, _, mock_cache, mock_searxng = _make_agent(
        completeness_return='{"status": "INCOMPLETE", "reasoning": "Missing cost data.", "follow_up_queries": ["What are the costs?"]}',
        config_overrides={"max_rounds": 1, "deep_scrape": False, "max_completeness_checks": 1},
    )

    events = []
    async for event in agent.research("RISC-V vs ARM"):
        events.append(event)

    types = [e.event_type for e in events]
    round_starts = [e for e in events if e.event_type == ResearchEventType.ROUND_START]

    # Should have initial round + 1 completeness-driven extra round
    assert len(round_starts) == 2

    # Should have a completeness check event with INCOMPLETE status
    cc_events = [e for e in events if e.event_type == ResearchEventType.COMPLETENESS_CHECK]
    assert len(cc_events) == 1
    assert cc_events[0].data["status"] == "INCOMPLETE"


@pytest.mark.asyncio
async def test_research_completeness_stuck_stops():
    """When completeness check says STUCK, research should stop without extra rounds."""
    agent, _, _, _ = _make_agent(
        completeness_return='{"status": "STUCK", "reasoning": "Cannot be answered via web.", "follow_up_queries": []}',
        config_overrides={"max_rounds": 1, "deep_scrape": False, "max_completeness_checks": 2},
    )

    events = []
    async for event in agent.research("test query"):
        events.append(event)

    round_starts = [e for e in events if e.event_type == ResearchEventType.ROUND_START]
    assert len(round_starts) == 1  # No extra rounds

    cc_events = [e for e in events if e.event_type == ResearchEventType.COMPLETENESS_CHECK]
    assert len(cc_events) == 1
    assert cc_events[0].data["status"] == "STUCK"


@pytest.mark.asyncio
async def test_continue_research_loads_prior_and_runs():
    """continue_research should load prior findings and run additional rounds."""
    prior_research = {
        "id": "abc123def456",  # pragma: allowlist secret
        "query": "RISC-V vs ARM",
        "report": "# Prior Report\n\nInitial findings.",
        "sub_queries": ["sub q1", "sub q2"],
        "rounds_used": 1,
        "total_sources": 4,
        "model_used": "qwen3.5:9b",
        "slug": "risc-v-arm",
        "parent_id": None,
    }
    prior_findings = [
        {"sub_query": "sub q1", "summary": "ARM is power efficient.", "sources": [], "round_num": 1},
        {"sub_query": "sub q2", "summary": "RISC-V is open.", "sources": [], "round_num": 1},
    ]

    agent, _, mock_cache, _ = _make_agent(
        config_overrides={"max_rounds": 1, "deep_scrape": False, "max_completeness_checks": 0},
    )
    mock_cache.get_research_by_slug = AsyncMock(return_value=prior_research)
    mock_cache.get_findings_for_research = AsyncMock(return_value=prior_findings)

    events = []
    async for event in agent.continue_research("risc-v-arm"):
        events.append(event)

    types = [e.event_type for e in events]
    assert ResearchEventType.PLAN in types
    assert ResearchEventType.ROUND_START in types
    assert ResearchEventType.DONE in types

    done = next(e for e in events if e.event_type == ResearchEventType.DONE)
    assert done.data["continued_from"] == "risc-v-arm"
    assert done.data["prior_findings_loaded"] == 2


@pytest.mark.asyncio
async def test_continue_research_slug_not_found():
    """continue_research should emit error DONE event when slug doesn't exist."""
    agent, _, mock_cache, _ = _make_agent()
    mock_cache.get_research_by_slug = AsyncMock(return_value=None)

    events = []
    async for event in agent.continue_research("nonexistent-slug"):
        events.append(event)

    done = next(e for e in events if e.event_type == ResearchEventType.DONE)
    assert "error" in done.data
    assert done.data["rounds_used"] == 0


@pytest.mark.asyncio
async def test_refine_research_uses_directive():
    """refine_research should use the directive to generate targeted follow-up queries."""
    prior_research = {
        "id": "abc123def456",  # pragma: allowlist secret
        "query": "RISC-V vs ARM",
        "report": "# Prior Report\n\nInitial findings about architecture differences.",
        "sub_queries": ["sub q1"],
        "rounds_used": 1,
        "total_sources": 2,
        "model_used": "qwen3.5:9b",
        "slug": "risc-v-arm",
        "parent_id": None,
    }
    prior_findings = [
        {"sub_query": "sub q1", "summary": "RISC-V is open-source ISA.", "sources": [], "round_num": 1},
    ]

    agent, _, mock_cache, _ = _make_agent(
        config_overrides={"max_rounds": 1, "deep_scrape": False, "max_completeness_checks": 0},
    )
    mock_cache.get_research_by_slug = AsyncMock(return_value=prior_research)
    mock_cache.get_findings_for_research = AsyncMock(return_value=prior_findings)

    events = []
    async for event in agent.refine_research("risc-v-arm", directive="Focus on power consumption comparison"):
        events.append(event)

    types = [e.event_type for e in events]
    assert ResearchEventType.PLAN in types
    assert ResearchEventType.DONE in types

    done = next(e for e in events if e.event_type == ResearchEventType.DONE)
    assert done.data["refined_from"] == "risc-v-arm"
    assert done.data["directive"] == "Focus on power consumption comparison"


@pytest.mark.asyncio
async def test_refine_research_slug_not_found():
    """refine_research should emit error DONE event when slug doesn't exist."""
    agent, _, mock_cache, _ = _make_agent()
    mock_cache.get_research_by_slug = AsyncMock(return_value=None)

    events = []
    async for event in agent.refine_research("nonexistent", directive="more details"):
        events.append(event)

    done = next(e for e in events if e.event_type == ResearchEventType.DONE)
    assert "error" in done.data
