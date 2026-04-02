"""Tests for follow-up suggestion generation and conversation history."""
import pytest
from unittest.mock import AsyncMock

from pythia.server.search import SearchOrchestrator, EventType
from pythia.server.searxng import SearchResult
from pythia.server.oracle_cache import CacheEntry


@pytest.mark.asyncio
async def test_suggestions_emitted_on_cache_miss():
    """Search should emit SUGGESTIONS event after DONE on cache miss."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(
        return_value=["What is DPO?", "How does PPO work?", "RLHF vs DPO"]
    )

    async def fake_stream(system, user, model=None):
        yield "RLHF is great [1]."

    mock_ollama.generate_stream = fake_stream
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1]"))
    mock_cache.store = AsyncMock()
    mock_cache.record_search = AsyncMock()
    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(return_value=[
        SearchResult(index=1, title="RLHF", url="https://t.com", snippet="RLHF explained"),
    ])

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("What is RLHF?"):
        events.append(event)

    suggestions_events = [e for e in events if e.event_type == EventType.SUGGESTIONS]
    assert len(suggestions_events) == 1
    assert suggestions_events[0].data["suggestions"] == [
        "What is DPO?", "How does PPO work?", "RLHF vs DPO"
    ]


@pytest.mark.asyncio
async def test_suggestions_emitted_on_cache_hit():
    """Suggestions should also work on cache hits."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(
        return_value=["Follow-up 1", "Follow-up 2"]
    )

    cached = CacheEntry(
        query="test", answer="cached answer [1].", sources=[
            {"index": 1, "title": "T", "url": "https://t.com", "snippet": "s"}
        ],
        model_used="test", similarity=0.95,
    )
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(cached, "[0.1]"))
    mock_cache.record_search = AsyncMock()
    mock_searxng = AsyncMock()

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("test"):
        events.append(event)

    suggestions_events = [e for e in events if e.event_type == EventType.SUGGESTIONS]
    assert len(suggestions_events) == 1


@pytest.mark.asyncio
async def test_no_suggestions_when_empty():
    """No SUGGESTIONS event should be emitted when LLM returns empty list."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    async def fake_stream(system, user, model=None):
        yield "answer"

    mock_ollama.generate_stream = fake_stream
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1]"))
    mock_cache.store = AsyncMock()
    mock_cache.record_search = AsyncMock()
    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(return_value=[
        SearchResult(index=1, title="T", url="https://t.com", snippet="s"),
    ])

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("test"):
        events.append(event)

    suggestions_events = [e for e in events if e.event_type == EventType.SUGGESTIONS]
    assert len(suggestions_events) == 0


@pytest.mark.asyncio
async def test_conversation_history_passed_to_prompt():
    """Conversation history should be passed through to prompt building."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    captured_args = {}

    async def fake_stream(system, user, model=None):
        captured_args["system"] = system
        captured_args["user"] = user
        yield "answer"

    mock_ollama.generate_stream = fake_stream
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1]"))
    mock_cache.store = AsyncMock()
    mock_cache.record_search = AsyncMock()
    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(return_value=[
        SearchResult(index=1, title="T", url="https://t.com", snippet="s"),
    ])

    history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("Tell me more", conversation_history=history):
        events.append(event)

    # The user prompt should contain the conversation history
    assert "Conversation History" in captured_args["user"]
    assert "What is Python?" in captured_args["user"]


@pytest.mark.asyncio
async def test_grounding_event_emitted():
    """GROUNDING event should be emitted on both cache hit and miss."""
    mock_ollama = AsyncMock()
    mock_ollama.model = "test"
    mock_ollama.generate_suggestions = AsyncMock(return_value=[])

    async def fake_stream(system, user, model=None):
        yield "Python is popular [1]. It supports OOP [2]."

    mock_ollama.generate_stream = fake_stream
    mock_cache = AsyncMock()
    mock_cache.lookup = AsyncMock(return_value=(None, "[0.1]"))
    mock_cache.store = AsyncMock()
    mock_cache.record_search = AsyncMock()
    mock_searxng = AsyncMock()
    mock_searxng.search = AsyncMock(return_value=[
        SearchResult(index=1, title="Python", url="https://t.com", snippet="Python is a popular language"),
        SearchResult(index=2, title="OOP", url="https://t2.com", snippet="Object oriented programming in Python"),
    ])

    orch = SearchOrchestrator(ollama=mock_ollama, cache=mock_cache, searxng=mock_searxng)
    events = []
    async for event in orch.search("test"):
        events.append(event)

    grounding_events = [e for e in events if e.event_type == EventType.GROUNDING]
    assert len(grounding_events) == 1
    g = grounding_events[0].data
    assert "score" in g
    assert "label" in g
    assert "total_claims" in g
    assert "grounded_claims" in g
