"""Tests for history list widget."""
from pythia.tui.widgets.history_list import HistoryList, HistoryEntry, HistoryFilter


def test_history_list_creation():
    hl = HistoryList()
    assert hl._entries == []
    assert hl._filter == HistoryFilter.ALL


def test_load_entries():
    entries = [
        HistoryEntry(query="test query", cache_hit=True, response_time_ms=23, model="qwen3.5:9b", is_research=False),
        HistoryEntry(query="[research] deep topic", cache_hit=False, response_time_ms=8420, model="qwen3.5:9b", is_research=True),
    ]
    hl = HistoryList()
    hl.load_entries(entries)
    assert len(hl._entries) == 2


def test_filter_cache_hits():
    entries = [
        HistoryEntry(query="q1", cache_hit=True, response_time_ms=23, model="m", is_research=False),
        HistoryEntry(query="q2", cache_hit=False, response_time_ms=2000, model="m", is_research=False),
    ]
    hl = HistoryList()
    hl.load_entries(entries)
    hl.set_filter(HistoryFilter.CACHE_HITS)
    visible = hl._get_visible()
    assert len(visible) == 1
    assert visible[0].query == "q1"


def test_filter_research():
    entries = [
        HistoryEntry(query="q1", cache_hit=False, response_time_ms=100, model="m", is_research=False),
        HistoryEntry(query="[research] deep", cache_hit=False, response_time_ms=8000, model="m", is_research=True),
    ]
    hl = HistoryList()
    hl.load_entries(entries)
    hl.set_filter(HistoryFilter.RESEARCH)
    visible = hl._get_visible()
    assert len(visible) == 1
    assert visible[0].is_research


def test_text_filter():
    entries = [
        HistoryEntry(query="vector databases", cache_hit=False, response_time_ms=100, model="m", is_research=False),
        HistoryEntry(query="quantum computing", cache_hit=False, response_time_ms=100, model="m", is_research=False),
    ]
    hl = HistoryList()
    hl.load_entries(entries)
    hl.set_text_filter("vector")
    visible = hl._get_visible()
    assert len(visible) == 1
    assert "vector" in visible[0].query
