"""Tests for session divider widget."""
from pythia.tui.widgets.session_divider import SessionDivider


def test_session_divider_creation():
    div = SessionDivider(query="test query", timestamp="3:42 PM")
    assert div._query == "test query"
    assert div._timestamp == "3:42 PM"


def test_session_divider_truncates_long_query():
    long_query = "a" * 200
    div = SessionDivider(query=long_query, timestamp="3:42 PM")
    assert len(div._query) <= 63  # 60 + "..."
