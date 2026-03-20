"""Tests for research progress bar."""
from pythia.tui.widgets.research_progress import ResearchProgressBar


def test_progress_creation():
    bar = ResearchProgressBar()
    assert bar._current_round == 0
    assert bar._max_rounds == 0


def test_progress_update():
    bar = ResearchProgressBar()
    bar.update_progress(round_num=2, max_rounds=3, findings=5, sources=12, elapsed_ms=4200)
    assert bar._current_round == 2
    assert bar._max_rounds == 3
    assert bar._findings == 5
    assert bar._sources == 12
