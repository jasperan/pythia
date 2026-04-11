"""Tests for research tree widget."""
from pythia.tui.widgets.research_tree import ResearchTree, NodeState


def test_research_tree_creation():
    tree = ResearchTree()
    assert tree._rounds == []


def test_add_plan():
    tree = ResearchTree()
    tree.add_plan(["What is X?", "How does Y work?"])
    assert len(tree._rounds) == 1
    assert len(tree._rounds[0]["sub_queries"]) == 2
    assert tree._rounds[0]["sub_queries"][0]["state"] == NodeState.PENDING


def test_complete_finding():
    tree = ResearchTree()
    tree.add_plan(["What is X?"])
    tree.complete_finding("What is X?", num_sources=3, preview="Found that X is...")
    sq = tree._rounds[0]["sub_queries"][0]
    assert sq["state"] == NodeState.COMPLETE
    assert sq["num_sources"] == 3


def test_add_gaps_creates_new_round():
    tree = ResearchTree()
    tree.add_plan(["What is X?"])
    tree.add_gaps(["Follow-up A", "Follow-up B"], reasoning="Missing context on A")
    assert len(tree._rounds) == 2
    assert len(tree._rounds[1]["sub_queries"]) == 2


def test_set_recall():
    tree = ResearchTree()
    tree.set_recall([{"sub_query": "prior Q", "similarity": 0.85, "from_query": "old research"}])
    assert tree._recall_count == 1


def test_reset():
    tree = ResearchTree()
    tree.add_plan(["What is X?"])
    tree.reset()
    assert tree._rounds == []
    assert tree._recall_count == 0
