"""Tests for CLI runner module."""
import json
from pythia.cli_runner import run_embed_single, run_embed_batch


def test_run_embed_single():
    result = run_embed_single("hello world")
    data = json.loads(result)
    assert data["text"] == "hello world"
    assert data["dimensions"] == 384
    assert data["model"] == "all-MiniLM-L6-v2"
    assert len(data["embedding"]) == 384
    assert all(isinstance(x, float) for x in data["embedding"])


def test_run_embed_batch():
    texts = ["hello", "world"]
    results = run_embed_batch(texts)
    assert len(results) == 2
    for i, line in enumerate(results):
        data = json.loads(line)
        assert data["text"] == texts[i]
        assert data["dimensions"] == 384
