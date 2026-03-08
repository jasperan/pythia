"""Tests for Ollama client."""
import pytest
from pythia.server.ollama import OllamaClient, build_search_prompt
from pythia.server.searxng import SearchResult


def test_build_search_prompt():
    results = [
        SearchResult(index=1, title="RLHF Paper", url="https://arxiv.org/rlhf", snippet="RLHF aligns LLMs."),
        SearchResult(index=2, title="HF Blog", url="https://hf.co/rlhf", snippet="Step by step RLHF guide."),
    ]
    system, user = build_search_prompt("How does RLHF work?", results)
    assert "Pythia" in system
    assert "[1]" in user
    assert "[2]" in user
    assert "RLHF Paper" in user
    assert "How does RLHF work?" in user


def test_build_search_prompt_empty_results():
    system, user = build_search_prompt("test query", [])
    assert "test query" in user
