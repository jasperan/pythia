"""Tests for embedding module."""
from pythia.embeddings import generate_embedding, generate_embedding_list


def test_generate_embedding_returns_json_string():
    result = generate_embedding("hello world")
    assert isinstance(result, str)
    assert result.startswith("[")
    assert result.endswith("]")


def test_generate_embedding_list_returns_floats():
    result = generate_embedding_list("hello world")
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(x, float) for x in result)


def test_generate_embedding_deterministic():
    a = generate_embedding_list("test query")
    b = generate_embedding_list("test query")
    assert a == b
