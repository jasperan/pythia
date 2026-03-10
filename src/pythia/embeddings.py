"""Shared embedding generation using sentence-transformers."""
from __future__ import annotations

_model = None

MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSIONS = 384


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def generate_embedding_list(text: str) -> list[float]:
    """Generate embedding as a list of floats."""
    model = _get_model()
    return model.encode(text).tolist()


def generate_embedding(text: str) -> str:
    """Generate embedding as a JSON array string (for Oracle TO_VECTOR)."""
    values = generate_embedding_list(text)
    return '[' + ','.join(map(str, values)) + ']'
