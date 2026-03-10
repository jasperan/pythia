"""Integration tests for CLI runner — requires no running services."""
import json
import tempfile
from pathlib import Path

from pythia.cli_runner import run_embed_single, run_embed_batch


def test_embed_single_valid_json():
    result = run_embed_single("integration test")
    data = json.loads(result)
    assert data["text"] == "integration test"
    assert len(data["embedding"]) == 384


def test_embed_batch_jsonl():
    texts = ["query one", "query two", "query three"]
    results = run_embed_batch(texts)
    assert len(results) == 3
    for i, line in enumerate(results):
        data = json.loads(line)
        assert data["text"] == texts[i]


def test_embed_batch_file_format():
    """Test that JSONL file parsing works end-to-end."""
    lines = ['{"text": "first line"}', '{"text": "second line"}', "plain text third"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(lines))
        f.flush()
        path = Path(f.name)

    parsed_texts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            parsed_texts.append(obj.get("text", line))
        except json.JSONDecodeError:
            parsed_texts.append(line)

    assert parsed_texts == ["first line", "second line", "plain text third"]
    results = run_embed_batch(parsed_texts)
    assert len(results) == 3

    path.unlink()
