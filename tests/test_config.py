"""Tests for config loader."""
import tempfile
from pathlib import Path

from pythia.config import load_config, PythiaConfig


def test_load_config_from_yaml():
    yaml_content = """
server:
  host: "127.0.0.1"
  port: 9000
ollama:
  base_url: "http://localhost:11434"
  model: "qwen3.5:9b"
searxng:
  base_url: "http://localhost:8888"
  max_results: 5
  categories:
    - general
oracle:
  dsn: "localhost:1521/FREEPDB1"
  user: "pythia"
  password: "pythia"  # pragma: allowlist secret
  cache_similarity_threshold: 0.90
  embedding_model: "ALL_MINILM_L6_V2"
tui:
  theme: "dark"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = load_config(f.name)

    assert isinstance(cfg, PythiaConfig)
    assert cfg.server.port == 9000
    assert cfg.ollama.model == "qwen3.5:9b"
    assert cfg.searxng.max_results == 5
    assert cfg.oracle.cache_similarity_threshold == 0.90


def test_load_config_defaults():
    yaml_content = """
ollama:
  model: "qwen3.5:9b"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = load_config(f.name)

    assert cfg.server.port == 8900
    assert cfg.searxng.base_url == "http://localhost:8889"
    assert cfg.searxng.categories == ["general"]
    assert cfg.oracle.cache_similarity_threshold == 0.85
