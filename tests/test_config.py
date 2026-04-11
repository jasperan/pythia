"""Tests for config loader."""
import tempfile

from pythia.config import load_config, PythiaConfig, ResearchConfig, resolve_config_path


def test_research_config_completeness_checks_default():
    cfg = ResearchConfig()
    assert cfg.max_completeness_checks == 2


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


def test_resolve_config_path_prefers_explicit_path_over_env(monkeypatch, tmp_path):
    env_config = tmp_path / "env.yaml"
    env_config.write_text("server:\n  port: 9001\n")
    explicit_config = tmp_path / "explicit.yaml"
    explicit_config.write_text("server:\n  port: 9002\n")
    monkeypatch.setenv("PYTHIA_CONFIG", str(env_config))

    assert resolve_config_path(explicit_config) == explicit_config.resolve()


def test_resolve_config_path_uses_env_for_default_name(monkeypatch, tmp_path):
    config_path = tmp_path / "env.yaml"
    config_path.write_text("server:\n  port: 9001\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHIA_CONFIG", str(config_path))

    assert resolve_config_path("pythia.yaml") == config_path.resolve()


def test_resolve_config_path_returns_none_when_default_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHIA_CONFIG", raising=False)

    assert resolve_config_path("pythia.yaml") is None
