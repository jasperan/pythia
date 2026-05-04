"""Tests for the LLM client factory."""
import pytest

from pythia.config import PythiaConfig
from pythia.server.llm_client import create_llm_client
from pythia.server.oci_genai import OciGenAIClient
from pythia.server.ollama import OllamaClient


def test_factory_defaults_to_ollama():
    cfg = PythiaConfig()
    client = create_llm_client(cfg)
    assert isinstance(client, OllamaClient)
    assert client.model == cfg.ollama.model


def test_factory_ollama_backend_uses_ollama_config():
    cfg = PythiaConfig()
    cfg.backend = "ollama"
    cfg.ollama.base_url = "http://example:11434"
    cfg.ollama.model = "custom-ollama"
    client = create_llm_client(cfg)
    assert isinstance(client, OllamaClient)
    assert client.base_url == "http://example:11434"
    assert client.model == "custom-ollama"


def test_factory_oci_genai_backend():
    cfg = PythiaConfig()
    cfg.backend = "oci-genai"
    client = create_llm_client(cfg)
    assert isinstance(client, OciGenAIClient)
    assert client.model == "xai.grok-4"
    assert client.base_url == "http://localhost:9999/v1"
    assert client.api_key == "oci-genai"  # pragma: allowlist secret


def test_factory_model_override_ollama():
    cfg = PythiaConfig()
    cfg.backend = "ollama"
    client = create_llm_client(cfg, model_override="override-m")
    assert client.model == "override-m"


def test_factory_model_override_oci_genai():
    cfg = PythiaConfig()
    cfg.backend = "oci-genai"
    client = create_llm_client(cfg, model_override="xai.grok-3-mini")
    assert isinstance(client, OciGenAIClient)
    assert client.model == "xai.grok-3-mini"


def test_factory_oci_genai_propagates_full_config():
    cfg = PythiaConfig()
    cfg.backend = "oci-genai"
    cfg.oci_genai.base_url = "http://remote:9000/v1"
    cfg.oci_genai.api_key = "bearer-token"  # pragma: allowlist secret
    cfg.oci_genai.timeout_read = 300
    client = create_llm_client(cfg)
    assert isinstance(client, OciGenAIClient)
    assert client.base_url == "http://remote:9000/v1"
    assert client.api_key == "bearer-token"  # pragma: allowlist secret
    assert client.timeout_read == 300


def test_factory_oci_genai_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OCI_PROXY_API_KEY", "env-proxy-token")

    cfg = PythiaConfig()
    cfg.backend = "oci-genai"
    client = create_llm_client(cfg)

    assert isinstance(client, OciGenAIClient)
    assert client.api_key == "env-proxy-token"  # pragma: allowlist secret


def test_factory_unknown_backend_raises():
    cfg = PythiaConfig()
    cfg.backend = "ollama"
    # bypass pydantic validation
    object.__setattr__(cfg, "backend", "mystery")
    with pytest.raises(ValueError, match="Unknown backend"):
        create_llm_client(cfg)
