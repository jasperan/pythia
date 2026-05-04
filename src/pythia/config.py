"""Config loader and config-path resolution helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_CONFIG_NAME = "pythia.yaml"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8900
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:9b"


class OciGenAIConfig(BaseModel):
    base_url: str = "http://localhost:9999/v1"
    model: str = "xai.grok-4"
    api_key: str = "oci-genai"
    timeout_read: int = 180

    @model_validator(mode="before")
    @classmethod
    def env_overrides(cls, data):
        """Allow environment variables to override OCI GenAI proxy config."""
        if not isinstance(data, dict):
            data = {}
        for field_name, env_var in [
            ("base_url", "PYTHIA_OCI_GENAI_BASE_URL"),
            ("model", "PYTHIA_OCI_GENAI_MODEL"),
            ("api_key", "PYTHIA_OCI_GENAI_API_KEY"),
            ("api_key", "OCI_PROXY_API_KEY"),
            ("timeout_read", "PYTHIA_OCI_GENAI_TIMEOUT_READ"),
        ]:
            val = os.environ.get(env_var)
            if val:
                data[field_name] = int(val) if field_name == "timeout_read" else val
        return data


class SearxngConfig(BaseModel):
    base_url: str = "http://localhost:8889"
    max_results: int = 8
    categories: list[str] = Field(default_factory=lambda: ["general"])


class OracleConfig(BaseModel):
    dsn: str = "localhost:1523/FREEPDB1"
    user: str = "pythia"
    password: str = ""
    cache_similarity_threshold: float = 0.85
    embedding_model: str = "ALL_MINILM_L6_V2"

    @model_validator(mode="before")
    @classmethod
    def env_overrides(cls, data):
        """Allow environment variables to override sensitive Oracle config."""
        if isinstance(data, dict):
            for field_name, env_var in [
                ("dsn", "PYTHIA_ORACLE_DSN"),
                ("user", "PYTHIA_ORACLE_USER"),
                ("password", "PYTHIA_ORACLE_PASSWORD"),
            ]:
                val = os.environ.get(env_var)
                if val:
                    data[field_name] = val
        return data


class ResearchConfig(BaseModel):
    max_rounds: int = 3
    max_sub_queries: int = 5
    deep_scrape: bool = True
    recall_threshold: float = 0.70
    max_completeness_checks: int = 2


class TuiConfig(BaseModel):
    theme: str = "dark"


class PythiaConfig(BaseModel):
    backend: Literal["ollama", "oci-genai"] = "ollama"
    server: ServerConfig = Field(default_factory=ServerConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    oci_genai: OciGenAIConfig = Field(default_factory=OciGenAIConfig)
    searxng: SearxngConfig = Field(default_factory=SearxngConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    tui: TuiConfig = Field(default_factory=TuiConfig)


def load_config(path: str | Path) -> PythiaConfig:
    """Load config from YAML file. Missing sections get defaults."""
    resolved = resolve_config_path(path)
    if resolved is None:
        return PythiaConfig()
    with resolved.open() as f:
        data = yaml.safe_load(f) or {}
    return PythiaConfig(**data)


def resolve_config_path(path: str | Path = DEFAULT_CONFIG_NAME) -> Path | None:
    """Resolve the config path using CLI/env precedence.

    Resolution order:
    1. Explicit ``--config`` path when it is not the default filename.
    2. ``PYTHIA_CONFIG`` when the caller is using the default filename.
    3. ``pythia.yaml`` in the current working directory.
    """
    requested = Path(path).expanduser()

    if requested != Path(DEFAULT_CONFIG_NAME):
        return requested.resolve() if requested.exists() else None

    env_path = os.environ.get("PYTHIA_CONFIG")
    if env_path:
        env_candidate = Path(env_path).expanduser()
        return env_candidate.resolve() if env_candidate.exists() else None

    return requested.resolve() if requested.exists() else None
