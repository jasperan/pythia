"""Config loader — reads pythia.yaml into Pydantic models."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8900
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:9b"


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


class TuiConfig(BaseModel):
    theme: str = "dark"


class PythiaConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    searxng: SearxngConfig = Field(default_factory=SearxngConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    tui: TuiConfig = Field(default_factory=TuiConfig)


def load_config(path: str | Path) -> PythiaConfig:
    """Load config from YAML file. Missing sections get defaults."""
    p = Path(path)
    if not p.exists():
        return PythiaConfig()
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    return PythiaConfig(**data)
