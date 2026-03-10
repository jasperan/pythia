"""Config loader — reads pythia.yaml into Pydantic models."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8900


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:9b"


class SearxngConfig(BaseModel):
    base_url: str = "http://localhost:8889"
    max_results: int = 8
    categories: list[str] = Field(default_factory=lambda: ["general", "science", "it"])


class OracleConfig(BaseModel):
    dsn: str = "localhost:1523/FREEPDB1"
    user: str = "pythia"
    password: str = "pythia"
    cache_similarity_threshold: float = 0.85
    embedding_model: str = "ALL_MINILM_L6_V2"


class TuiConfig(BaseModel):
    theme: str = "dark"


class PythiaConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    searxng: SearxngConfig = Field(default_factory=SearxngConfig)
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    tui: TuiConfig = Field(default_factory=TuiConfig)


def load_config(path: str | Path) -> PythiaConfig:
    """Load config from YAML file. Missing sections get defaults."""
    p = Path(path)
    if not p.exists():
        return PythiaConfig()
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    return PythiaConfig(**data)
