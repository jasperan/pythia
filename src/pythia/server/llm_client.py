"""LLM backend protocol and factory — picks Ollama or OCI GenAI based on config."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pythia.config import PythiaConfig


@runtime_checkable
class LLMClient(Protocol):
    """Structural type shared by Ollama and OCI GenAI clients."""

    model: str

    async def close(self) -> None: ...

    def generate_stream(
        self, system: str, user: str, model: str | None = None
    ) -> AsyncIterator[str]: ...

    async def generate(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        model: str | None = None,
    ) -> str: ...

    async def generate_suggestions(
        self, query: str, answer: str, model: str | None = None
    ) -> list[str]: ...

    async def health(self) -> bool: ...


def create_llm_client(
    cfg: PythiaConfig, model_override: str | None = None
) -> LLMClient:
    """Return an LLM client for the configured backend. Applies model_override if set."""
    backend = cfg.backend
    if backend == "ollama":
        from pythia.server.ollama import OllamaClient

        model = model_override or cfg.ollama.model
        return OllamaClient(cfg.ollama.base_url, model)
    if backend == "oci-genai":
        from pythia.server.oci_genai import OciGenAIClient

        model = model_override or cfg.oci_genai.model
        return OciGenAIClient(
            base_url=cfg.oci_genai.base_url,
            model=model,
            api_key=cfg.oci_genai.api_key,
            timeout_read=cfg.oci_genai.timeout_read,
        )
    raise ValueError(f"Unknown backend: {backend!r}. Use 'ollama' or 'oci-genai'.")
