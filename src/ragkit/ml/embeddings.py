"""ML: embedding providers behind one Protocol.

The rest of the system depends on the `Embedder` interface, never on OpenAI
directly. That's the Dependency Inversion principle: high-level code (services)
defines the contract; low-level details (vendors) implement it. Result: you can
unit-test, run offline, or switch vendors without touching a service.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from config.settings import Settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic, dependency-free embedder for offline runs and tests.

    It hashes token bucket counts into a fixed-width vector. It is NOT
    semantically meaningful, but it is stable and lets every other layer be
    exercised end-to-end without network or GPU.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class OpenAIEmbedder:
    def __init__(self, model: str, dim: int, api_key: str | None) -> None:
        from openai import OpenAI  # imported lazily so mock mode needs no dep

        self.model = model
        self.dim = dim
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(
            model=self.model, input=texts, dimensions=self.dim
        )
        return [d.embedding for d in resp.data]

# use openai
def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "ollama":
        return OllamaEmbedder(model="nomic-embed-text")
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(
            settings.embedding_model, settings.embedding_dim, settings.openai_api_key
        )
    return HashEmbedder(settings.embedding_dim)

#use ollama

class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text"):
        from openai import OpenAI
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]
