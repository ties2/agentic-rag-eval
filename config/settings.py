"""Configuration layer.

One typed, validated source of truth for every knob in the system, loaded from
environment variables / a .env file. This is an MLOps fundamental: no
hard-coded model names, paths, or hyper-parameters scattered through the code.
Change behaviour by changing config, never by editing logic.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="RAG_", extra="ignore"
    )

    # --- Providers ---------------------------------------------------------
    # "mock" makes the whole system run offline & deterministically (great for
    # tests/CI and for understanding the flow). Switch to "openai" for real use.
    llm_provider: str = Field(default="mock")
    embedding_provider: str = Field(default="mock")
    reranker_provider: str = Field(default="mock")

    openai_api_key: str | None = Field(default=None)
    llm_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")
    # embedding_dim: int = Field(default=256) #use for mock
    embedding_dim: int = Field(default=768) #use for ollama

    # --- Vector store ------------------------------------------------------
    # ":memory:" runs Qdrant entirely in-process — zero infra needed to learn.
    # In docker-compose this becomes "http://qdrant:6333".
    qdrant_url: str = Field(default=":memory:")
    collection_name: str = Field(default="rag_chunks")

    # --- Pipeline hyper-parameters ----------------------------------------
    chunk_size: int = Field(default=600)
    chunk_overlap: int = Field(default=80)
    top_k_vector: int = Field(default=12)   # candidates from vector search
    top_k_rerank: int = Field(default=4)    # survivors after reranking
    max_refine_loops: int = Field(default=1)  # agentic self-correction budget

    # --- Paths / observability --------------------------------------------
    corpus_dir: str = Field(default="data/corpus")
    golden_set_path: str = Field(default="data/eval/golden_set.jsonl")
    eval_report_dir: str = Field(default="reports")
    mlflow_tracking_uri: str | None = Field(default=None)
    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so the whole app shares one Settings instance."""
    return Settings()
