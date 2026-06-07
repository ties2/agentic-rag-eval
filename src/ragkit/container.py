"""Composition root.

One place that knows how to build the whole object graph from Settings. Every
entry point (API, CLI scripts, tests) calls `build_container()` instead of
constructing services by hand. This is dependency injection done simply — it is
why none of the layers ever call `get_settings()` or instantiate vendors
themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import Settings, get_settings
from ragkit.ml.embeddings import build_embedder
from ragkit.ml.llm import build_llm
from ragkit.ml.reranker import build_reranker
from ragkit.observability.logging import setup_logging
from ragkit.repositories.vector_store import VectorStore
from ragkit.services.agent import AgentService
from ragkit.services.ingestion import IngestionService
from ragkit.services.retrieval import RetrievalService


@dataclass
class Container:
    settings: Settings
    store: VectorStore
    ingestion: IngestionService
    retrieval: RetrievalService
    agent: AgentService


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    embedder = build_embedder(settings)
    reranker = build_reranker(settings)
    llm = build_llm(settings)
    store = VectorStore(settings)

    ingestion = IngestionService(settings, embedder, store)
    retrieval = RetrievalService(settings, embedder, store, reranker, llm)
    agent = AgentService(settings, retrieval, llm)

    return Container(
        settings=settings,
        store=store,
        ingestion=ingestion,
        retrieval=retrieval,
        agent=agent,
    )
