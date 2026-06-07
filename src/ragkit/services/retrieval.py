"""Service: retrieval (online / query-time).

Two-stage retrieval with an optional LLM query-rewrite in front:

    rewrite(query) -> vector_search(top_k_vector) -> rerank(top_k_rerank)

Each stage is logged into the answer trace so the agent (and you) can see how
the candidate set narrowed. This is the precision/recall engine of the system.
"""

from __future__ import annotations

from config.settings import Settings
from ragkit.ml.embeddings import Embedder
from ragkit.ml.llm import LLM
from ragkit.ml.reranker import Reranker
from ragkit.observability.logging import get_logger
from ragkit.repositories.vector_store import VectorStore

log = get_logger("services.retrieval")

_REWRITE_SYSTEM = (
    "You rewrite a user question into a denser search query that maximises "
    "retrieval recall. Keep key entities. Return ONLY the rewritten query."
)


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker,
        llm: LLM,
    ) -> None:
        self._s = settings
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._llm = llm

    def rewrite(self, query: str) -> str:
        rewritten = self._llm.complete(_REWRITE_SYSTEM, query).strip()
        return rewritten or query

    def retrieve(self, query: str, trace: list[str] | None = None):
        trace = trace if trace is not None else []

        search_query = self.rewrite(query)
        trace.append(f"rewrite: {search_query!r}")

        [qvec] = self._embedder.embed([search_query])
        candidates = self._store.search(qvec, self._s.top_k_vector)
        trace.append(f"vector_search: {len(candidates)} candidates")

        reranked = self._reranker.rerank(query, candidates, self._s.top_k_rerank)
        trace.append(
            f"rerank: kept {len(reranked)} "
            f"(top score {reranked[0].score:.3f})" if reranked else "rerank: empty"
        )
        return reranked
