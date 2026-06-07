"""ML: reranker.

Vector search optimises for recall (cast a wide net, top_k_vector candidates);
the reranker optimises for precision (re-score those candidates against the
query with a more expensive, more accurate model, keep top_k_rerank). This
two-stage retrieval is the single biggest, cheapest quality win in RAG and is
exactly what most portfolio projects omit.
"""

from __future__ import annotations

from typing import Protocol

from config.settings import Settings
from ragkit.domain.models import RetrievedChunk


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class LexicalReranker:
    """Offline mock: token-overlap (Jaccard) re-scoring.

    Crude, but it genuinely re-orders results and makes the two-stage pipeline
    observable without downloading a model.
    """

    def rerank(self, query, candidates, top_k):
        q = set(query.lower().split())
        rescored = []
        for c in candidates:
            d = set(c.chunk.text.lower().split())
            overlap = len(q & d) / (len(q | d) or 1)
            rescored.append(
                RetrievedChunk(chunk=c.chunk, score=overlap, stage="rerank")
            )
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # lazy import

        self._model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k):
        pairs = [[query, c.chunk.text] for c in candidates]
        scores = self._model.predict(pairs)
        rescored = [
            RetrievedChunk(chunk=c.chunk, score=float(s), stage="rerank")
            for c, s in zip(candidates, scores)
        ]
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker_provider == "cross_encoder":
        return CrossEncoderReranker()
    return LexicalReranker()
