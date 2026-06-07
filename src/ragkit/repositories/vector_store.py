"""Repository layer: vector persistence.

Equivalent to your PHP `Database/` folder. Services ask this repository to
"upsert chunks" and "search" — they never see Qdrant types. If you later move
to pgvector, Weaviate, or Pinecone, only this file changes.

Default URL ":memory:" runs Qdrant in-process: no Docker, no server, perfect
for learning. docker-compose flips it to a real Qdrant container.
"""

from __future__ import annotations

import uuid

from config.settings import Settings
from ragkit.domain.models import Chunk, RetrievedChunk
from ragkit.observability.logging import get_logger

log = get_logger("repositories.vector_store")


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._settings = settings
        self._dim = settings.embedding_dim
        if settings.qdrant_url == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(url=settings.qdrant_url)
        self._collection = settings.collection_name
        self._VectorParams = VectorParams
        self._Distance = Distance
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=self._VectorParams(
                    size=self._dim, distance=self._Distance.COSINE
                ),
            )
            log.info("Created collection '%s' (dim=%d)", self._collection, self._dim)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "chunk_id": ch.chunk_id,
                    "doc_id": ch.doc_id,
                    "text": ch.text,
                    "metadata": ch.metadata,
                },
            )
            for ch, vec in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        log.info("Upserted %d points", len(points))

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        hits = self._client.query_points(
            collection_name=self._collection, query=query_vector, limit=top_k
        ).points
        results = []
        for h in hits:
            p = h.payload or {}
            results.append(
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id=p.get("chunk_id", ""),
                        doc_id=p.get("doc_id", ""),
                        text=p.get("text", ""),
                        metadata=p.get("metadata", {}),
                    ),
                    score=float(h.score),
                    stage="vector",
                )
            )
        return results

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count
