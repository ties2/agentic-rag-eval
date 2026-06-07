"""Service: ingestion pipeline (offline / batch).

load -> chunk -> embed -> store. This is the "training-time" half of RAG.
A service orchestrates ML components and the repository; it owns the WORKFLOW
but delegates the HOW to the layers below. Notice it depends only on
interfaces, so it is trivially unit-testable.
"""

from __future__ import annotations

from pathlib import Path

from config.settings import Settings
from ragkit.domain.models import Document
from ragkit.ml.chunking import chunk_document
from ragkit.ml.embeddings import Embedder
from ragkit.observability.logging import get_logger
from ragkit.repositories.vector_store import VectorStore

log = get_logger("services.ingestion")


class IngestionService:
    def __init__(
        self, settings: Settings, embedder: Embedder, store: VectorStore
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store

    def load_corpus(self, corpus_dir: str | None = None) -> list[Document]:
        directory = Path(corpus_dir or self._settings.corpus_dir)
        docs: list[Document] = []
        for path in sorted(directory.glob("*.txt")):
            docs.append(
                Document(
                    doc_id=path.stem,
                    text=path.read_text(encoding="utf-8"),
                    metadata={"source": path.name},
                )
            )
        log.info("Loaded %d documents from %s", len(docs), directory)
        return docs

    def ingest(self, docs: list[Document]) -> int:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(
                chunk_document(
                    doc, self._settings.chunk_size, self._settings.chunk_overlap
                )
            )
        log.info("Produced %d chunks", len(all_chunks))
        # Embed in batches to stay within provider limits.
        batch = 64
        for i in range(0, len(all_chunks), batch):
            window = all_chunks[i : i + batch]
            vectors = self._embedder.embed([c.text for c in window])
            self._store.upsert(window, vectors)
        return len(all_chunks)

    def run(self, corpus_dir: str | None = None) -> int:
        return self.ingest(self.load_corpus(corpus_dir))
