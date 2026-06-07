"""ML: chunking strategy.

Chunking quality silently determines retrieval quality — bad chunks cap your
ceiling no matter how good the embedder is. We use a simple, well-behaved
recursive splitter (paragraph -> sentence -> hard cut) with overlap so context
isn't severed mid-thought. Swapping this for a smarter strategy (semantic /
markdown-aware) is a clean, isolated experiment thanks to the layering.
"""

from __future__ import annotations

import re
import uuid

from ragkit.domain.models import Chunk, Document

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Hard-window fallback for spans that exceed `size`."""
    out, start = [], 0
    while start < len(text):
        end = start + size
        out.append(text[start:end])
        start = end - overlap
    return out


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    pieces: list[str] = []
    for para in _PARAGRAPH.split(doc.text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            pieces.append(para)
            continue
        # Pack sentences greedily up to `size`.
        buf = ""
        for sent in _SENTENCE.split(para):
            if len(buf) + len(sent) + 1 > size:
                if buf:
                    pieces.append(buf.strip())
                buf = sent if len(sent) <= size else ""
                if not buf:
                    pieces.extend(_split_long(sent, size, overlap))
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            pieces.append(buf.strip())

    chunks: list[Chunk] = []
    for i, text in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{i}::{uuid.uuid4().hex[:8]}",
                doc_id=doc.doc_id,
                text=text,
                metadata={**doc.metadata, "position": i},
            )
        )
    return chunks
