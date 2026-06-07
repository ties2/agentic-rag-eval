"""Domain layer: the core entities every other layer speaks in.

This is the equivalent of your PHP `Models/` folder. These objects carry NO
behaviour that depends on a database, an LLM, or HTTP — they are pure data.
Keeping them dependency-free is the single most important DDD habit: it means
your business rules never accidentally couple to a vendor SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A raw source document before it is split."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit produced by the chunker."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by search, plus the score that ranked it."""

    chunk: Chunk
    score: float
    # `stage` records WHERE the score came from (vector vs. reranker) so the
    # eval harness and the API can show the full retrieval provenance.
    stage: str = "vector"


class Citation(Enum):
    """How confident the agent is that the answer is grounded in context."""

    GROUNDED = "grounded"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


@dataclass(slots=True)
class Answer:
    """The final object returned to the caller."""

    question: str
    text: str
    contexts: list[RetrievedChunk] = field(default_factory=list)
    citation_status: Citation = Citation.GROUNDED
    # `trace` is the agent's step-by-step decision log — invaluable for both
    # debugging and for showing reviewers that the system reasons, not guesses.
    trace: list[str] = field(default_factory=list)

    @property
    def source_ids(self) -> list[str]:
        return [c.chunk.chunk_id for c in self.contexts]


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One row of the golden test set."""

    question: str
    ground_truth: str
    # IDs of the chunks/docs that SHOULD be retrieved — lets us score retrieval
    # independently of generation.
    relevant_doc_ids: list[str] = field(default_factory=list)
