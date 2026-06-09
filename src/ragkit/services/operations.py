"""Service: shared RAG operations.

The agent's individual *operations* (grade context, generate, check
faithfulness) are separated from its *orchestration* (the control flow). Both
the hand-rolled `AgentService` and the `LangGraphAgent` reuse these exact
operations, so the two orchestrators are guaranteed to behave identically at the
step level — only the wiring differs. This is a clean way to show two
implementations of the same behaviour.
"""

from __future__ import annotations

from ragkit.domain.models import RetrievedChunk
from ragkit.ml.llm import LLM

GENERATE_SYSTEM = (
    "Answer the question using ONLY the provided context. Cite chunk ids in "
    "[brackets]. If the context is insufficient, say you don't know."
)
GRADE_SYSTEM = (
    "You grade whether the context is sufficient to answer the question. "
    "Judge strictly. Reply with a single word: YES or NO."
)
FAITHFUL_SYSTEM = (
    "You are a faithfulness judge. Decide whether EVERY claim in the answer is "
    "supported by the context. Reply with a single word: YES or NO."
)


def _yes(text: str) -> bool:
    return text.strip().upper().startswith("YES")


class RagOperations:
    """Stateless RAG step operations over an LLM."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def grade_context(self, question: str, ctx: list[RetrievedChunk]) -> bool:
        if not ctx:
            return False
        joined = "\n".join(c.chunk.text for c in ctx)
        return _yes(
            self._llm.complete(GRADE_SYSTEM, f"Question: {question}\nContext: {joined}")
        )

    def generate(self, question: str, ctx: list[RetrievedChunk]) -> str:
        blocks = "\n\n".join(f"[{c.chunk.chunk_id}] {c.chunk.text}" for c in ctx)
        return self._llm.complete(
            GENERATE_SYSTEM, f"Question: {question}\nContext:\n{blocks}"
        ).strip()

    def is_faithful(self, answer: str, ctx: list[RetrievedChunk]) -> bool:
        if not ctx:
            return False
        joined = "\n".join(c.chunk.text for c in ctx)
        return _yes(
            self._llm.complete(FAITHFUL_SYSTEM, f"Answer: {answer}\nContext: {joined}")
        )
