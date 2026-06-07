"""Service: the agent — what makes this 'agentic' RAG, not plain RAG.

A plain RAG is one straight shot: retrieve -> generate. An agent adds DECISIONS
and SELF-CORRECTION. Here we implement a small explicit state machine:

    retrieve -> grade_context -> [if weak: refine query & retry] ->
    generate -> self_check_faithfulness -> [if unsupported: regenerate]

We hand-roll the graph (instead of pulling in LangGraph) so the control flow is
fully visible and teachable. The README explains how to lift this into LangGraph
once you understand the moving parts — the node functions map 1:1.
"""

from __future__ import annotations

from config.settings import Settings
from ragkit.domain.models import Answer, Citation, RetrievedChunk
from ragkit.ml.llm import LLM
from ragkit.observability.logging import get_logger
from ragkit.services.retrieval import RetrievalService

log = get_logger("services.agent")

_GENERATE_SYSTEM = (
    "Answer the question using ONLY the provided context. Cite chunk ids in "
    "[brackets]. If the context is insufficient, say you don't know."
)
_GRADE_SYSTEM = (
    "You grade whether the context is sufficient to answer the question. "
    "Judge strictly. Reply with a single word: YES or NO."
)
_FAITHFUL_SYSTEM = (
    "You are a faithfulness judge. Decide whether EVERY claim in the answer is "
    "supported by the context. Reply with a single word: YES or NO."
)


class AgentService:
    def __init__(
        self, settings: Settings, retrieval: RetrievalService, llm: LLM
    ) -> None:
        self._s = settings
        self._retrieval = retrieval
        self._llm = llm

    # --- nodes -------------------------------------------------------------
    def _grade_context(self, question: str, ctx: list[RetrievedChunk]) -> bool:
        if not ctx:
            return False
        joined = "\n".join(c.chunk.text for c in ctx)
        verdict = self._llm.complete(
            _GRADE_SYSTEM, f"Question: {question}\nContext: {joined}"
        )
        return verdict.strip().upper().startswith("YES")

    def _generate(self, question: str, ctx: list[RetrievedChunk]) -> str:
        blocks = "\n\n".join(f"[{c.chunk.chunk_id}] {c.chunk.text}" for c in ctx)
        return self._llm.complete(
            _GENERATE_SYSTEM, f"Question: {question}\nContext:\n{blocks}"
        ).strip()

    def _is_faithful(self, answer: str, ctx: list[RetrievedChunk]) -> bool:
        joined = "\n".join(c.chunk.text for c in ctx)
        verdict = self._llm.complete(
            _FAITHFUL_SYSTEM, f"Answer: {answer}\nContext: {joined}"
        )
        return verdict.strip().upper().startswith("YES")

    # --- orchestration -----------------------------------------------------
    def answer(self, question: str) -> Answer:
        trace: list[str] = []
        ctx = self._retrieval.retrieve(question, trace)

        # Self-correction loop on retrieval: if context is judged weak, retry
        # with the already-rewritten (denser) query up to a small budget.
        loops = 0
        while not self._grade_context(question, ctx) and loops < self._s.max_refine_loops:
            loops += 1
            trace.append(f"grade_context: weak -> refine (loop {loops})")
            ctx = self._retrieval.retrieve(question + " (be specific)", trace)

        if not ctx:
            trace.append("no_context -> abstain")
            return Answer(
                question=question,
                text="I don't have enough information to answer that.",
                contexts=[],
                citation_status=Citation.UNSUPPORTED,
                trace=trace,
            )

        text = self._generate(question, ctx)
        trace.append("generate: drafted answer")

        faithful = self._is_faithful(text, ctx)
        status = Citation.GROUNDED if faithful else Citation.UNSUPPORTED
        if not faithful:
            trace.append("self_check: UNSUPPORTED -> regenerate once")
            text = self._generate(question, ctx)
            status = (
                Citation.PARTIAL if self._is_faithful(text, ctx) else Citation.UNSUPPORTED
            )

        trace.append(f"final_status: {status.value}")
        return Answer(
            question=question,
            text=text,
            contexts=ctx,
            citation_status=status,
            trace=trace,
        )
