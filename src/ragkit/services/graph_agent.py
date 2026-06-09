"""Service: LangGraph agent — the same agent as a real graph.

This is the M8 port. It builds a `StateGraph` whose nodes are the individual
steps and whose CONDITIONAL EDGES express the two interesting control-flow
features:

  * a refine CYCLE:  grade -> refine -> rewrite -> ... (retry weak retrieval)
  * a regenerate CYCLE:  verify -> generate -> verify  (fix unfaithful answers)

It reuses `RetrievalService` (rewrite/search/rerank) and `RagOperations`
(grade/generate/verify) verbatim, so behaviour matches the hand-rolled agent —
only the orchestration is now declarative.

ASCII view of the graph:

    START -> rewrite -> search -> rerank -> grade
                ^                              |
                |                      (weak & budget left)
                +------------- refine <--------+
                                               |
                          (ok / budget spent)  v
                                            generate -> verify
                                                ^          |
                                  (unfaithful & retry left)|
                                                +----------+
                                                           |
                                                  (done)   v
                                                        finalize -> END
"""

from __future__ import annotations

from typing import TypedDict

from config.settings import Settings
from ragkit.domain.models import Answer, Citation, RetrievedChunk
from ragkit.observability.logging import get_logger
from ragkit.services.operations import RagOperations
from ragkit.services.retrieval import RetrievalService

log = get_logger("services.graph_agent")


class GraphState(TypedDict):
    question: str
    search_query: str
    contexts: list[RetrievedChunk]
    answer: str
    sufficient: bool
    faithful: bool
    refine_count: int
    gen_attempts: int
    trace: list[str]


class LangGraphAgent:
    def __init__(
        self, settings: Settings, retrieval: RetrievalService, ops: RagOperations
    ) -> None:
        self._s = settings
        self._retrieval = retrieval
        self._ops = ops
        self._graph = self._build()

    # --- nodes (each returns a partial state update) -----------------------
    def _rewrite(self, s: GraphState) -> dict:
        sq = self._retrieval.rewrite(s["question"])
        return {"search_query": sq, "trace": s["trace"] + [f"rewrite: {sq!r}"]}

    def _search(self, s: GraphState) -> dict:
        trace = list(s["trace"])
        ctx = self._retrieval.vector_search(s["search_query"], trace)
        return {"contexts": ctx, "trace": trace}

    def _rerank(self, s: GraphState) -> dict:
        trace = list(s["trace"])
        ctx = self._retrieval.rerank_candidates(s["question"], s["contexts"], trace)
        return {"contexts": ctx, "trace": trace}

    def _grade(self, s: GraphState) -> dict:
        ok = self._ops.grade_context(s["question"], s["contexts"])
        return {"sufficient": ok, "trace": s["trace"] + [f"grade_context: {ok}"]}

    def _refine(self, s: GraphState) -> dict:
        n = s["refine_count"] + 1
        return {
            "refine_count": n,
            "question": s["question"] + " (be specific)",
            "trace": s["trace"] + [f"refine: loop {n}"],
        }

    def _generate(self, s: GraphState) -> dict:
        text = self._ops.generate(s["question"], s["contexts"])
        attempts = s["gen_attempts"] + 1
        return {
            "answer": text,
            "gen_attempts": attempts,
            "trace": s["trace"] + [f"generate: attempt {attempts}"],
        }

    def _verify(self, s: GraphState) -> dict:
        ok = self._ops.is_faithful(s["answer"], s["contexts"])
        return {"faithful": ok, "trace": s["trace"] + [f"verify: faithful={ok}"]}

    def _abstain(self, s: GraphState) -> dict:
        return {
            "answer": "I don't have enough information to answer that.",
            "trace": s["trace"] + ["abstain"],
        }

    # --- routers (read-only; pick the next node) ---------------------------
    def _route_after_grade(self, s: GraphState) -> str:
        if not s["contexts"]:
            return "abstain"
        if s["sufficient"] or s["refine_count"] >= self._s.max_refine_loops:
            return "generate"
        return "refine"

    def _route_after_verify(self, s: GraphState) -> str:
        if s["faithful"] or s["gen_attempts"] >= 2:
            return "finalize"
        return "generate"

    def _finalize(self, s: GraphState) -> dict:
        if s["faithful"]:
            status = Citation.PARTIAL if s["gen_attempts"] > 1 else Citation.GROUNDED
        else:
            status = Citation.UNSUPPORTED
        return {"trace": s["trace"] + [f"final_status: {status.value}"]}

    # --- graph assembly ----------------------------------------------------
    def _build(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(GraphState)
        g.add_node("rewrite", self._rewrite)
        g.add_node("search", self._search)
        g.add_node("rerank", self._rerank)
        g.add_node("grade", self._grade)
        g.add_node("refine", self._refine)
        g.add_node("generate", self._generate)
        g.add_node("verify", self._verify)
        g.add_node("finalize", self._finalize)
        g.add_node("abstain", self._abstain)

        g.add_edge(START, "rewrite")
        g.add_edge("rewrite", "search")
        g.add_edge("search", "rerank")
        g.add_edge("rerank", "grade")
        g.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {"abstain": "abstain", "generate": "generate", "refine": "refine"},
        )
        g.add_edge("refine", "rewrite")
        g.add_edge("generate", "verify")
        g.add_conditional_edges(
            "verify",
            self._route_after_verify,
            {"generate": "generate", "finalize": "finalize"},
        )
        g.add_edge("finalize", END)
        g.add_edge("abstain", END)
        return g.compile()

    # --- public API (same signature as AgentService) ----------------------
    def answer(self, question: str) -> Answer:
        init: GraphState = {
            "question": question,
            "search_query": "",
            "contexts": [],
            "answer": "",
            "sufficient": False,
            "faithful": False,
            "refine_count": 0,
            "gen_attempts": 0,
            "trace": [],
        }
        final = self._graph.invoke(init)

        if final["faithful"]:
            status = (
                Citation.PARTIAL if final["gen_attempts"] > 1 else Citation.GROUNDED
            )
        elif not final["contexts"]:
            status = Citation.UNSUPPORTED
        else:
            status = Citation.UNSUPPORTED

        return Answer(
            question=question,
            text=final["answer"],
            contexts=final["contexts"],
            citation_status=status,
            trace=final["trace"],
        )
