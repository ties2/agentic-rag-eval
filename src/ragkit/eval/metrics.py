"""Eval: metrics — the part 90% of RAG portfolios skip.

We measure the two halves of RAG separately, because they fail for different
reasons and you must know WHICH half is broken:

RETRIEVAL metrics (did we fetch the right chunks? — uses labelled relevant ids)
  - hit_rate@k   : was at least one relevant doc retrieved?
  - precision@k  : fraction of retrieved that are relevant
  - recall@k     : fraction of relevant that were retrieved
  - mrr          : 1/rank of the first relevant doc (rewards ranking quality)

GENERATION metrics (given the context, is the answer good? — uses an LLM judge)
  - faithfulness     : every claim grounded in the retrieved context (anti-hallucination)
  - answer_relevance : does the answer actually address the question?

The judge is the SAME LLM interface, so in mock mode the whole harness runs
offline and deterministically. With provider=openai you get real LLM-as-judge.
"""

from __future__ import annotations

from ragkit.domain.models import Answer, RetrievedChunk
from ragkit.ml.llm import LLM

# --- retrieval metrics (no LLM needed) ------------------------------------


def _retrieved_doc_ids(contexts: list[RetrievedChunk]) -> list[str]:
    seen: list[str] = []
    for c in contexts:
        if c.chunk.doc_id not in seen:
            seen.append(c.chunk.doc_id)
    return seen


def hit_rate(contexts, relevant_ids) -> float:
    got = set(_retrieved_doc_ids(contexts))
    return 1.0 if got & set(relevant_ids) else 0.0


def precision_at_k(contexts, relevant_ids) -> float:
    got = _retrieved_doc_ids(contexts)
    if not got:
        return 0.0
    return sum(1 for d in got if d in set(relevant_ids)) / len(got)


def recall_at_k(contexts, relevant_ids) -> float:
    if not relevant_ids:
        return 1.0
    got = set(_retrieved_doc_ids(contexts))
    return len(got & set(relevant_ids)) / len(set(relevant_ids))


def mrr(contexts, relevant_ids) -> float:
    rel = set(relevant_ids)
    for rank, doc_id in enumerate(_retrieved_doc_ids(contexts), start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


# --- generation metrics (LLM judge) ---------------------------------------

_FAITHFUL_JUDGE = (
    "You grade faithfulness. Given an ANSWER and CONTEXT, reply YES only if "
    "every factual claim in the answer is supported by the context, else NO."
)
_RELEVANCE_JUDGE = (
    "You grade answer relevance. Given a QUESTION and ANSWER, reply YES if the "
    "answer directly addresses the question, else NO."
)


def _yes(verdict: str) -> float:
    return 1.0 if verdict.strip().upper().startswith("YES") else 0.0


def faithfulness(answer: Answer, judge: LLM) -> float:
    if not answer.contexts:
        return 0.0
    ctx = "\n".join(c.chunk.text for c in answer.contexts)
    return _yes(judge.complete(_FAITHFUL_JUDGE, f"ANSWER: {answer.text}\nCONTEXT: {ctx}"))


def answer_relevance(answer: Answer, judge: LLM) -> float:
    return _yes(
        judge.complete(
            _RELEVANCE_JUDGE, f"QUESTION: {answer.question}\nANSWER: {answer.text}"
        )
    )
