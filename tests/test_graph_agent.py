"""The LangGraph orchestrator must produce the same shape of result as the
hand-rolled one, fully offline. This proves the graph compiles, the cycles
terminate, and the node/edge wiring is correct.
"""

from config.settings import Settings
from ragkit.container import build_container
from ragkit.domain.models import Citation


def _settings(tmp_path, provider: str) -> Settings:
    corpus = tmp_path / f"corpus_{provider}"
    corpus.mkdir(exist_ok=True)
    (corpus / "policy.txt").write_text(
        "Refunds are issued within 14 days of the transaction date.",
        encoding="utf-8",
    )
    return Settings(
        llm_provider="mock",
        embedding_provider="mock",
        reranker_provider="mock",
        agent_provider=provider,
        qdrant_url=":memory:",
        corpus_dir=str(corpus),
        embedding_dim=64,
    )


def test_langgraph_agent_runs(tmp_path):
    c = build_container(_settings(tmp_path, "langgraph"))
    c.ingestion.run()
    ans = c.agent.answer("How long do I have to request a refund?")

    assert ans.text
    assert ans.contexts
    assert ans.citation_status in {
        Citation.GROUNDED,
        Citation.PARTIAL,
        Citation.UNSUPPORTED,
    }
    joined = " ".join(ans.trace)
    # graph node trace markers
    for marker in ("rewrite", "vector_search", "rerank", "grade", "generate", "verify"):
        assert marker in joined, f"missing trace stage: {marker}"


def test_both_orchestrators_agree_on_shape(tmp_path):
    hand = build_container(_settings(tmp_path, "hand_rolled"))
    hand.ingestion.run()
    a1 = hand.agent.answer("When are refunds issued?")

    graph = build_container(_settings(tmp_path, "langgraph"))
    graph.ingestion.run()
    a2 = graph.agent.answer("When are refunds issued?")

    assert bool(a1.contexts) == bool(a2.contexts)
    assert a1.text and a2.text
