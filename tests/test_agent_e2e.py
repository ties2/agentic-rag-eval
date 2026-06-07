"""End-to-end test of the agent in fully-offline mock mode.

This proves the entire layered pipeline wires together: ingestion -> vector
store -> retrieval (rewrite + search + rerank) -> agent (grade + generate +
self-check) -> Answer. It needs no API key and no network.
"""

from config.settings import Settings
from ragkit.container import build_container
from ragkit.domain.models import Citation


def _mock_settings(tmp_path) -> Settings:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "policy.txt").write_text(
        "Refunds are issued within 14 days of the transaction date. "
        "International transfers cost 5 euros plus a conversion margin.",
        encoding="utf-8",
    )
    return Settings(
        llm_provider="mock",
        embedding_provider="mock",
        reranker_provider="mock",
        qdrant_url=":memory:",
        corpus_dir=str(corpus),
        embedding_dim=64,
    )


def test_agent_end_to_end(tmp_path):
    container = build_container(_mock_settings(tmp_path))
    n = container.ingestion.run()
    assert n >= 1
    assert container.store.count() >= 1

    answer = container.agent.answer("How long do I have to request a refund?")
    assert answer.text
    assert answer.contexts  # retrieval returned something
    assert answer.citation_status in {
        Citation.GROUNDED,
        Citation.PARTIAL,
        Citation.UNSUPPORTED,
    }
    # The trace should record every stage of the agent loop.
    joined = " ".join(answer.trace)
    assert "rewrite" in joined
    assert "vector_search" in joined
    assert "rerank" in joined
