from ragkit.domain.models import Chunk, RetrievedChunk
from ragkit.eval import metrics as M


def _ctx(doc_ids):
    return [
        RetrievedChunk(
            chunk=Chunk(chunk_id=f"{d}::0", doc_id=d, text="x"), score=1.0
        )
        for d in doc_ids
    ]


def test_hit_rate():
    assert M.hit_rate(_ctx(["a", "b"]), ["b"]) == 1.0
    assert M.hit_rate(_ctx(["a", "b"]), ["z"]) == 0.0


def test_precision_and_recall():
    ctx = _ctx(["a", "b", "c"])
    assert M.precision_at_k(ctx, ["a", "b"]) == 2 / 3
    assert M.recall_at_k(ctx, ["a", "b"]) == 1.0
    assert M.recall_at_k(ctx, ["a", "x"]) == 0.5


def test_mrr_rewards_top_rank():
    assert M.mrr(_ctx(["rel", "x", "y"]), ["rel"]) == 1.0
    assert M.mrr(_ctx(["x", "rel", "y"]), ["rel"]) == 0.5
    assert M.mrr(_ctx(["x", "y"]), ["rel"]) == 0.0
