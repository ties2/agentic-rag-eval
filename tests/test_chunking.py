from config.settings import Settings
from ragkit.domain.models import Document
from ragkit.ml.chunking import chunk_document


def test_chunking_respects_size():
    text = "\n\n".join("Sentence number %d here." % i for i in range(200))
    doc = Document(doc_id="d1", text=text)
    chunks = chunk_document(doc, size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 260 for c in chunks)  # size + slack for word packing
    assert all(c.doc_id == "d1" for c in chunks)
    # chunk ids are unique
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunking_short_doc_single_chunk():
    doc = Document(doc_id="d2", text="A short paragraph.")
    chunks = chunk_document(doc, size=600, overlap=80)
    assert len(chunks) == 1
    assert chunks[0].metadata["position"] == 0
