import numpy as np

from kbqa.index.vector_store import VectorStore, Passage
from kbqa.data.chunking import chunk_document, chunk_text
from kbqa.data.samples import SAMPLE_DOCS


def test_vector_store_build_and_search():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(5, 8)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    passages = [Passage(id=f"p{i}", text=f"text {i}") for i in range(5)]
    store = VectorStore().build(vecs, passages)
    # querying with an exact vector should return that passage first
    hits = store.search(vecs[2], top_k=3)
    assert hits[0][0].id == "p2"
    assert len(hits) == 3


def test_vector_store_roundtrip(tmp_path):
    vecs = np.eye(3, dtype="float32")
    passages = [Passage(id=str(i), text=f"t{i}") for i in range(3)]
    VectorStore().build(vecs, passages).save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 3
    assert loaded.search(vecs[0], top_k=1)[0][0].id == "0"


def test_chunking_overlap_and_ids():
    text = " ".join(f"word{i}" for i in range(500))
    passages = chunk_document(text, doc_id="d", title="T", chunk_size=100, overlap=20)
    assert len(passages) > 1
    assert all(p.id.startswith("d::") for p in passages)
    # consecutive chunks overlap
    assert chunk_text(text, 100, 20)[0].split()[-1] in chunk_text(text, 100, 20)[1]


def test_chunk_corpus_on_samples():
    from kbqa.data.chunking import chunk_corpus
    passages = chunk_corpus(SAMPLE_DOCS, chunk_size=80, overlap=16)
    assert len(passages) >= len(SAMPLE_DOCS)
