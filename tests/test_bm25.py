from services import bm25_retrieval_service
from services.bm25_retrieval_service import tokenize, bm25_search


def test_tokenize_lowercases_and_splits_on_word_boundaries():
    assert tokenize("Atomic Habits, 2nd Edition!") == ["atomic", "habits", "2nd", "edition"]


class _FakeCollection:
    def __init__(self, documents, metadatas, ids):
        self._documents = documents
        self._metadatas = metadatas
        self._ids = ids

    def get(self, include=None):
        return {"documents": self._documents, "metadatas": self._metadatas, "ids": self._ids}


def test_bm25_search_returns_empty_list_when_no_chunks_indexed(monkeypatch):
    monkeypatch.setattr(bm25_retrieval_service, "_get_collection", lambda: _FakeCollection([], [], []))

    assert bm25_search("anything") == []


def test_bm25_search_ranks_the_more_relevant_chunk_first(monkeypatch):
    # rank_bm25's classic IDF degenerates to 0 for a 2-document corpus where a term
    # appears in exactly half the docs, so a third unrelated document keeps IDF meaningful.
    documents = [
        "The stock market fluctuates based on investor sentiment.",
        "Atomic habits compound over time to produce remarkable results.",
        "Diversifying a portfolio reduces long-term investment risk.",
    ]
    metadatas = [
        {"source": "wall-street.pdf", "chunk_index": 0, "page": 1},
        {"source": "atomic-habits.pdf", "chunk_index": 5, "page": 12},
        {"source": "wall-street.pdf", "chunk_index": 1, "page": 2},
    ]
    ids = ["wall-street.pdf_chunk_0", "atomic-habits.pdf_chunk_5", "wall-street.pdf_chunk_1"]

    monkeypatch.setattr(
        bm25_retrieval_service, "_get_collection", lambda: _FakeCollection(documents, metadatas, ids)
    )

    results = bm25_search("atomic habits compound", top_k=2)

    assert results[0]["source"] == "atomic-habits.pdf"
    assert results[0]["chunk_index"] == 5
    assert results[0]["page"] == 12
    assert results[0]["bm25_score"] > results[1]["bm25_score"]
