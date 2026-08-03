from services import retrieval_service


class _FakeModel:
    def encode(self, text):
        class _Array(list):
            def tolist(self):
                return list(self)

        return _Array([0.1, 0.2, 0.3])


class _FakeCollection:
    def __init__(self, documents, metadatas, distances):
        self._documents = documents
        self._metadatas = metadatas
        self._distances = distances

    def query(self, query_embeddings, n_results, include):
        return {
            "documents": [self._documents[:n_results]],
            "metadatas": [self._metadatas[:n_results]],
            "distances": [self._distances[:n_results]],
        }


def test_retrieve_relevant_chunks_maps_metadata(monkeypatch):
    monkeypatch.setattr(retrieval_service, "_get_model", lambda: _FakeModel())
    monkeypatch.setattr(
        retrieval_service,
        "_get_collection",
        lambda: _FakeCollection(
            documents=["chunk text"],
            metadatas=[{"source": "doc.pdf", "chunk_index": 2, "page": 5}],
            distances=[0.12],
        ),
    )

    results = retrieval_service.retrieve_relevant_chunks("a question", top_k=1)

    assert results == [
        {"text": "chunk text", "source": "doc.pdf", "chunk_index": 2, "page": 5, "distance": 0.12}
    ]


def test_hybrid_retrieval_returns_empty_for_blank_question():
    assert retrieval_service.hybrid_retrieval("   ") == []


def test_hybrid_retrieval_boosts_chunks_found_by_both_methods(monkeypatch):
    vector_only = {"text": "v", "source": "doc.pdf", "chunk_index": 0, "page": 1}
    bm25_only = {"text": "b", "source": "doc.pdf", "chunk_index": 1, "page": 1, "bm25_score": 1.0}
    found_by_both = {"text": "both", "source": "doc.pdf", "chunk_index": 2, "page": 1}

    monkeypatch.setattr(
        retrieval_service,
        "retrieve_relevant_chunks",
        lambda question, top_k: [found_by_both, vector_only],
    )
    monkeypatch.setattr(
        retrieval_service,
        "bm25_search",
        lambda question, top_k: [{**found_by_both, "bm25_score": 1.0}, bm25_only],
    )

    results = retrieval_service.hybrid_retrieval("question", top_k=5, vector_k=5, bm25_k=5)

    assert results[0]["chunk_index"] == 2
    assert results[0]["vector_rank"] == 1
    assert results[0]["bm25_rank"] == 1
    assert results[0]["hybrid_score"] == 2.0
