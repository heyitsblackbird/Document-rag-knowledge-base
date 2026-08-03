import pytest

from services import embedding_service


class _FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)

        class _Array(list):
            def tolist(self):
                return list(self)

        return _Array([[0.1, 0.2] for _ in texts])


class _FakeCollection:
    def __init__(self):
        self.upserted = None

    def upsert(self, ids, documents, embeddings, metadatas):
        self.upserted = {"ids": ids, "documents": documents, "embeddings": embeddings, "metadatas": metadatas}


def test_store_embeddings_raises_on_empty_input():
    with pytest.raises(ValueError):
        embedding_service.store_embeddings([])


def test_store_embeddings_batches_a_single_model_call(monkeypatch):
    fake_model = _FakeModel()
    fake_collection = _FakeCollection()
    monkeypatch.setattr(embedding_service, "_get_model", lambda: fake_model)
    monkeypatch.setattr(embedding_service, "_get_collection", lambda: fake_collection)

    chunks = [
        {"chunk_id": "doc_chunk_0", "source": "doc.pdf", "index": 0, "page": 1, "text": "first"},
        {"chunk_id": "doc_chunk_1", "source": "doc.pdf", "index": 1, "page": 2, "text": "second"},
    ]

    count = embedding_service.store_embeddings(chunks)

    assert count == 2
    assert len(fake_model.calls) == 1
    assert fake_model.calls[0] == ["first", "second"]
    assert fake_collection.upserted["ids"] == ["doc_chunk_0", "doc_chunk_1"]


def test_store_embeddings_uses_sentinel_for_missing_page(monkeypatch):
    fake_model = _FakeModel()
    fake_collection = _FakeCollection()
    monkeypatch.setattr(embedding_service, "_get_model", lambda: fake_model)
    monkeypatch.setattr(embedding_service, "_get_collection", lambda: fake_collection)

    chunks = [{"chunk_id": "notes.txt_chunk_0", "source": "notes.txt", "index": 0, "page": None, "text": "hi"}]

    embedding_service.store_embeddings(chunks)

    assert fake_collection.upserted["metadatas"][0]["page"] == -1
