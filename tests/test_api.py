import io

from fastapi.testclient import TestClient

import api
from api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_document_rejects_unsupported_type():
    response = client.post(
        "/documents", files={"file": ("notes.exe", io.BytesIO(b"junk"), "application/octet-stream")}
    )

    assert response.status_code == 400


def test_upload_document_indexes_and_returns_chunk_count(monkeypatch):
    monkeypatch.setattr(api, "ingest_document", lambda path: {"source": "notes.txt", "file-type": ".txt", "pages": [{"page": None, "text": "hello"}]})
    monkeypatch.setattr(api, "chunk_document", lambda pages, source: [{"chunk_id": f"{source}_chunk_0", "source": source, "index": 0, "page": None, "text": "hello"}])
    monkeypatch.setattr(api, "store_embeddings", lambda chunks: len(chunks))

    response = client.post(
        "/documents", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    )

    assert response.status_code == 200
    assert response.json() == {"source": "notes.txt", "chunks_indexed": 1}


def test_upload_document_rejects_file_with_no_extractable_text(monkeypatch):
    monkeypatch.setattr(api, "ingest_document", lambda path: {"source": "blank.txt", "file-type": ".txt", "pages": []})
    monkeypatch.setattr(api, "chunk_document", lambda pages, source: [])

    response = client.post(
        "/documents", files={"file": ("blank.txt", io.BytesIO(b""), "text/plain")}
    )

    assert response.status_code == 422


def test_query_rejects_blank_question():
    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 400


def test_query_returns_answer_and_citations(monkeypatch):
    monkeypatch.setattr(api, "hybrid_retrieval", lambda question, top_k: [{"text": "chunk", "source": "doc.pdf", "chunk_index": 0, "page": 1}])
    monkeypatch.setattr(api, "rerank_chunks", lambda question, chunks, top_k: chunks)

    async def fake_generate_answer(question, chunks):
        return {
            "answer": "Habits compound over time. [Source 1]",
            "citations": [
                {"source_id": "Source 1", "source": "doc.pdf", "chunk_index": 0, "page": 1, "text": "chunk"}
            ],
        }

    monkeypatch.setattr(api, "generate_answer", fake_generate_answer)

    response = client.post("/query", json={"question": "How do habits compound?", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Habits compound over time. [Source 1]"
    assert body["citations"][0]["source"] == "doc.pdf"
