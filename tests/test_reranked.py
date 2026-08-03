from services import reranker_service
from services.reranker_service import rerank_chunks


class _FakeCrossEncoder:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):
        return self._scores


def test_rerank_chunks_returns_empty_for_no_chunks():
    assert rerank_chunks("question", []) == []


def test_rerank_chunks_returns_empty_for_blank_question():
    assert rerank_chunks("   ", [{"text": "a"}]) == []


def test_rerank_chunks_reorders_by_score_and_respects_top_k(monkeypatch):
    chunks = [
        {"text": "low relevance", "source": "a.pdf"},
        {"text": "high relevance", "source": "b.pdf"},
        {"text": "medium relevance", "source": "c.pdf"},
    ]
    monkeypatch.setattr(reranker_service, "_get_model", lambda: _FakeCrossEncoder([0.1, 0.9, 0.5]))

    reranked = rerank_chunks("question", chunks, top_k=2)

    assert [c["source"] for c in reranked] == ["b.pdf", "c.pdf"]
    assert reranked[0]["rerank_score"] == 0.9
