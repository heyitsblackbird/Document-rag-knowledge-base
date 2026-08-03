from sentence_transformers import CrossEncoder

_model = None

def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _model

def rerank_chunks(question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks or not question.strip():
        return []

    texts = [chunk.get("text", "") for chunk in chunks]
    pairs = [[question, text] for text in texts]

    scores = _get_model().predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    reranked = [
        {
            **chunk,
            "rerank_score": float(score),
        }
        for chunk, score in ranked
    ]

    return reranked[:top_k]