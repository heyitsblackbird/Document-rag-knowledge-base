from services.retrieval_service import hybrid_retrieval
from services.reranker_service import rerank_chunks

question = "How do habits compound over time?"

candidates = hybrid_retrieval(
    question=question,
    top_k=10,
)

reranked = rerank_chunks(
    question=question,
    chunks=candidates,
    top_k=5,
)

for result in reranked:
    print("\n--- RESULT ---")
    print("Source:", result.get("source"))
    print("Page:", result.get("page"))
    print("Chunk:", result.get("chunk_index"))
    print("Hybrid score:", result.get("hybrid_score"))
    print("Rerank score:", result.get("rerank_score"))
    print(result.get("text", "")[:300])