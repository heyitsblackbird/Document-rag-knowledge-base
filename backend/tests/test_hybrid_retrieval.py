from app.services.retrieval_service import hybrid_retrieval

question = "how do habits compunds over time?"

results = hybrid_retrieval(question, top_k=5)

print("Hybrid results:", len(results))

for result in results:
    print("\n--- RESULT ---")
    print("Source:", result.get("source"))
    print("Page:", result.get("page"))
    print("Chunk:", result.get("chunk_index"))
    print("Vector rank:", result.get("vector_rank"))
    print("BM25 rank:", result.get("bm25_rank"))
    print("Hybrid score:", result.get("hybrid_score"))
    print(result.get("text", "")[:300])