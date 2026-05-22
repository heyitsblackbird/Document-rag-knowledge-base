from services.bm25_retrieval_service import bm25_search

query = "atomic habits"

results = bm25_search(query, top_k=5)

print("BM25 results:", len(results))

for result in results:
    print("\n--- RESULT ---")
    print("Source:", result["source"])
    print("Page:", result.get("page"))
    print("Chunk:", result["chunk_index"])
    print("Score:", result["bm25_score"])
    print(result["text"][:300])