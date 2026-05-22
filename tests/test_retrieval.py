from services.retrieval_service import retrieve_relevant_chunks


question = "What is this document about?"

chunks = retrieve_relevant_chunks(question, top_k=3)

print("Retrieved chunks:", len(chunks))

for chunk in chunks:
    print("\n--- CHUNK ---")
    print("Source:", chunk["source"])
    print("Chunk index:", chunk["chunk_index"])
    print("Distance:", chunk["distance"])
    print(chunk["text"][:500])