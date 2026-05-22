import re
import chromadb
from rank_bm25 import BM25Okapi


CHROMA_PATH = "data/chromaDB"
COLLECTION_NAME = "documents"

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def get_all_chunks() -> list[dict]:
    results = _collection.get(
        include=["documents", "metadatas"]
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    ids = results.get("ids") or []

    chunks = []

    for i, document in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) and metadatas[i] else {}

        chunks.append({
            "id": ids[i],
            "text": document,
            "source": metadata.get("source"),
            "chunk_index": metadata.get("chunk_index"),
            "page": metadata.get("page"),
        })

    return chunks


def bm25_search(query: str, top_k: int = 5) -> list[dict]:
    chunks = get_all_chunks()

    if not chunks:
        return []

    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []

    for chunk, score in ranked[:top_k]:
        results.append({
            **chunk,
            "bm25_score": float(score),
        })

    return results