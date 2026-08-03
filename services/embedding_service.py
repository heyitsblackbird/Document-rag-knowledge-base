'''
Handles Embedding generation and storage using ChromaDB.
'''

from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_PATH = 'data/chromaDB'
COLLECTION_NAME = 'documents'

_model = None
_collection = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    '''
    Generate embeddings for a batch of texts using SentenceTransformer.
    '''
    return _get_model().encode(texts).tolist()

def store_embeddings(chunks_text:list)-> int:
    '''
    Store the embedding in ChromaDB and return total number of documents.
    '''
    if not chunks_text:
        raise ValueError("Chunks text cannot be empty.")

    ids = [chunk["chunk_id"] for chunk in chunks_text]
    documents = [chunk["text"] for chunk in chunks_text]
    embeddings = generate_embeddings(documents)
    # Chroma metadata values can't be None (e.g. .txt has no page number), so -1 is the sentinel for "no page".
    metadatas = [
        {
            "source": chunk["source"],
            "chunk_index": chunk["index"],
            "page": chunk.get("page") if chunk.get("page") is not None else -1,
        }
        for chunk in chunks_text
    ]

    _get_collection().upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)