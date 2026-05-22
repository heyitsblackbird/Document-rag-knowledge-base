'''
Handles Embedding generation and storage using ChromaDB.
'''

from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_PATH = 'data/chromaDB'
COLLECTION_NAME = 'documents'

_model = SentenceTransformer('all-MiniLM-L6-v2')
_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)

def generate_embeddings(text:str) -> list:
    '''
    Generate embeddings for the given text using SentenceTransformer.
    '''
    return _model.encode(text).tolist()

def store_embeddings(chunks_text:list)-> int:
    '''
    Store the embedding in ChromaDB and return total number of documents.
    '''
    if not chunks_text:
        raise ValueError("Chunks text cannot be empty.")
    
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for chunk in chunks_text:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        embeddings.append(generate_embeddings(chunk["text"]))

        metadatas.append({
            "source": chunk["source"],
            "chunk_index": chunk["index"]
        })
    
    _collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(_collection.get(ids=ids)['ids'])