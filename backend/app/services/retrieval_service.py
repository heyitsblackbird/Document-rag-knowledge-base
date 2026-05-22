'''
Retrieval service for fetching relevant chunks based on user queries.
'''

from sentence_transformers import SentenceTransformer
import chromadb
from app.services.bm25_retrieval_service import bm25_search

CHROMA_PATH = 'data/chromaDB'
COLLECTION_NAME = 'documents'
_model = SentenceTransformer('all-MiniLM-L6-v2')
_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)

def generate_query_embedding(query:str) -> list:
    '''
    Generate embedding for the user query.
    '''
    return _model.encode(query).tolist()

def retrieve_relevant_chunks(question:str, top_k:int = 5) -> list[dict]:
    '''
    Retrieve relevant chunks from ChromaDB based on the query embedding.
    '''

    query_embedding = generate_query_embedding(question)
    
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )

    retrieved_chunks = []

    documents = results.get('documents', [[]])
    metadatas = results.get('metadatas', [[]])
    distances = results.get('distances', [[]])

    documents = documents[0] if documents else []
    metadatas = metadatas[0] if metadatas else []
    distances = distances[0] if distances else []

    for i in range(len(documents)):
        metadata = metadatas[i]
        distance = distances[i]

        retrieved_chunks.append({
            'text': documents[i],
            'source': metadata.get('source', 'unknown'),
            'chunk_index': metadata.get('chunk_index'),
            'distance': distance,
        })

    return retrieved_chunks

def hybrid_retrieval(question: str, top_k: int = 5, vector_k: int = 5, bm25_k: int = 5) -> list[dict]:
    '''
    Perform hybrid retrieval using both BM25 and embedding-based search.
    '''
    if not question or not question.strip():
        return []
    
    bm25_results = bm25_search(question, top_k=bm25_k)
    embedding_results = retrieve_relevant_chunks(question, top_k=vector_k)

    combined = {}

    # Vector results are keyed by source and chunk index
    for rank, chunk in enumerate(embedding_results, start=1):
        key = (chunk['source'], chunk['chunk_index'])
        combined[key] = {
            'text': chunk['text'],
            'source': chunk['source'],
            'chunk_index': chunk['chunk_index'],
            'vector_rank': rank,
            'bm25_rank': None,
            'hybrid_score': 1/ (rank),
        }
    
    # BM25 results are keyed by source and chunk index
    for rank, chunk in enumerate(bm25_results, start=1):
        key = (chunk['source'], chunk['chunk_index'])
        if key in combined:
            combined[key]['bm25_rank'] = rank
            combined[key]['hybrid_score'] += 1/ (rank)
        else:
            combined[key] = {
                'text': chunk['text'],
                'source': chunk['source'],
                'chunk_index': chunk['chunk_index'],
                'vector_rank': None,
                'bm25_rank': rank,
                'hybrid_score': 1/ (rank),
            }
    
    ranked = sorted(
        combined.values(),
        key=lambda x: x['hybrid_score'],
        reverse=True
    )
    
    return ranked[:top_k]