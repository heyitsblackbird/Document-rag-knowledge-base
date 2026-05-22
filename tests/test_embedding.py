'''
Test cases for embedding module.
'''

from services.ingestion_service import ingest_document
from services.chunking_service import chunk__document
from services.embedding_service import store_embeddings

doc = ingest_document('data/uploads/sample.pdf')
chunks = chunk__document(doc['text'], doc['source'])


total_embeddings = store_embeddings(chunks)
print(f"Total embeddings stored: {total_embeddings}")