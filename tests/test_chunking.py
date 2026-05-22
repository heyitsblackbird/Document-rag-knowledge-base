'''
Test cases for chunking module.
'''

from services.ingestion_service import ingest_document
from services.chunking_service import chunk__document

doc = ingest_document('data/uploads/sample.pdf')
chunks = chunk__document(doc['text'], doc['source'])
print(f"Total chunks created: {len(chunks)}")
print(f"First chunk: {chunks[0]}")