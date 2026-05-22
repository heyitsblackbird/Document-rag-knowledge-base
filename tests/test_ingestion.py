'''
Test cases for ingestion module.
'''

from services.ingestion_service import ingest_document

doc = ingest_document('data/uploads/sample.pdf')
print(doc['source'])
print(doc['text'][:500])  # Print the first 500 characters of the metadata