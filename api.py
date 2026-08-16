'''
FastAPI service exposing document indexing and citation-grounded question answering.
'''

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from services.chunking_service import chunk_document
from services.embedding_service import store_embeddings
from services.generation_service import generate_answer
from services.ingestion_service import ingest_document
from services.metrics_service import queries_total
from services.reranker_service import rerank_chunks
from services.retrieval_service import hybrid_retrieval

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx"}

app = FastAPI(title="RAG Learning Assistant API")

# Exposes /metrics with HTTP-level request count, latency, and status codes.
Instrumentator().instrument(app).expose(app)


class IndexResponse(BaseModel):
    source: str
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class Citation(BaseModel):
    source_id: str
    source: str | None
    chunk_index: int | None
    page: int | None
    text: str | None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=IndexResponse)
async def upload_document(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        doc = ingest_document(tmp_path)
        chunks = chunk_document(doc["pages"], file.filename)
        if not chunks:
            raise HTTPException(status_code=422, detail="No extractable text found in document")
        count = store_embeddings(chunks)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IndexResponse(source=file.filename, chunks_indexed=count)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question.strip():
        queries_total.labels(outcome="bad_request").inc()
        raise HTTPException(status_code=400, detail="Question must not be empty")

    candidates = hybrid_retrieval(request.question, top_k=10)
    reranked = rerank_chunks(request.question, candidates, top_k=request.top_k)

    try:
        result = await generate_answer(request.question, reranked)
    except ValueError as e:
        queries_total.labels(outcome="generation_unavailable").inc()
        raise HTTPException(status_code=503, detail=str(e)) from e

    queries_total.labels(outcome="success").inc()
    return QueryResponse(**result)
