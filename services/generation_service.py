import re
import time

from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from core.config import GEMINI_API_KEY
from services.metrics_service import (
    generation_latency_seconds,
    generation_errors_total,
    chunks_retrieved_total,
    chunks_cited_total,
)

_model = None

def _get_model() -> ChatGoogleGenerativeAI:
    global _model
    if _model is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        _model = ChatGoogleGenerativeAI(
            api_key=SecretStr(GEMINI_API_KEY),
            model="gemini-3-flash-preview",
            temperature=0,
        )
    return _model


def build_context(chunks: list[dict]) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        chunk_index = chunk.get("chunk_index", "unknown")
        page = chunk.get("page")
        text = chunk.get("text", "")

        page_label = f", Page: {page}" if page not in (None, -1) else ""
        context_parts.append(
            f"[Source {i}] File: {source}, Chunk: {chunk_index}{page_label}\n{text}"
        )

    return "\n\n".join(context_parts)

def extract_response_content(response) -> str:
    if isinstance(response, str):
        return response
    
    if isinstance(response, list):
        text_parts = []

        for item in response:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
        return " ".join(text_parts).strip()
    
    return str(response).strip()

async def generate_answer(question: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "answer": "I don't have enough evidence to answer that.",
            "citations": [],
        }

    chunks_retrieved_total.inc(len(chunks))

    context = build_context(chunks)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a citation-based RAG assistant.

Rules:
- Answer ONLY using the provided context.
- Do not use outside knowledge.
- If the context is not enough, say: "I don't have enough evidence to answer that."
- Cite evidence using [Source 1], [Source 2], etc.
- Keep the answer clear and concise.
"""
        ),
        (
            "user",
            """
Context:
{context}

Question:
{question}
"""
        )
    ])

    try:
        chain = prompt | _get_model()
    except ValueError as e:
        generation_errors_total.labels(reason="config_error").inc()
        raise

    start = time.perf_counter()
    try:
        response = await chain.ainvoke({
            "context": context,
            "question": question,
        })
    except Exception:
        generation_errors_total.labels(reason="llm_call_failed").inc()
        raise
    finally:
        generation_latency_seconds.observe(time.perf_counter() - start)

    answer_text = extract_response_content(response.content)
    cited_sources = set(re.findall(r"\[Source (\d+)\]", answer_text))
    chunks_cited_total.inc(len(cited_sources))

    citations = [
        {
            "source_id": f"Source {i}",
            "source": chunk.get("source"),
            "chunk_index": chunk.get("chunk_index"),
            "page": chunk.get("page") if chunk.get("page") not in (None, -1) else None,
            "text": chunk.get("text"),
        }
        for i, chunk in enumerate(chunks, start=1)
    ]

    return {
        "answer": answer_text,
        "citations": citations,
    }