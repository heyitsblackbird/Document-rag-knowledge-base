import asyncio

from langchain_core.runnables import RunnableLambda

from services import generation_service
from services.generation_service import build_context, generate_answer


class _FakeMessage:
    def __init__(self, content):
        self.content = content


def test_build_context_includes_page_when_present():
    chunks = [{"source": "doc.pdf", "chunk_index": 3, "page": 12, "text": "some text"}]

    context = build_context(chunks)

    assert "[Source 1] File: doc.pdf, Chunk: 3, Page: 12" in context
    assert "some text" in context


def test_build_context_omits_page_for_txt_sources():
    chunks = [{"source": "notes.txt", "chunk_index": 0, "page": None, "text": "some text"}]

    context = build_context(chunks)

    assert "Page:" not in context


def test_generate_answer_refuses_when_no_chunks_retrieved():
    result = asyncio.run(generate_answer("question", []))

    assert result["answer"] == "I don't have enough evidence to answer that."
    assert result["citations"] == []


def test_generate_answer_returns_grounded_answer_with_citations(monkeypatch):
    fake_model = RunnableLambda(lambda prompt_value: _FakeMessage("Habits compound over time. [Source 1]"))
    monkeypatch.setattr(generation_service, "_get_model", lambda: fake_model)

    chunks = [{"source": "atomic-habits.pdf", "chunk_index": 42, "page": 12, "text": "Habits are..."}]

    result = asyncio.run(generate_answer("How do habits compound?", chunks))

    assert result["answer"] == "Habits compound over time. [Source 1]"
    assert result["citations"] == [
        {
            "source_id": "Source 1",
            "source": "atomic-habits.pdf",
            "chunk_index": 42,
            "page": 12,
            "text": "Habits are...",
        }
    ]
