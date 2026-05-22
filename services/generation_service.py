from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from core.config import GEMINI_API_KEY


if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

model = ChatGoogleGenerativeAI(
    api_key=SecretStr(GEMINI_API_KEY),
    model="gemini-3-flash-preview",
    temperature=0,
)


def build_context(chunks: list[dict]) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        chunk_index = chunk.get("chunk_index", "unknown")
        text = chunk.get("text", "")

        context_parts.append(
            f"[Source {i}] File: {source}, Chunk: {chunk_index}\n{text}"
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

    chain = prompt | model

    response = await chain.ainvoke({
        "context": context,
        "question": question,
    })

    citations = [
        {
            "source_id": f"Source {i}",
            "source": chunk.get("source"),
            "chunk_index": chunk.get("chunk_index"),
            "text": chunk.get("text"),
        }
        for i, chunk in enumerate(chunks, start=1)
    ]

    return {
        "answer": extract_response_content(response.content),
        "citations": citations,
    }