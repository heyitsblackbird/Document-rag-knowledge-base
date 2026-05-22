import asyncio

from services.retrieval_service import retrieve_relevant_chunks
from services.generation_service import generate_answer


async def main():
    question = "What is this document about?"

    chunks = retrieve_relevant_chunks(question, top_k=3)
    result = await generate_answer(question, chunks)

    print("ANSWER:")
    print(result["answer"])

    print("\nCITATIONS:")
    for citation in result["citations"]:
        print(citation["source_id"], citation["source"], citation["chunk_index"])


asyncio.run(main())