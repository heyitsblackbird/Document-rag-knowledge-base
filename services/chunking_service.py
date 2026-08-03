'''
Handles chunking of documents for better processing and retrieval.
'''

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document(pages: list[dict], source_name: str, chunk_size: int = 800, chunk_overlap: int = 100):
    """
    Chunks a document into smaller pieces for better processing and retrieval,
    preserving the source page number for each chunk.

    Args:
        pages (list[dict]): Per-page text, e.g. [{"page": 1, "text": "..."}].
        source_name (str): The name of the source document.
        chunk_size (int): The size of each chunk in characters. Default is 800.
        chunk_overlap (int): The number of characters to overlap between chunks. Default is 100.
    Returns:
        List[dict]: A list of dictionaries, each containing a chunk of text and its metadata.
    """

    if not pages:
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    result = []
    idx = 0

    for page in pages:
        page_chunks = text_splitter.split_text(page["text"])
        for chunk in page_chunks:
            result.append({
                "chunk_id": f"{source_name}_chunk_{idx}",
                "source": source_name,
                "index": idx,
                "page": page.get("page"),
                "text": chunk,
            })
            idx += 1

    return result