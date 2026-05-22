'''
Handles chunking of documents for better processing and retrieval.
'''

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk__document(text: str, source_name: str, chunk_size: int = 800, chunk_overlap: int = 100):
    """
    Chunks a document into smaller pieces for better processing and retrieval.

    Args:
        text (str): The text to be chunked.
        source_name (str): The name of the source document.
        chunk_size (int): The size of each chunk in characters. Default is 800.
        chunk_overlap (int): The number of characters to overlap between chunks. Default is 100.
    Returns:
        List[dict]: A list of dictionaries, each containing a chunk of text and its metadata.
    """

    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)

    return [{
        "chunk_id": f"{source_name}_chunk_{idx}",
        "source": source_name,
        "index": idx,
        "text": chunk,
    } for idx, chunk in enumerate(chunks)]