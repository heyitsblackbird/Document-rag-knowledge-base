from services.chunking_service import chunk_document


def test_empty_pages_returns_empty_list():
    assert chunk_document([], "doc.txt") == []


def test_chunk_ids_and_indices_are_sequential_across_pages():
    pages = [
        {"page": 1, "text": "a" * 900},
        {"page": 2, "text": "b" * 900},
    ]

    chunks = chunk_document(pages, "doc.pdf", chunk_size=800, chunk_overlap=100)

    assert len(chunks) >= 2
    assert [c["index"] for c in chunks] == list(range(len(chunks)))
    assert [c["chunk_id"] for c in chunks] == [f"doc.pdf_chunk_{i}" for i in range(len(chunks))]


def test_chunk_retains_its_source_page():
    pages = [
        {"page": 1, "text": "a" * 900},
        {"page": 2, "text": "b" * 900},
    ]

    chunks = chunk_document(pages, "doc.pdf", chunk_size=800, chunk_overlap=100)

    page_1_chunks = [c for c in chunks if c["page"] == 1]
    page_2_chunks = [c for c in chunks if c["page"] == 2]

    assert page_1_chunks and all("a" in c["text"] for c in page_1_chunks)
    assert page_2_chunks and all("b" in c["text"] for c in page_2_chunks)


def test_txt_source_has_no_page_number():
    pages = [{"page": None, "text": "hello world"}]

    chunks = chunk_document(pages, "notes.txt")

    assert chunks[0]["page"] is None


def test_page_with_short_text_produces_single_chunk():
    pages = [{"page": 1, "text": "short text"}]

    chunks = chunk_document(pages, "doc.pdf")

    assert len(chunks) == 1
    assert chunks[0]["text"] == "short text"
