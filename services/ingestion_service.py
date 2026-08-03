'''
Resposible for ingesting data and provide cleaned metadata for the rest of the application.
'''

from pathlib import Path
from pypdf import PdfReader

def _cleaned_data(text: str) -> str:
    '''
    Clean the extracted text by removing extra whitespace and newlines.

    Args:
        text (str): The raw extracted text.
    Returns:
        str: The cleaned text.
    '''

    if not text:
        return ""
    
    # Replace null characters with spaces
    text = text.replace('\x00',' ')
    

    # Remove extra whitespace and newlines
    text = ' '.join(text.split())

    return text.strip()


def _extract_pdf_pages(file_path: str) -> list[dict]:
    '''
    Extract per-page text from a PDF document, preserving page numbers.

    Args:
        file_path (str): Path to the PDF document.
    Returns:
        list[dict]: One entry per non-empty page: {"page": int, "text": str}.
    '''

    reader = PdfReader(file_path)
    pages = []

    for page_number, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            cleaned = _cleaned_data(text)
            if cleaned:
                pages.append({"page": page_number + 1, "text": cleaned})
        except Exception as e:
            print(f"Error extracting text from page {page_number}: {e}")

    return pages


def _extract_text_pages(file_path: str) -> list[dict]:
    '''
    Extract text from a text-based document (e.g., .docx, .txt) as a single page.

    Args:
        file_path (str): Path to the text-based document.
    Returns:
        list[dict]: A single-entry list: [{"page": None, "text": str}].
    '''

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            cleaned = _cleaned_data(text)
            return [{"page": None, "text": cleaned}] if cleaned else []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def ingest_document(file_path: str) -> dict:
    '''
    Main function to ingest documents and extract per-page text.

    Args:
        file_path (str): Path to the document to be ingested.
    Returns:
        dict: {"source": str, "file-type": str, "pages": list[dict]}.
    '''

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == '.pdf':
        pages = _extract_pdf_pages(file_path)
    elif path.suffix.lower() in ['.docx', '.doc', '.txt']:
        pages = _extract_text_pages(file_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return {
        'source': path.name,
        "file-type": path.suffix.lower(),
        'pages': pages
    }
