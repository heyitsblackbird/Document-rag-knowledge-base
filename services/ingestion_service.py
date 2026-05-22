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


def _extract_pdf_metadata(file_path: str) -> str:
    '''
    Extract metadata from a PDF document.

    Args:
        file_path (str): Path to the PDF document.
    Returns:
        str: Extracted metadata as a string.
    '''

    reader = PdfReader(file_path)
    pages_text = []

    for page_number, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text.strip():
                pages_text.append(f"\n\n--- Page {page_number + 1} ---\n{text}")
        except Exception as e:
            print(f"Error extracting text from page {page_number}: {e}")

    return _cleaned_data('\n'.join(pages_text))


def _extract_text_metadata(file_path: str) -> str:
    '''
    Extract metadata from a text-based document (e.g., .docx, .txt).

    Args:
        file_path (str): Path to the text-based document.
    Returns:
        str: Extracted metadata as a string.
    '''

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            return _cleaned_data(text)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return "" 

def ingest_document(file_path: str) -> dict:
    '''
    Main function to ingest documents and extract metadata.

    Args:
        file_path (str): Path to the document to be ingested.
    Returns:
        dict: A dictionary containing the extracted metadata.
    '''

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix.lower() == '.pdf':
        cleaned_metadata = _extract_pdf_metadata(file_path)
    elif path.suffix.lower() in ['.docx', '.doc', '.txt']:
        cleaned_metadata = _extract_text_metadata(file_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    return {
        'source': path.name,
        "file-type": path.suffix.lower(),
        'text': cleaned_metadata
    }
