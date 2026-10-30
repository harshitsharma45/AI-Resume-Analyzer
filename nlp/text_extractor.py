from io import BytesIO
from pathlib import Path

def extract_text_from_file(file):
    """Accept a Streamlit UploadedFile or file path. Supports PDF, DOCX and TXT."""
    if isinstance(file, (str, Path)):
        name = Path(file).name
        raw = Path(file).read_bytes()
    else:
        name = file.name
        raw = file.getvalue()
    ext = Path(name).suffix.lower()
    if ext == ".txt":
        return raw.decode("utf-8", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(BytesIO(raw)).paragraphs)
    raise ValueError("Supported formats: PDF, DOCX, TXT")
