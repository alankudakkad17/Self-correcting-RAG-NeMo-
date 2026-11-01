"""Turn an uploaded file into plain text, then into overlapping chunks for embedding."""

import io
import re
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}
MAX_BYTES = 20 * 1024 * 1024  # 20 MB per file


class UnsupportedFile(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    """Plain text of a .txt, .md or .pdf file. Raises UnsupportedFile for anything else or an empty result."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedFile(f"{filename}: only {', '.join(sorted(SUPPORTED))} files are supported")
    if len(data) > MAX_BYTES:
        raise UnsupportedFile(f"{filename}: larger than {MAX_BYTES // (1024 * 1024)} MB")
    if suffix == ".pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:  # malformed or encrypted PDF
            raise UnsupportedFile(f"{filename}: could not read the PDF ({type(e).__name__})") from e
    else:
        text = data.decode("utf-8", errors="replace")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        raise UnsupportedFile(f"{filename}: no text found (a scanned PDF needs OCR first)")
    return text


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split into ~size-character chunks on sentence boundaries, each starting with the last ~overlap characters
    of the previous chunk so facts spanning a boundary stay findable."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        while len(sentence) > size:  # a single over-long "sentence" (tables, no punctuation): hard split
            head, sentence = sentence[:size], sentence[size:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if current and len(current) + 1 + len(sentence) > size:
            chunks.append(current)
            tail = current[-overlap:]
            current = tail[tail.find(" ") + 1:] if " " in tail else tail  # start the overlap at a word
        current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks
