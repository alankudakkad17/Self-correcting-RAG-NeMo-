"""Uploaded documents: text extraction, chunking, and the ChromaDB store (real Chroma in a temp folder, fake
embedder). No network."""

import hashlib
import re

import numpy as np
import pytest

from src.uploads.extract import MAX_BYTES
from src.uploads.extract import UnsupportedFile
from src.uploads.extract import chunk_text
from src.uploads.extract import extract_text
from src.uploads.store import UploadStore


def minimal_pdf(text: str) -> bytes:
    """A valid one-page PDF showing `text` (xref offsets computed, so strict readers accept it)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
               b"/Resources << /Font << /F1 5 0 R >> >> >>",
               b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_extract_text_and_markdown():
    assert extract_text("notes.txt", b"Hello   world.\tAgain.") == "Hello world. Again."
    assert extract_text("README.md", "# Title\n\nCafé".encode()) == "# Title\n\nCafé"


def test_extract_pdf():
    assert "Quarterly revenue grew" in extract_text("report.pdf", minimal_pdf("Quarterly revenue grew 12 percent"))


@pytest.mark.parametrize("name, data, message", [
    ("slides.pptx", b"x", "only"),
    ("empty.txt", b"   ", "no text"),
    ("broken.pdf", b"%PDF-1.4 not really", "could not read"),
], ids=["unsupported-type", "empty", "broken-pdf"])
def test_rejected_files_have_clear_reasons(name, data, message):
    with pytest.raises(UnsupportedFile, match=message):
        extract_text(name, data)


def test_oversized_file_is_rejected():
    # built here, not as a parameter: pytest puts parameters into the test id (an environment variable)
    with pytest.raises(UnsupportedFile, match="larger than"):
        extract_text("big.txt", b"a" * (MAX_BYTES + 1))


def test_chunks_respect_size_overlap_and_cover_the_text():
    sentences = [f"Sentence number {i} says something about topic {i}." for i in range(60)]
    chunks = chunk_text(" ".join(sentences), size=300, overlap=60)
    assert len(chunks) > 5 and all(len(c) <= 300 + 60 for c in chunks)
    assert all(s in " ".join(chunks) for s in sentences)  # nothing lost
    assert chunks[1].split()[0] in chunks[0]  # the next chunk starts inside the previous one (overlap)


def test_overlong_unpunctuated_text_is_hard_split():
    chunks = chunk_text("x" * 2000, size=800)
    assert [len(c) for c in chunks] == [800, 800, 400]


class FakeEmbedder:
    """Deterministic bag-of-words vectors: texts sharing words are close. Normalised like the real embedder."""

    model = "fake"

    async def embed(self, texts, input_type):
        vecs = np.zeros((len(texts), 64), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                vecs[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


@pytest.fixture
def store(tmp_path):
    return UploadStore(FakeEmbedder(), path=tmp_path / "chroma")


async def test_ingest_search_list_delete(store):
    text = ("The Falcon project budget is 4 million dollars. " * 3 + "\n\n"
            + "The Heron project launched in March. " * 3).encode()
    added = await store.ingest("plans.txt", text)
    assert added["status"] == "added" and added["doc_id"].startswith("up_") and added["chunks"] >= 1

    hits = await store.search("Falcon budget", k=2)
    assert hits and "Falcon" in hits[0].text
    assert hits[0].doc_id.startswith(added["doc_id"]) and hits[0].title.startswith("plans.txt (part")

    again = await store.ingest("copy-of-plans.txt", text)  # same content: no duplicate chunks
    assert again["status"] == "already uploaded" and again["doc_id"] == added["doc_id"]
    assert [d["filename"] for d in store.list_documents()] == ["plans.txt"]

    assert store.delete(added["doc_id"]) is True
    assert store.list_documents() == [] and await store.search("Falcon", k=2) == []
    assert store.delete(added["doc_id"]) is False


async def test_store_persists_across_instances(tmp_path):
    first = UploadStore(FakeEmbedder(), path=tmp_path / "db")
    await first.ingest("a.md", b"Persistent facts live here.")
    second = UploadStore(FakeEmbedder(), path=tmp_path / "db")
    assert [d["filename"] for d in second.list_documents()] == ["a.md"]
