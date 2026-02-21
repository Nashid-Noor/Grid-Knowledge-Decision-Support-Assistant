"""Document ingestion: loading, chunking, and metadata extraction."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkMetadata:
    source_file: str
    page: Optional[int]
    section: Optional[str]
    chunk_index: int
    start_char: int
    end_char: int


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    token_estimate: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_estimate", len(self.text.split()))

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "text": self.text, "metadata": asdict(self.metadata)}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> list[tuple[str, int]]:
    """Return list of (page_text, page_number) tuples."""
    reader = PdfReader(str(path))
    pages: list[tuple[str, int]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, i + 1))
    return pages


def _load_text(path: Path) -> list[tuple[str, int]]:
    """Return entire text file as a single 'page'."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return [(text, 1)] if text.strip() else []


LOADERS = {
    ".pdf": _load_pdf,
    ".txt": _load_text,
    ".md": _load_text,
    ".rst": _load_text,
}


# ---------------------------------------------------------------------------
# Section detection (best-effort heading extraction)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r"^(?:#{1,4}\s+|[A-Z][A-Z0-9 /&\-]{4,80}$|(?:\d+\.)+\s+[A-Z])",
    re.MULTILINE,
)


def _detect_section(text: str) -> Optional[str]:
    """Return the last detected heading-like line before text, or None."""
    matches = list(_HEADING_RE.finditer(text))
    if matches:
        return matches[-1].group(0).strip().rstrip("#").strip()
    return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _deterministic_id(source: str, page: int, chunk_index: int, text: str) -> str:
    """SHA-256-based deterministic chunk ID."""
    raw = f"{source}|{page}|{chunk_index}|{text[:200]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[tuple[str, int, int]]:
    """Split text into overlapping word-boundary chunks. Returns (chunk, start_char, end_char)."""
    words = text.split()
    chunks: list[tuple[str, int, int]] = []
    i = 0
    while i < len(words):
        window = words[i : i + chunk_size]
        chunk_text = " ".join(window)
        start_char = text.find(window[0], sum(len(w) + 1 for w in words[:i])) if window else 0
        end_char = start_char + len(chunk_text)
        chunks.append((chunk_text, start_char, end_char))
        i += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_document(
    path: str | Path,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[DocumentChunk]:
    """Ingest a single document and return its chunks."""
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    pages = loader(path)
    chunks: list[DocumentChunk] = []
    global_idx = 0

    for page_text, page_num in pages:
        for chunk_text, start, end in _chunk_text(page_text, chunk_size, overlap):
            section = _detect_section(chunk_text)
            meta = ChunkMetadata(
                source_file=path.name,
                page=page_num,
                section=section,
                chunk_index=global_idx,
                start_char=start,
                end_char=end,
            )
            cid = _deterministic_id(path.name, page_num, global_idx, chunk_text)
            chunks.append(DocumentChunk(chunk_id=cid, text=chunk_text, metadata=meta))
            global_idx += 1

    return chunks


def ingest_directory(
    directory: str | Path = "data/docs",
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[DocumentChunk]:
    """Ingest all supported documents from a directory."""
    directory = Path(directory)
    if not directory.exists():
        return []

    all_chunks: list[DocumentChunk] = []
    for fpath in sorted(directory.iterdir()):
        if fpath.suffix.lower() in LOADERS and not fpath.name.startswith("."):
            all_chunks.extend(ingest_document(fpath, chunk_size, overlap))
    return all_chunks


def save_chunks_manifest(chunks: list[DocumentChunk], path: str | Path = "data/cache/chunks.json") -> None:
    """Persist chunk metadata to JSON for reuse without reprocessing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([c.to_dict() for c in chunks], f, indent=2)


def load_chunks_manifest(path: str | Path = "data/cache/chunks.json") -> list[DocumentChunk]:
    """Load previously saved chunks from JSON manifest."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        records = json.load(f)
    chunks: list[DocumentChunk] = []
    for r in records:
        meta = ChunkMetadata(**r["metadata"])
        chunks.append(DocumentChunk(chunk_id=r["chunk_id"], text=r["text"], metadata=meta))
    return chunks
