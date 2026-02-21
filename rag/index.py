"""FAISS index management with sentence-transformer embeddings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.ingest import DocumentChunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = Path("data/cache/faiss.index")
ID_MAP_PATH = Path("data/cache/id_map.json")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dim


# ---------------------------------------------------------------------------
# Singleton encoder
# ---------------------------------------------------------------------------

_encoder: Optional[SentenceTransformer] = None


def get_encoder(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(model_name)
    return _encoder


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """Encode a list of strings into a float32 numpy matrix."""
    encoder = get_encoder(model_name)
    embeddings = encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Index build / save / load
# ---------------------------------------------------------------------------

def build_index(chunks: list[DocumentChunk], model_name: str = DEFAULT_MODEL_NAME) -> tuple[faiss.IndexFlatIP, list[str]]:
    """Build a FAISS inner-product index from document chunks.

    Returns the index and an ordered list of chunk_ids that map FAISS
    integer positions back to chunk identifiers.
    """
    if not chunks:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        return index, []

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts, model_name)
    faiss.normalize_L2(embeddings)  # normalise so IP == cosine similarity

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    id_map = [c.chunk_id for c in chunks]
    return index, id_map


def save_index(index: faiss.IndexFlatIP, id_map: list[str]) -> None:
    """Persist FAISS index and ID mapping to disk."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(ID_MAP_PATH, "w") as f:
        json.dump(id_map, f)


def load_index() -> tuple[Optional[faiss.IndexFlatIP], list[str]]:
    """Load a previously saved FAISS index. Returns (None, []) if absent."""
    if not INDEX_PATH.exists() or not ID_MAP_PATH.exists():
        return None, []
    index = faiss.read_index(str(INDEX_PATH))
    with open(ID_MAP_PATH) as f:
        id_map = json.load(f)
    return index, id_map


def index_ready() -> bool:
    """Check whether a cached index exists on disk."""
    return INDEX_PATH.exists() and ID_MAP_PATH.exists()
