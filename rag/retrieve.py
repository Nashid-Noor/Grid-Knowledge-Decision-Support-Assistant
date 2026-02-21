"""Retrieval: query the FAISS index and return ranked passages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np

from rag.index import embed_texts, get_encoder, EMBEDDING_DIM
from rag.ingest import DocumentChunk


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RetrievedPassage:
    chunk_id: str
    text: str
    score: float
    source_file: str
    page: Optional[int]
    section: Optional[str]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    id_map: list[str],
    chunks_by_id: dict[str, DocumentChunk],
    top_k: int = 5,
    score_threshold: float = 0.25,
) -> list[RetrievedPassage]:
    """Retrieve the most relevant passages for a query.

    Args:
        query: Natural-language question.
        index: FAISS inner-product index (normalised vectors → cosine sim).
        id_map: Ordered chunk IDs matching FAISS row positions.
        chunks_by_id: Mapping of chunk_id → DocumentChunk for metadata.
        top_k: Maximum passages to return.
        score_threshold: Minimum cosine similarity to include a result.

    Returns:
        List of RetrievedPassage, sorted descending by score.
    """
    if index.ntotal == 0:
        return []

    query_vec = embed_texts([query])
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, min(top_k, index.ntotal))

    results: list[RetrievedPassage] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < score_threshold:
            continue
        cid = id_map[idx]
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            continue
        results.append(
            RetrievedPassage(
                chunk_id=cid,
                text=chunk.text,
                score=float(score),
                source_file=chunk.metadata.source_file,
                page=chunk.metadata.page,
                section=chunk.metadata.section,
            )
        )

    return results


def compute_confidence(passages: list[RetrievedPassage]) -> float:
    """Heuristic confidence score (0-1) based on retrieval quality.

    Combines top score, score spread, and number of qualifying passages.
    """
    if not passages:
        return 0.0

    top_score = passages[0].score
    count_factor = min(len(passages) / 3.0, 1.0)  # saturates at 3 passages
    avg_score = sum(p.score for p in passages) / len(passages)

    confidence = 0.5 * top_score + 0.3 * avg_score + 0.2 * count_factor
    return round(min(max(confidence, 0.0), 1.0), 3)
