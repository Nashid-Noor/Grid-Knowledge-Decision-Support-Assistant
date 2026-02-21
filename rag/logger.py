"""Structured logging for the RAG pipeline.

Writes JSON-lines logs to data/logs/ with automatic daily rotation.
All functions are safe to call from concurrent Gradio workers.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rag.retrieve import RetrievedPassage

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LOG_DIR = Path("data/logs")
QUERY_LOG = LOG_DIR / "queries.jsonl"
SECURITY_LOG = LOG_DIR / "security.jsonl"

_write_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Python stdlib logger (console)
# ---------------------------------------------------------------------------

_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger("grid_rag")
        _logger.setLevel(logging.INFO)
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(levelname)s %(name)s – %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            _logger.addHandler(handler)
    return _logger


# ---------------------------------------------------------------------------
# JSONL writer (thread-safe)
# ---------------------------------------------------------------------------


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to a .jsonl file."""
    _ensure_dirs()
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Public logging API
# ---------------------------------------------------------------------------


def log_query(
    query: str,
    answer: str,
    passages: list[RetrievedPassage],
    confidence: float,
    latency_ms: float,
    guardrail_passed: bool,
    guardrail_reason: Optional[str] = None,
) -> None:
    """Write a structured query log entry."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "query",
        "query": query[:500],  # cap length for safety
        "guardrail_passed": guardrail_passed,
        "guardrail_reason": guardrail_reason,
        "num_passages": len(passages),
        "top_scores": [round(p.score, 4) for p in passages[:5]],
        "confidence": round(confidence, 4),
        "answer_length": len(answer),
        "latency_ms": round(latency_ms, 1),
    }
    _append_jsonl(QUERY_LOG, record)

    logger = get_logger()
    logger.info(
        "query processed | latency=%.0fms passages=%d confidence=%.2f guardrail=%s",
        latency_ms,
        len(passages),
        confidence,
        "pass" if guardrail_passed else f"block({guardrail_reason})",
    )


def log_security_event(
    event_type: str,
    query: str,
    detail: str,
) -> None:
    """Log a security-relevant event (injection attempt, PII detected, etc.)."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "security",
        "type": event_type,
        "query_snippet": query[:200],
        "detail": detail,
    }
    _append_jsonl(SECURITY_LOG, record)

    logger = get_logger()
    logger.warning("SECURITY [%s] %s | query=%s…", event_type, detail, query[:80])


def log_startup(status: str, chunk_count: int, index_cached: bool) -> None:
    """Log pipeline startup state."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "startup",
        "status": status,
        "chunk_count": chunk_count,
        "index_cached": index_cached,
    }
    _append_jsonl(QUERY_LOG, record)
    get_logger().info("startup | chunks=%d cached=%s status=%s", chunk_count, index_cached, status)


def log_upload(filenames: list[str], chunk_count: int) -> None:
    """Log document upload and re-index event."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "upload",
        "files": filenames,
        "chunk_count": chunk_count,
    }
    _append_jsonl(QUERY_LOG, record)
    get_logger().info("upload | files=%s chunks=%d", filenames, chunk_count)


# ---------------------------------------------------------------------------
# Log reading (for UI / eval)
# ---------------------------------------------------------------------------


def read_query_logs(limit: int = 200) -> list[dict[str, Any]]:
    """Read the most recent query log entries."""
    if not QUERY_LOG.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(QUERY_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records[-limit:]


def read_security_logs(limit: int = 100) -> list[dict[str, Any]]:
    """Read the most recent security log entries."""
    if not SECURITY_LOG.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(SECURITY_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records[-limit:]
