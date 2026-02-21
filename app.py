"""Grid Knowledge & Decision Support Assistant – Gradio application.

Phase 2: integrated structured logging, consolidated guardrails,
and evaluation access via the UI.
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from rag.ingest import (
    DocumentChunk,
    ingest_directory,
    load_chunks_manifest,
    save_chunks_manifest,
)
from rag.index import build_index, index_ready, load_index, save_index
from rag.retrieve import RetrievedPassage, compute_confidence, retrieve
from rag.guardrails import run_input_guardrails
from rag.llm_gemini import configure_gemini, generate_answer
from rag.logger import (
    get_logger,
    log_query,
    log_security_event,
    log_startup,
    log_upload,
    read_query_logs,
    read_security_logs,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_DIR = Path("data/docs")
CACHE_DIR = Path("data/cache")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_faiss_index = None
_id_map: list[str] = []
_chunks_by_id: dict[str, DocumentChunk] = {}

logger = get_logger()

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _startup() -> str:
    global _faiss_index, _id_map, _chunks_by_id

    try:
        configure_gemini()
        gemini_status = "✅ Gemini configured"
    except EnvironmentError as exc:
        gemini_status = f"⚠️ {exc}"

    if index_ready():
        _faiss_index, _id_map = load_index()
        chunks = load_chunks_manifest()
        _chunks_by_id = {c.chunk_id: c for c in chunks}
        status = f"{gemini_status} | ✅ Loaded cached index ({len(chunks)} chunks)"
        log_startup(status, len(chunks), index_cached=True)
        return status

    chunks = ingest_directory(DOCS_DIR)
    if not chunks:
        status = f"{gemini_status} | ⚠️ No documents in {DOCS_DIR}. Upload PDFs or text files."
        log_startup(status, 0, index_cached=False)
        return status

    _faiss_index, _id_map = build_index(chunks)
    save_index(_faiss_index, _id_map)
    save_chunks_manifest(chunks)
    _chunks_by_id = {c.chunk_id: c for c in chunks}
    status = f"{gemini_status} | ✅ Indexed {len(chunks)} chunks from {DOCS_DIR}"
    log_startup(status, len(chunks), index_cached=False)
    return status


# ---------------------------------------------------------------------------
# Query pipeline
# ---------------------------------------------------------------------------


def _format_sources(passages: list[RetrievedPassage]) -> str:
    if not passages:
        return "*No passages retrieved.*"
    lines: list[str] = []
    for i, p in enumerate(passages, 1):
        header = f"**[{i}] {p.source_file}"
        if p.page:
            header += f" – Page {p.page}"
        header += f"** (score: {p.score:.3f})"
        if p.section:
            header += f"\n*Section: {p.section}*"
        lines.append(f"{header}\n\n{p.text}\n")
    return "\n---\n".join(lines)


def _confidence_badge(score: float) -> str:
    if score >= 0.65:
        return f"🟢 High ({score:.0%})"
    if score >= 0.40:
        return f"🟡 Medium ({score:.0%})"
    return f"🔴 Low ({score:.0%})"


def handle_query(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.25,
) -> tuple[str, str, str]:
    t0 = time.perf_counter()

    # Input guardrails
    guard = run_input_guardrails(query)
    if not guard.passed:
        latency_ms = (time.perf_counter() - t0) * 1000
        log_query(
            query=query,
            answer=guard.reason or "",
            passages=[],
            confidence=0.0,
            latency_ms=latency_ms,
            guardrail_passed=False,
            guardrail_reason=guard.category.value,
        )
        return guard.reason or "Query rejected.", "", "⛔ Blocked"

    if _faiss_index is None or not _id_map:
        return (
            "The document index is empty. Please add documents to `data/docs/` and restart.",
            "",
            "⛔ No index",
        )

    # Retrieve
    passages = retrieve(
        query=query,
        index=_faiss_index,
        id_map=_id_map,
        chunks_by_id=_chunks_by_id,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    confidence = compute_confidence(passages)

    # Generate
    try:
        answer = generate_answer(query, passages)
    except Exception as exc:
        answer = f"⚠️ LLM generation failed: {exc}"

    latency_ms = (time.perf_counter() - t0) * 1000

    # Log
    log_query(
        query=query,
        answer=answer,
        passages=passages,
        confidence=confidence,
        latency_ms=latency_ms,
        guardrail_passed=True,
    )

    return answer, _format_sources(passages), _confidence_badge(confidence)


# ---------------------------------------------------------------------------
# File upload handler
# ---------------------------------------------------------------------------


def handle_upload(files: list) -> str:
    global _faiss_index, _id_map, _chunks_by_id

    if not files:
        return "No files selected."

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for f in files:
        dest = DOCS_DIR / Path(f.name).name
        with open(dest, "wb") as out:
            out.write(Path(f.name).read_bytes())
        saved.append(dest.name)

    chunks = ingest_directory(DOCS_DIR)
    _faiss_index, _id_map = build_index(chunks)
    save_index(_faiss_index, _id_map)
    save_chunks_manifest(chunks)
    _chunks_by_id = {c.chunk_id: c for c in chunks}

    log_upload(saved, len(chunks))
    return f"✅ Uploaded {len(saved)} file(s): {', '.join(saved)} → {len(chunks)} total chunks indexed."


# ---------------------------------------------------------------------------
# Log viewer (Gradio tab)
# ---------------------------------------------------------------------------


def _render_query_logs() -> str:
    logs = read_query_logs(limit=50)
    query_logs = [l for l in logs if l.get("event") == "query"]
    if not query_logs:
        return "*No query logs yet.*"
    lines: list[str] = ["| Time | Query | Passages | Confidence | Latency | Guardrail |", "| --- | --- | --- | --- | --- | --- |"]
    for entry in reversed(query_logs[-20:]):
        ts = entry.get("timestamp", "")[:19]
        q = entry.get("query", "")[:50]
        n = entry.get("num_passages", 0)
        c = entry.get("confidence", 0)
        lat = entry.get("latency_ms", 0)
        gp = "✅" if entry.get("guardrail_passed") else f"⛔ {entry.get('guardrail_reason', '')}"
        lines.append(f"| {ts} | {q} | {n} | {c:.2f} | {lat:.0f}ms | {gp} |")
    return "\n".join(lines)


def _render_security_logs() -> str:
    logs = read_security_logs(limit=30)
    if not logs:
        return "*No security events logged.*"
    lines: list[str] = ["| Time | Type | Detail | Query Snippet |", "| --- | --- | --- | --- |"]
    for entry in reversed(logs[-15:]):
        ts = entry.get("timestamp", "")[:19]
        etype = entry.get("type", "")
        detail = entry.get("detail", "")[:60]
        snippet = entry.get("query_snippet", "")[:40]
        lines.append(f"| {ts} | {etype} | {detail} | {snippet} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    status_msg = _startup()

    with gr.Blocks(
        title="Grid Knowledge & Decision Support Assistant",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown(
            "# ⚡ Grid Knowledge & Decision Support Assistant\n"
            "Ask questions about electricity distribution safety, maintenance, and operations. "
            "Answers are grounded exclusively in uploaded documents."
        )
        gr.Markdown(f"**System status:** {status_msg}")

        with gr.Tabs():
            # ---- Tab 1: Query ----
            with gr.Tab("🔍 Query"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_box = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g. What are the PPE requirements for live-line work?",
                            lines=2,
                        )
                        with gr.Row():
                            top_k_slider = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1, label="Top-K Passages"
                            )
                            threshold_slider = gr.Slider(
                                minimum=0.0, maximum=0.8, value=0.25, step=0.05, label="Score Threshold"
                            )
                        submit_btn = gr.Button("🔍 Ask", variant="primary")

                    with gr.Column(scale=1):
                        confidence_display = gr.Textbox(label="Confidence", interactive=False)

                answer_box = gr.Markdown(label="Answer")

                with gr.Accordion("📄 Retrieved Source Passages", open=False):
                    sources_box = gr.Markdown()

            # ---- Tab 2: Upload ----
            with gr.Tab("📁 Upload"):
                gr.Markdown("### Upload Documents\nAdd PDF or text files to the knowledge base.")
                with gr.Row():
                    upload_files = gr.File(
                        label="Upload PDF / TXT files",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".md"],
                    )
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)
                upload_btn = gr.Button("📤 Upload & Re-index")

            # ---- Tab 3: Logs ----
            with gr.Tab("📊 Logs"):
                gr.Markdown("### Query Log (last 20)")
                query_log_display = gr.Markdown()
                gr.Markdown("### Security Events (last 15)")
                security_log_display = gr.Markdown()
                refresh_logs_btn = gr.Button("🔄 Refresh Logs")

        # ---- Event wiring ----
        submit_btn.click(
            fn=handle_query,
            inputs=[query_box, top_k_slider, threshold_slider],
            outputs=[answer_box, sources_box, confidence_display],
        )
        query_box.submit(
            fn=handle_query,
            inputs=[query_box, top_k_slider, threshold_slider],
            outputs=[answer_box, sources_box, confidence_display],
        )
        upload_btn.click(
            fn=handle_upload,
            inputs=[upload_files],
            outputs=[upload_status],
        )
        refresh_logs_btn.click(
            fn=lambda: (_render_query_logs(), _render_security_logs()),
            outputs=[query_log_display, security_log_display],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
