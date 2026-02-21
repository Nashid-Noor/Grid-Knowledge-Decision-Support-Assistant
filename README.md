# ⚡ Grid Knowledge & Decision Support Assistant

A production-structured Retrieval-Augmented Generation (RAG) system for electricity distribution operations, safety, and maintenance guidance.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [RAG Pipeline](#rag-pipeline)
3. [Guardrail Design](#guardrail-design)
4. [Evaluation Framework](#evaluation-framework)
5. [Structured Logging](#structured-logging)
6. [Project Structure](#project-structure)
7. [Setup & Running](#setup--running)
8. [Deployment to Hugging Face Spaces](#deployment-to-hugging-face-spaces)
9. [Migrating to Azure OpenAI](#migrating-to-azure-openai)
10. [Enterprise Considerations](#enterprise-considerations)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GRADIO UI (app.py)                          │
│   Query Tab  │  Upload Tab  │  Logs Tab                             │
└──────┬───────┴──────┬───────┴──────┬────────────────────────────────┘
       │              │              │
       ▼              │              ▼
┌──────────────┐      │       ┌──────────────┐
│  INPUT       │      │       │  STRUCTURED  │
│  GUARDRAILS  │      │       │  LOGGER      │
│              │      │       │              │
│  • Injection │      │       │  • queries   │
│  • PII       │      │       │  • latency   │
│  • Domain    │      │       │  • security  │
└──────┬───────┘      │       └──────────────┘
       │              │
       ▼              ▼
┌──────────────┐  ┌──────────────┐
│  RETRIEVER   │  │  INGESTION   │
│              │  │              │
│  • Embed q   │  │  • PDF/TXT   │
│  • FAISS     │  │  • Chunking  │
│  • Top-k     │  │  • Metadata  │
│  • Threshold │  │  • FAISS idx │
└──────┬───────┘  └──────────────┘
       │
       ▼
┌──────────────┐
│  PROMPT      │
│  ASSEMBLY    │
│              │
│  • System    │
│  • Context   │
│  • Question  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  GEMINI LLM  │
│              │
│  • T=0.1     │
│  • Grounded  │
│  • Cited     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  OUTPUT      │
│  GUARDRAILS  │
│              │
│  • Citation  │
│  • Refusal   │
└──────────────┘
```

The system follows a linear pipeline: user query → input validation → vector retrieval → prompt construction → LLM synthesis → output validation → response. Every stage is a separate module with no cross-cutting dependencies beyond explicit imports.

## RAG Pipeline

### Ingestion (`rag/ingest.py`)

Documents are loaded from `data/docs/` using format-specific readers. PDF extraction uses `pypdf`; plain text and Markdown are read directly. Each document is split into overlapping word-boundary chunks (default: 512 words, 64-word overlap) to preserve context across boundaries.

Every chunk receives a deterministic ID computed as a truncated SHA-256 hash of `source_file|page|chunk_index|text_prefix`. This ensures that re-ingesting the same document produces identical IDs, enabling safe deduplication and incremental updates.

Metadata attached to each chunk includes: source filename, page number (PDF only), best-effort section heading detection via regex, character offsets, and a word-count token estimate.

The full chunk manifest is persisted as `data/cache/chunks.json` so that the system can reload without re-processing documents.

### Embedding & Indexing (`rag/index.py`)

Text chunks are encoded using `sentence-transformers/all-MiniLM-L6-v2`, a 384-dimension model that runs efficiently on CPU. The encoder is loaded as a process-global singleton to avoid repeated model initialization.

All vectors are L2-normalised before insertion into a FAISS `IndexFlatIP` index. This means inner-product search computes cosine similarity — the standard metric for semantic text similarity. The index and its ID mapping are saved to disk (`data/cache/faiss.index`, `data/cache/id_map.json`) and loaded on subsequent startups without recomputation.

### Retrieval (`rag/retrieve.py`)

At query time, the user question is embedded with the same model, normalised, and searched against the FAISS index. Two filtering controls are exposed:

- **Top-K** (default 5): Maximum number of passages returned.
- **Score threshold** (default 0.25): Minimum cosine similarity to include a passage. This filters out low-relevance noise that would confuse the LLM.

A heuristic confidence score (0–1) is computed from the top retrieval score, the average score across returned passages, and the count of qualifying passages. This is displayed to the user as a colour-coded badge.

### Prompt Construction (`rag/prompt.py`)

The system prompt enforces strict grounding: the LLM must answer only from the provided context, cite every claim using `[Source: filename, Page N]` format, and produce a standard refusal if context is insufficient.

Retrieved passages are formatted into a numbered CONTEXT block with metadata headers (source, page, section, relevance score) injected directly into the user message.

### LLM Synthesis (`rag/llm_gemini.py`)

The Gemini 1.5 Flash model is called with temperature 0.1 and a 1024-token output cap. Low temperature minimises creative drift — critical in a safety-oriented domain where fabricated procedures could cause physical harm.

After generation, output guardrails validate that citations are present and that the model did not hallucinate source references.

## Guardrail Design

Guardrails are consolidated in `rag/guardrails.py` and operate in two layers.

### Input Guardrails (run before retrieval)

| Check | Method | Action |
|---|---|---|
| **Prompt injection** | 14 regex patterns covering instruction override, role hijacking, system probing, and token injection | Block query, log to security log |
| **System prompt override** | Subset of injection patterns targeting system/admin prompt extraction | Block query, log as `system_override` |
| **PII detection** | Regex patterns for SSN, email, phone, credit card, IP address, date of birth, passport numbers | Block query, log redacted version |
| **Domain enforcement** | Keyword allowlist (~50 terms covering grid operations vocabulary) | Block query, inform user of scope |
| **Empty query** | Whitespace/length check | Block with message |

Checks run in priority order: injection → PII → domain → proceed. This ensures that a query containing both an injection attempt and PII is classified as injection (the higher-severity threat).

PII detection includes a safelist for common technical IP addresses (127.0.0.1, 192.168.x.x) that appear frequently in infrastructure documentation.

### Output Guardrails (run after LLM generation)

| Check | Condition | Action |
|---|---|---|
| **No context** | Zero passages retrieved | Return standard refusal |
| **Model refusal** | Answer contains "Insufficient context" | Pass through (model correctly refused) |
| **Missing citations** | No `[Source:` pattern and answer >80 chars | Append verification warning |

### Security Logging

All guardrail blocks are written to `data/logs/security.jsonl` with timestamps, threat category, redacted query snippets, and pattern match details. This supports incident review and pattern analysis.

## Evaluation Framework

### Dataset Structure (`data/eval/eval_dataset.json`)

The evaluation dataset contains 40 QA pairs spanning five domain tags (safety, maintenance, operations, compliance) and three difficulty levels (basic, intermediate, advanced). Each entry has:

- `question`: Natural-language query
- `expected_keywords`: Terms the answer should contain
- `expected_source_files`: Filenames where the answer should originate (populated after ingestion)
- `domain_tag` and `difficulty`: For stratified analysis

### Metrics (`rag/evaluation.py`)

| Metric | What it measures | How |
|---|---|---|
| **Retrieval hit rate** | Whether relevant passages were retrieved | ≥50% of expected keywords found in any single passage |
| **Keyword recall** | Answer completeness | Fraction of expected keywords present in the LLM answer |
| **Citation presence rate** | Whether the model cited sources at all | Regex check for `[Source:` pattern |
| **Citation correctness rate** | Whether cited filenames match retrieved passages | Cross-reference extracted filenames against passage metadata |
| **Mean / Median / P95 latency** | End-to-end response time | Wall-clock measurement per query |

Results are broken down by domain tag and difficulty level. A detailed JSON report is saved to `data/eval/eval_results.json`.

### Running Evaluation

```bash
python -m rag.evaluation --dataset data/eval/eval_dataset.json --top-k 5 --threshold 0.25
```

The evaluation runner works without a Gemini API key — it falls back to measuring retrieval-only metrics using concatenated passage text.

## Structured Logging

All logging is in `rag/logger.py` and writes JSON-lines files to `data/logs/`.

### Query Log (`queries.jsonl`)

Every query (successful or blocked) produces a record:

```json
{
  "timestamp": "2025-06-15T14:30:22+00:00",
  "event": "query",
  "query": "What PPE is required...",
  "guardrail_passed": true,
  "num_passages": 4,
  "top_scores": [0.7823, 0.6541, 0.5912, 0.4103],
  "confidence": 0.682,
  "answer_length": 847,
  "latency_ms": 1823.4
}
```

### Security Log (`security.jsonl`)

Guardrail violations are recorded separately:

```json
{
  "timestamp": "2025-06-15T14:31:05+00:00",
  "event": "security",
  "type": "prompt_injection",
  "query_snippet": "ignore all previous instructions and...",
  "detail": "Matched pattern sub-type: instruction_override"
}
```

Startup events, uploads, and re-indexing are also logged. All writes are thread-safe (mutex-protected) for concurrent Gradio workers.

The Gradio UI includes a Logs tab that renders the last 20 query logs and 15 security events as sortable Markdown tables.

## Project Structure

```
├── app.py                          # Gradio UI, pipeline orchestration
├── requirements.txt                # Pinned dependencies (CPU-only)
├── .env.example                    # Environment variable template
├── rag/
│   ├── __init__.py
│   ├── ingest.py                   # Document loading, chunking, metadata
│   ├── index.py                    # FAISS index + sentence-transformer embeddings
│   ├── retrieve.py                 # Top-k retrieval with score filtering
│   ├── prompt.py                   # System prompt + context assembly
│   ├── llm_gemini.py               # Gemini API calls + output guardrails
│   ├── guardrails.py               # Input guardrails: injection, PII, domain
│   ├── evaluation.py               # Eval framework: metrics, reporting
│   └── logger.py                   # Structured JSON-lines logging
├── data/
│   ├── docs/                       # Source documents (PDF, TXT, MD)
│   ├── cache/                      # FAISS index, chunk manifest (auto-generated)
│   ├── eval/
│   │   └── eval_dataset.json       # 40 QA evaluation pairs
│   └── logs/                       # Query + security logs (auto-generated)
```

## Setup & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY

# Add documents
cp your-safety-manual.pdf data/docs/

# Run
python app.py
```

The first run indexes all documents and caches the FAISS index. Subsequent starts load from cache in under 2 seconds.

## Deployment to Hugging Face Spaces

1. Create a new Space → select **Gradio** SDK.
2. Add `GEMINI_API_KEY` as a **Space Secret** (Settings → Secrets).
3. Push the repository. HF auto-detects `app.py` and `requirements.txt`.
4. Pre-populate `data/docs/` with your documents, or use the Upload tab after deployment.

The system runs entirely on CPU. No GPU quota is required. The `all-MiniLM-L6-v2` model (~80MB) is downloaded on first start and cached by HF infrastructure.

For persistent storage across Space restarts, use a [Hugging Face Dataset repository](https://huggingface.co/docs/hub/datasets) mounted as a volume, or enable persistent storage in Space settings.

## Migrating to Azure OpenAI

To switch from Gemini to Azure OpenAI, modify only `rag/llm_gemini.py`:

```python
# Replace google.generativeai with:
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-06-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)

def generate_answer(query, passages, model_name="gpt-4o"):
    messages = build_prompt(query, passages)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": messages[0]["content"]},
            {"role": "user", "content": messages[1]["content"]},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content or ""
    return check_output(raw, passages)
```

Add `openai>=1.30.0` to `requirements.txt` and set `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` environment variables. No other module changes are required — the prompt format and guardrails are LLM-agnostic.

## Enterprise Considerations

### Authentication & Access Control

The current system has no auth layer. For enterprise deployment, add Gradio's built-in `auth` parameter, or deploy behind an identity-aware proxy (Azure AD App Proxy, AWS ALB with Cognito, Cloudflare Access). Role-based access could gate the Logs tab to administrators only.

### Data Residency & Compliance

Queries are sent to Google's Gemini API (or Azure OpenAI). For data residency requirements, use Azure OpenAI with a region-specific deployment, or switch to a self-hosted model (Llama 3, Mistral) via vLLM or Ollama. The prompt module (`rag/prompt.py`) works with any chat-completion API.

### Document Security

Uploaded documents are stored in plaintext on disk. For sensitive materials, encrypt `data/docs/` at rest (LUKS, Azure Disk Encryption), restrict filesystem permissions, and consider document-level access control in the retrieval layer (tag chunks with access groups and filter at query time).

### Scalability

FAISS `IndexFlatIP` performs exact search — suitable for up to ~100K chunks on CPU. Beyond that, switch to `IndexIVFFlat` or `IndexHNSWFlat` for approximate nearest-neighbor search with sub-linear query time. For multi-node deployments, consider Qdrant, Weaviate, or Azure AI Search as managed vector stores.

### Monitoring & Alerting

The JSON-lines logs are designed for ingestion into centralized logging (ELK, Datadog, Azure Monitor). Key alerts to configure: security event rate spikes, P95 latency exceeding SLA, retrieval confidence consistently below threshold (indicates stale or missing documents).

### Model Governance

The evaluation framework should be run on every document update and model change. Track metrics over time to detect retrieval degradation. For regulated environments, log the full prompt and response (not just metadata) to an append-only audit store with configurable retention.

### Cost Management

Gemini 1.5 Flash is priced per input/output token. The prompt template injects 5 passages per query (~2,500 tokens input). At 1,000 queries/day this is approximately 2.5M input tokens/day. Monitor token usage via the Gemini API dashboard and set billing alerts. For cost reduction, consider reducing `top_k` or switching to smaller context windows.
