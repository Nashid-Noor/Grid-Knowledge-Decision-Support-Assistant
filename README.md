---
title: Grid QA Assistant
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---
# ⚡ Grid Knowledge & Decision Support Assistant

A production-structured Retrieval-Augmented Generation (RAG) system for electricity distribution operations, safety, and maintenance guidance.

---


# ⚡ Grid QA & Decision Support Assistant

**Live Demo:** [Try the Gradio RAG Assistant here!](https://huggingface.co/spaces/nashid16/grid-qa-assistant)

## What is this?
I built this project because I wanted to explore how we can use LLMs safely in high-stakes environments. If you ask ChatGPT a question about grid maintenance, and it hallucinates a power rating, the results could be catastrophic. **Grid QA** is my attempt at solving that problem.

It's a production-grade Retrieval-Augmented Generation (RAG) system built specifically for electricity distribution operations. It ingests thousands of pages of dense technical manuals and allows operators to query them safely. The core focus here isn't just about getting an answer; it's about getting a *grounded* answer with strict citations, safety guardrails, and no hallucinations.

---

## 🏗️ How it's Built (The Architecture)

I designed the pipeline to be modular, prioritizing security and predictability at every step:

```text
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

### 1. Ingestion & Vector Search (`rag/ingest.py`, `rag/index.py`, `rag/retrieve.py`)
Instead of relying on an expensive cloud vector database, I wanted to keep this self-contained. The system parses PDF and text manuals, chunks them with overlapping boundaries so context isn't lost, and assigns deterministic SHA-256 chunk IDs. It then embeds everything locally using `sentence-transformers` and stores the vectors in a highly efficient **FAISS IndexFlatIP** database.

### 2. Multi-Layer Guardrails (`rag/guardrails.py`)
This is the most critical part of the system.
* **Input side:** Before the LLM even sees a prompt, the system checks for prompt injections, filters out any Personal Identifiable Information (PII) using regex/patterns, and enforces a strict grid-operations domain allowlist.
* **Output side:** The system parses the LLM's response before showing it to the user. It explicitly verifies that the LLM has included citations in the exact format `[Source: document.pdf, Page N]`, preventing ungrounded claims.

### 3. The LLM Call (`rag/llm_gemini.py`)
I used the Gemini API and tuned the temperature way down to `0.1`. The system prompt is incredibly strict, essentially telling the model: *"If you cannot find the answer in the retrieved context, you must refuse to answer."*

### 4. Evaluation Framework (`rag/evaluation.py`)
You can't improve what you don't measure. I built a testing suite that measures **retrieval hit rate**, **keyword recall**, and **citation correctness** against a dataset of complex operational questions, acting as a unit test for the LLM pipeline itself.

---

## 💻 Tech Stack
* **Python**
* **Gemini 1.5 Flash API** (Generation)
* **FAISS & Sentence-Transformers** (Retrieval & Local Embedding)
* **Gradio** (Frontend Interface)
* **Regex/Pydantic** (Guardrails and validation)

---

## 🚀 Running it Locally

If you want to spin this up yourself, it's pretty straightforward.

1.  **Clone and Install dependencies:**
    ```bash
    git clone https://github.com/Nashid-Noor/Grid-Knowledge-Decision-Support-Assistant.git
    cd Grid-Knowledge-Decision-Support-Assistant
    pip install -r requirements.txt
    ```

2.  **Add your Gemini API Key:**
    Create a `.env` file in the root directory and add:
    ```env
    GEMINI_API_KEY=your_key_here
    ```

3.  **Start the App:**
    ```bash
    python app.py
    ```
    *The app will be available in your browser at `http://localhost:7860`.*

---

## 📂 Project Structure

```text
grid-qa/
├── app.py                  # Gradio UI & main application orchestration
├── data/
│   ├── raw/                # Dump PDF/TXT technical safety manuals here
│   ├── index/              # Where the FAISS index and chunk manifest are saved
│   └── logs/               # JSON-lines structured event logs
├── rag/                    # The brain of the application
│   ├── evaluation.py       # Metrics testing framework
│   ├── guardrails.py       # Input/Output security validation
│   ├── index.py            # Local FAISS embedding wrapper
│   ├── ingest.py           # Document parsing & chunking engine
│   ├── llm_gemini.py       # Interfacing with the Gemini API
│   ├── logger.py           # Structured event logger
│   ├── prompt.py           # Prompt assembler handling system context
│   └── retrieve.py         # FAISS nearest-neighbor semantic search
├── tests/                  # Scripts for hitting edge cases (test_guardrail.py, etc.)
└── requirements.txt
```

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
