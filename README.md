
#  Grid QA & Decision Support Assistant

**Live Demo:** (https://huggingface.co/spaces/nashid16/grid-qa-assistant)

## What is this?
I built this project because I wanted to explore how we can use LLMs safely in high-stakes environments. If you ask ChatGPT a question about grid maintenance, and it hallucinates a power rating, the results could be catastrophic. **Grid QA** is my attempt at solving that problem.

It's a production-grade Retrieval-Augmented Generation (RAG) system built specifically for electricity distribution operations. It ingests thousands of pages of dense technical manuals and allows operators to query them safely. The core focus here isn't just about getting an answer, it's about getting a *grounded* answer with strict citations, safety guardrails, and no hallucinations.

---

## The Architecture



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

##  Running it Locally

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
    





