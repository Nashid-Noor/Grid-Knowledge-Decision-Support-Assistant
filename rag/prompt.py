"""Prompt construction for grounded, citation-backed answers."""

from __future__ import annotations

from rag.retrieve import RetrievedPassage

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the **Grid Knowledge & Decision Support Assistant**, an expert system \
for electricity distribution operations, safety, and maintenance.

STRICT RULES:
1. Answer ONLY using the provided CONTEXT passages below. Do NOT use prior knowledge.
2. For every factual claim, cite the source using [Source: <filename>, Page <N>] format.
3. If the context does not contain sufficient information to answer, respond EXACTLY:
   "Insufficient context: The available documents do not contain enough information to \
answer this question. Please refine your query or upload additional documents."
4. Do NOT speculate, hallucinate, or extrapolate beyond the provided passages.
5. Use clear, professional language suitable for utility operations engineers.
6. When safety-critical information is referenced, highlight it explicitly.
7. Structure longer answers with numbered steps or brief headings where appropriate.
"""

# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def format_context(passages: list[RetrievedPassage]) -> str:
    """Build the CONTEXT block injected into the user prompt."""
    if not passages:
        return "CONTEXT:\n[No relevant passages found.]\n"

    blocks: list[str] = ["CONTEXT:"]
    for i, p in enumerate(passages, 1):
        header = f"[{i}] Source: {p.source_file}"
        if p.page:
            header += f", Page {p.page}"
        if p.section:
            header += f" | Section: {p.section}"
        header += f" (relevance: {p.score:.2f})"
        blocks.append(f"{header}\n{p.text}\n")

    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Full prompt assembly
# ---------------------------------------------------------------------------

def build_prompt(query: str, passages: list[RetrievedPassage]) -> list[dict[str, str]]:
    """Return the messages list ready for the Gemini chat completion call."""
    context_block = format_context(passages)
    user_content = f"{context_block}\n---\nQUESTION: {query}\n"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
