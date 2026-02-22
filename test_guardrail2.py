import numpy as np
from rag.index import embed_texts
from rag.guardrails import _DOMAIN_ANCHOR_TEXT

_domain_anchor_vec = None

queries = [
    # Technical
    "What PPE is required for 400V breaker maintenance?",
    # Natural language
    "What do I do if the lights go out in the neighborhood?",
    # Off topic completely
    "How do I bake a chocolate cake?",
    # Off topic but keyword overlap
    "Where can I buy a portable generator for my camping tent?",
]

def print_scores():
    global _domain_anchor_vec
    vec = embed_texts([_DOMAIN_ANCHOR_TEXT])
    _domain_anchor_vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)

    for q in queries:
        query_vec = embed_texts([q])
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        similarity = float(np.dot(_domain_anchor_vec, query_vec.T)[0][0])
        print(f"[{similarity:.4f}] {q}")

print_scores()
