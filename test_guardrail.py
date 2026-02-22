from rag.guardrails import _is_on_domain

queries = [
    # Should pass (technical, on domain)
    ("What PPE is required for 400V breaker maintenance?", True),
    # Should pass (natural language, NO keywords, on domain intent)
    ("What do I do if the lights go out in the neighborhood?", True),
    # Should fail (completely off topic)
    ("How do I bake a chocolate cake?", False),
    # Should fail (keyword overlap, but off domain intent)
    ("Where can I buy a portable generator for my camping tent?", False),
]

for q, expected in queries:
    passed = _is_on_domain(q)
    status = "✅ PASS" if passed == expected else "❌ FAIL"
    print(f"[{status}] (Expected {expected}, Got {passed}) Query: {q}")
