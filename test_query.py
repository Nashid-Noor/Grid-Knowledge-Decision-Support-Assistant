import numpy as np
from rag.index import embed_texts
from rag.guardrails import _DOMAIN_ANCHOR_TEXT

vec = embed_texts([_DOMAIN_ANCHOR_TEXT])
anchor_vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)

q = "what do you think about the doc uplaoded?"
q_vec = embed_texts([q])
q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)
sim = float(np.dot(anchor_vec, q_vec.T)[0][0])
print(sim)
