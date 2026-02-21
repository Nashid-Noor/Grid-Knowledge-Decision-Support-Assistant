"""Guardrails: input validation, PII detection, injection defence, output checks.

Centralises all safety logic previously split across llm_gemini.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

from rag.logger import log_security_event
from rag.retrieve import RetrievedPassage
from rag.index import embed_texts

# =========================================================================
# Enums & data models
# =========================================================================


class ThreatCategory(str, Enum):
    NONE = "none"
    INJECTION = "prompt_injection"
    SYSTEM_OVERRIDE = "system_override"
    PII_DETECTED = "pii_detected"
    OFF_DOMAIN = "off_domain"
    EMPTY = "empty_query"


@dataclass
class GuardrailResult:
    passed: bool
    category: ThreatCategory = ThreatCategory.NONE
    reason: Optional[str] = None
    pii_types_found: list[str] = field(default_factory=list)
    sanitised_query: Optional[str] = None


# =========================================================================
# 1. PROMPT INJECTION DETECTION
# =========================================================================

_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I), "instruction_override"),
    (re.compile(r"ignore\s+(all\s+)?above", re.I), "instruction_override"),
    (re.compile(r"disregard\s+(all\s+)?(prior|previous|above)", re.I), "instruction_override"),
    (re.compile(r"forget\s+(all\s+)?(your|prior|previous)\s+(rules|instructions)", re.I), "instruction_override"),
    (re.compile(r"you\s+are\s+now\s+(a|an)\b", re.I), "role_hijack"),
    (re.compile(r"act\s+as\s+(a|an|if)\b", re.I), "role_hijack"),
    (re.compile(r"pretend\s+(you('re| are)|to be)", re.I), "role_hijack"),
    (re.compile(r"new\s+(role|persona|identity|instructions?)\s*:", re.I), "role_hijack"),
    (re.compile(r"(system|admin|developer)\s*prompt", re.I), "system_probe"),
    (re.compile(r"reveal\s+(your|the)\s+(system|secret|hidden|internal)", re.I), "system_probe"),
    (re.compile(r"(show|print|output|repeat)\s+(your\s+)?(system|initial)\s+(prompt|instructions)", re.I), "system_probe"),
    (re.compile(r"what\s+(is|are)\s+your\s+(system|original|initial)\s+(prompt|instructions)", re.I), "system_probe"),
    (re.compile(r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|system\|>", re.I), "token_injection"),
    (re.compile(r"```\s*(system|assistant)\b", re.I), "token_injection"),
    (re.compile(r"<\s*/?\s*system\s*>", re.I), "token_injection"),
]


def _check_injection(query: str) -> Optional[str]:
    """Return the matched injection sub-type, or None."""
    for pattern, label in _INJECTION_PATTERNS:
        if pattern.search(query):
            return label
    return None


# =========================================================================
# 2. PII DETECTION (regex-based, best-effort)
# =========================================================================

_PII_PATTERNS: dict[str, re.Pattern] = {
    "ssn": re.compile(r"\b\d{3}[-–]?\d{2}[-–]?\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))\d{8,12}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "date_of_birth": re.compile(
        r"\b(?:dob|date\s+of\s+birth|born)\s*[:=]?\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", re.I
    ),
    "passport": re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
}

# False-positive safelist for IPs commonly found in technical docs
_IP_SAFELIST = {"0.0.0.0", "127.0.0.1", "192.168.0.1", "192.168.1.1", "255.255.255.0", "10.0.0.1"}


def _detect_pii(query: str) -> list[str]:
    """Return list of PII type labels detected in the query."""
    found: list[str] = []
    for pii_type, pattern in _PII_PATTERNS.items():
        matches = pattern.findall(query)
        if pii_type == "ip_address":
            matches = [m for m in matches if m not in _IP_SAFELIST]
        if matches:
            found.append(pii_type)
    return found


def _redact_pii(query: str) -> str:
    """Replace detected PII with redaction tokens for safe logging."""
    redacted = query
    for pii_type, pattern in _PII_PATTERNS.items():
        if pii_type == "ip_address":
            for m in pattern.finditer(redacted):
                if m.group() not in _IP_SAFELIST:
                    redacted = redacted.replace(m.group(), f"[REDACTED_{pii_type.upper()}]")
        else:
            redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)
    return redacted


# =========================================================================
# 3. DOMAIN ENFORCEMENT
# =========================================================================

_DOMAIN_ANCHOR_TEXT = (
    "electricity grid operations power distribution high voltage transmission lines "
    "substation maintenance electrical safety arc flash ppe lockout tagout circuits "
    "breakers transformers utility lineman restoration outage fault generator "
    "switchgear solar clearance grounding"
)

_domain_anchor_vec = None

def _is_on_domain(query: str, threshold: float = 0.20) -> bool:
    """Check if query is semantically similar to the grid domain anchor text."""
    global _domain_anchor_vec
    
    if _domain_anchor_vec is None:
        # Lazy load and normalise the anchor text vector
        vec = embed_texts([_DOMAIN_ANCHOR_TEXT])
        _domain_anchor_vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)

    # Embed and normalise the user query
    query_vec = embed_texts([query])
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    # Compute cosine similarity
    similarity = float(np.dot(_domain_anchor_vec, query_vec.T)[0][0])
    return similarity >= threshold


# =========================================================================
# 4. OUTPUT GUARDRAILS
# =========================================================================

_INSUFFICIENT_CONTEXT_MARKER = "Insufficient context"


def check_output(answer: str, passages: list[RetrievedPassage]) -> str:
    """Validate and post-process the LLM answer."""
    if not passages:
        return (
            "Insufficient context: The available documents do not contain "
            "enough information to answer this question. Please refine your "
            "query or upload additional documents."
        )
    if _INSUFFICIENT_CONTEXT_MARKER.lower() in answer.lower():
        return answer
    if not re.search(r"\[Source:", answer) and len(answer) > 80:
        answer += (
            "\n\n⚠️ *Note: The model did not provide inline citations. "
            "Please verify claims against the retrieved source passages below.*"
        )
    return answer


# =========================================================================
# 5. UNIFIED INPUT GUARDRAIL PIPELINE
# =========================================================================


def run_input_guardrails(query: str) -> GuardrailResult:
    """Run all input checks in priority order.

    Returns a GuardrailResult that tells the caller whether to proceed,
    and carries details for logging and user feedback.
    """
    # --- Empty check ---
    if not query or not query.strip():
        return GuardrailResult(passed=False, category=ThreatCategory.EMPTY, reason="Empty query.")

    # --- Prompt injection / system override ---
    injection_type = _check_injection(query)
    if injection_type:
        category = (
            ThreatCategory.SYSTEM_OVERRIDE
            if injection_type == "system_probe"
            else ThreatCategory.INJECTION
        )
        log_security_event(
            event_type=category.value,
            query=query,
            detail=f"Matched pattern sub-type: {injection_type}",
        )
        return GuardrailResult(
            passed=False,
            category=category,
            reason="Query rejected: potential prompt manipulation detected.",
        )

    # --- PII detection ---
    pii_types = _detect_pii(query)
    if pii_types:
        sanitised = _redact_pii(query)
        log_security_event(
            event_type=ThreatCategory.PII_DETECTED.value,
            query=sanitised,  # log the redacted version
            detail=f"PII types found: {pii_types}",
        )
        return GuardrailResult(
            passed=False,
            category=ThreatCategory.PII_DETECTED,
            reason=(
                "Your query appears to contain personal information "
                f"({', '.join(pii_types)}). Please remove PII before submitting."
            ),
            pii_types_found=pii_types,
            sanitised_query=sanitised,
        )

    # --- Domain enforcement ---
    if not _is_on_domain(query):
        log_security_event(
            event_type=ThreatCategory.OFF_DOMAIN.value,
            query=query,
            detail="No domain keyword matched.",
        )
        return GuardrailResult(
            passed=False,
            category=ThreatCategory.OFF_DOMAIN,
            reason=(
                "This assistant only answers questions related to electricity "
                "distribution, grid operations, safety, and maintenance. "
                "Please rephrase your query within that domain."
            ),
        )

    # --- All clear ---
    return GuardrailResult(passed=True)
