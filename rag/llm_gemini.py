"""Gemini API integration for answer synthesis.

Guardrail logic has been moved to rag.guardrails.
This module handles only LLM configuration and generation.
"""

from __future__ import annotations

import os
from typing import Optional

import google.generativeai as genai

from rag.guardrails import check_output
from rag.prompt import build_prompt
from rag.retrieve import RetrievedPassage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-1.5-flash"
TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 1024

# ---------------------------------------------------------------------------
# Gemini setup
# ---------------------------------------------------------------------------


def configure_gemini(api_key: Optional[str] = None) -> None:
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Provide it via environment variable or argument."
        )
    genai.configure(api_key=key)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_answer(
    query: str,
    passages: list[RetrievedPassage],
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Call Gemini to synthesise an answer from retrieved passages.

    Output guardrails are applied before returning.
    """
    messages = build_prompt(query, passages)

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=messages[0]["content"],
        generation_config=genai.GenerationConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )

    response = model.generate_content(messages[1]["content"])
    raw_answer = response.text if response.text else ""
    return check_output(raw_answer, passages)
