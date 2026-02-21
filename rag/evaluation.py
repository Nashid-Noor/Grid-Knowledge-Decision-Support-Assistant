"""Evaluation framework: retrieval accuracy, citation quality, latency.

Can be run standalone:
    python -m rag.evaluation --dataset data/eval/eval_dataset.json

Or imported and called programmatically from tests / CI.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rag.retrieve import RetrievedPassage

# =========================================================================
# Data models
# =========================================================================


@dataclass
class EvalCase:
    """One evaluation QA pair loaded from the dataset JSON."""
    id: str
    question: str
    expected_keywords: list[str]
    expected_source_files: list[str]
    domain_tag: str
    difficulty: str


@dataclass
class EvalResult:
    """Result of evaluating a single QA pair."""
    eval_id: str
    question: str
    retrieval_hit: bool
    keyword_recall: float
    citation_present: bool
    citation_correct: bool
    latency_ms: float
    answer_snippet: str
    top_retrieval_score: float
    num_passages: int


@dataclass
class EvalSummary:
    """Aggregated metrics across all evaluated pairs."""
    total: int = 0
    retrieval_hit_rate: float = 0.0
    mean_keyword_recall: float = 0.0
    citation_presence_rate: float = 0.0
    citation_correctness_rate: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    results_by_domain: dict[str, dict[str, float]] = field(default_factory=dict)
    results_by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)


# =========================================================================
# Dataset loader
# =========================================================================


def load_eval_dataset(path: str | Path) -> list[EvalCase]:
    """Load evaluation QA pairs from the structured JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return [
        EvalCase(
            id=item["id"],
            question=item["question"],
            expected_keywords=item.get("expected_keywords", []),
            expected_source_files=item.get("expected_source_files", []),
            domain_tag=item.get("domain_tag", "unknown"),
            difficulty=item.get("difficulty", "unknown"),
        )
        for item in data["qa_pairs"]
    ]


# =========================================================================
# Individual metric functions
# =========================================================================


def measure_keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def measure_retrieval_hit(
    passages: list[RetrievedPassage],
    expected_keywords: list[str],
) -> bool:
    """True if at least one retrieved passage contains ≥50% of expected keywords."""
    if not expected_keywords:
        return True
    for p in passages:
        text_lower = p.text.lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
        if hits / len(expected_keywords) >= 0.5:
            return True
    return False


def measure_citation_presence(answer: str) -> bool:
    """True if the answer contains at least one [Source: ...] citation."""
    return bool(re.search(r"\[Source:", answer))


def measure_citation_correctness(
    answer: str,
    passages: list[RetrievedPassage],
) -> bool:
    """True if every cited filename actually appears in retrieved passages.

    If no citations are present, returns False (citations were expected).
    """
    cited_files = re.findall(r"\[Source:\s*([^,\]]+)", answer)
    if not cited_files:
        return False
    retrieved_files = {p.source_file for p in passages}
    return all(cf.strip() in retrieved_files for cf in cited_files)


# =========================================================================
# Single-case evaluator
# =========================================================================


def evaluate_single(
    case: EvalCase,
    passages: list[RetrievedPassage],
    answer: str,
    latency_ms: float,
) -> EvalResult:
    """Evaluate one QA pair against the pipeline outputs."""
    return EvalResult(
        eval_id=case.id,
        question=case.question,
        retrieval_hit=measure_retrieval_hit(passages, case.expected_keywords),
        keyword_recall=measure_keyword_recall(answer, case.expected_keywords),
        citation_present=measure_citation_presence(answer),
        citation_correct=measure_citation_correctness(answer, passages),
        latency_ms=latency_ms,
        answer_snippet=answer[:300],
        top_retrieval_score=passages[0].score if passages else 0.0,
        num_passages=len(passages),
    )


# =========================================================================
# Aggregate summary
# =========================================================================


def _group_metric(
    results: list[EvalResult],
    group_key: str,
    cases: list[EvalCase],
) -> dict[str, dict[str, float]]:
    """Compute per-group (domain/difficulty) aggregates."""
    case_map = {c.id: c for c in cases}
    groups: dict[str, list[EvalResult]] = {}
    for r in results:
        c = case_map.get(r.eval_id)
        key = getattr(c, group_key, "unknown") if c else "unknown"
        groups.setdefault(key, []).append(r)

    out: dict[str, dict[str, float]] = {}
    for key, group in sorted(groups.items()):
        out[key] = {
            "count": len(group),
            "retrieval_hit_rate": sum(r.retrieval_hit for r in group) / len(group),
            "mean_keyword_recall": statistics.mean(r.keyword_recall for r in group),
            "citation_presence_rate": sum(r.citation_present for r in group) / len(group),
            "mean_latency_ms": statistics.mean(r.latency_ms for r in group),
        }
    return out


def compute_summary(results: list[EvalResult], cases: list[EvalCase]) -> EvalSummary:
    """Aggregate individual results into a summary report."""
    if not results:
        return EvalSummary()

    latencies = [r.latency_ms for r in results]
    return EvalSummary(
        total=len(results),
        retrieval_hit_rate=sum(r.retrieval_hit for r in results) / len(results),
        mean_keyword_recall=statistics.mean(r.keyword_recall for r in results),
        citation_presence_rate=sum(r.citation_present for r in results) / len(results),
        citation_correctness_rate=sum(r.citation_correct for r in results) / len(results),
        mean_latency_ms=statistics.mean(latencies),
        p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else latencies[0],
        median_latency_ms=statistics.median(latencies),
        results_by_domain=_group_metric(results, "domain_tag", cases),
        results_by_difficulty=_group_metric(results, "difficulty", cases),
    )


# =========================================================================
# Report printer
# =========================================================================


def print_report(summary: EvalSummary) -> str:
    """Format evaluation summary as a human-readable report."""
    lines: list[str] = [
        "=" * 70,
        "  GRID RAG EVALUATION REPORT",
        "=" * 70,
        "",
        f"  Total QA pairs evaluated:     {summary.total}",
        "",
        "  RETRIEVAL METRICS",
        f"    Retrieval hit rate:          {summary.retrieval_hit_rate:.1%}",
        f"    Mean keyword recall:         {summary.mean_keyword_recall:.1%}",
        "",
        "  CITATION METRICS",
        f"    Citation presence rate:      {summary.citation_presence_rate:.1%}",
        f"    Citation correctness rate:   {summary.citation_correctness_rate:.1%}",
        "",
        "  LATENCY METRICS",
        f"    Mean latency:                {summary.mean_latency_ms:.0f} ms",
        f"    Median latency:              {summary.median_latency_ms:.0f} ms",
        f"    P95 latency:                 {summary.p95_latency_ms:.0f} ms",
    ]

    if summary.results_by_domain:
        lines += ["", "  BY DOMAIN:"]
        for domain, metrics in summary.results_by_domain.items():
            lines.append(
                f"    {domain:20s}  n={metrics['count']:<3.0f}  "
                f"retrieval={metrics['retrieval_hit_rate']:.0%}  "
                f"keyword_recall={metrics['mean_keyword_recall']:.0%}  "
                f"latency={metrics['mean_latency_ms']:.0f}ms"
            )

    if summary.results_by_difficulty:
        lines += ["", "  BY DIFFICULTY:"]
        for diff, metrics in summary.results_by_difficulty.items():
            lines.append(
                f"    {diff:20s}  n={metrics['count']:<3.0f}  "
                f"retrieval={metrics['retrieval_hit_rate']:.0%}  "
                f"keyword_recall={metrics['mean_keyword_recall']:.0%}  "
                f"latency={metrics['mean_latency_ms']:.0f}ms"
            )

    lines += ["", "=" * 70]
    report = "\n".join(lines)
    print(report)
    return report


# =========================================================================
# Full evaluation runner
# =========================================================================


def run_evaluation(
    dataset_path: str | Path = "data/eval/eval_dataset.json",
    top_k: int = 5,
    score_threshold: float = 0.25,
) -> EvalSummary:
    """End-to-end evaluation: load dataset, run each query through the pipeline,
    measure metrics, print summary.

    Requires a populated FAISS index (call after startup).
    """
    # Import here to avoid circular dependency at module level
    from rag.index import load_index
    from rag.ingest import load_chunks_manifest
    from rag.retrieve import retrieve as do_retrieve
    from rag.llm_gemini import generate_answer, configure_gemini

    cases = load_eval_dataset(dataset_path)

    # Load index
    index, id_map = load_index()
    if index is None:
        print("ERROR: No FAISS index found. Ingest documents first.", file=sys.stderr)
        return EvalSummary()

    chunks = load_chunks_manifest()
    chunks_by_id = {c.chunk_id: c for c in chunks}

    # Configure Gemini (may fail if key missing – eval still measures retrieval)
    gemini_available = True
    try:
        configure_gemini()
    except EnvironmentError:
        gemini_available = False
        print("WARNING: Gemini API key not set. Evaluation will skip LLM metrics.\n", file=sys.stderr)

    results: list[EvalResult] = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case.id}: {case.question[:60]}…")
        t0 = time.perf_counter()

        passages = do_retrieve(
            query=case.question,
            index=index,
            id_map=id_map,
            chunks_by_id=chunks_by_id,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        if gemini_available:
            try:
                answer = generate_answer(case.question, passages)
            except Exception as exc:
                answer = f"[LLM error: {exc}]"
        else:
            # Fallback: concatenate passage texts for keyword eval
            answer = " ".join(p.text for p in passages)

        latency_ms = (time.perf_counter() - t0) * 1000
        result = evaluate_single(case, passages, answer, latency_ms)
        results.append(result)

    summary = compute_summary(results, cases)
    print_report(summary)

    # Save detailed results to JSON
    out_path = Path("data/eval/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "summary": {
                    "total": summary.total,
                    "retrieval_hit_rate": summary.retrieval_hit_rate,
                    "mean_keyword_recall": summary.mean_keyword_recall,
                    "citation_presence_rate": summary.citation_presence_rate,
                    "citation_correctness_rate": summary.citation_correctness_rate,
                    "mean_latency_ms": summary.mean_latency_ms,
                    "p95_latency_ms": summary.p95_latency_ms,
                },
                "results": [
                    {
                        "eval_id": r.eval_id,
                        "retrieval_hit": r.retrieval_hit,
                        "keyword_recall": round(r.keyword_recall, 3),
                        "citation_present": r.citation_present,
                        "citation_correct": r.citation_correct,
                        "latency_ms": round(r.latency_ms, 1),
                        "top_score": round(r.top_retrieval_score, 4),
                        "num_passages": r.num_passages,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"\n  Detailed results saved to {out_path}")
    return summary


# =========================================================================
# CLI entry point
# =========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument(
        "--dataset",
        default="data/eval/eval_dataset.json",
        help="Path to evaluation QA dataset JSON",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        top_k=args.top_k,
        score_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
