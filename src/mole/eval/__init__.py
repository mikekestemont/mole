"""Retrieval benchmark built from whatever partial labels exist."""

from __future__ import annotations

from mole.eval.compare import CompareReport, compare_evals, format_compare
from mole.eval.retrieval import (
    EvalResult,
    RetrievalScores,
    evaluate,
    format_per_hand,
    format_report,
    load_hand_set,
)

__all__ = [
    "CompareReport",
    "EvalResult",
    "RetrievalScores",
    "compare_evals",
    "evaluate",
    "format_compare",
    "format_per_hand",
    "format_report",
    "load_hand_set",
]
