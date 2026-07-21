"""Retrieval benchmark built from whatever partial labels exist."""

from __future__ import annotations

from mole.eval.compare import (
    CompareReport,
    MultiCompareReport,
    compare_evals,
    compare_evals_multi,
    format_compare,
    format_multi_compare,
)
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
    "MultiCompareReport",
    "compare_evals",
    "compare_evals_multi",
    "evaluate",
    "format_compare",
    "format_multi_compare",
    "format_per_hand",
    "format_report",
    "load_hand_set",
]
