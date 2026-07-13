"""Retrieval benchmark from partial labels.

Each labeled image queries a gallery of all other labeled images; reports mAP
and top-k accuracy, overall and per dataset. A cross-dataset breakdown (same
hand matched across different digitizations vs. within one) is mandatory — it is
the confound detector for repository/scan-quality shortcuts.

Scores are written into the lineage registry entry of the evaluated model, so
forgetting is visible across continual updates.
"""

from __future__ import annotations

from pathlib import Path


def evaluate(embeddings_path: str | Path, datasets_root: str | Path,
             model_id: str | None = None):
    """Run the retrieval benchmark and (optionally) record scores in the registry."""
    raise NotImplementedError("Evaluation is implemented in Phase 6+.")
