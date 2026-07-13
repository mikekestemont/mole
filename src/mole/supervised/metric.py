"""Triplet / metric-learning finetuning of the backbone (SCAFFOLD).

Produces a lineage branch; consumes per-dataset ``labels.csv`` with partial
coverage. No training logic in this phase — interface only.
"""

from __future__ import annotations

from pathlib import Path


def train_metric(config_path: str | Path, base_checkpoint: str | Path,
                 labels_root: str | Path, output_dir: str | Path | None = None):
    """Finetune the backbone with a triplet loss (+ miner) on labeled hands."""
    raise NotImplementedError("Metric learning is scaffolded now, implemented later.")
