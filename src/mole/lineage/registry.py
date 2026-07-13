"""Append-only model registry (single JSON or SQLite file in the models root).

Records provenance per checkpoint version and prints the lineage as a tree.
Wired into train/finetune/embed in Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RegistryEntry:
    """Provenance record for one checkpoint version."""

    model_id: str                       # e.g. "base@v2" or "base@v3/stgallen@v1"
    parent_id: str | None
    config_hash: str
    seed: int
    date: str
    dataset_manifests: list[str] = field(default_factory=list)
    replay_composition: dict[str, float] = field(default_factory=dict)
    eval_scores: dict[str, float] = field(default_factory=dict)


def list_models(models_root: str | Path):
    """Print the lineage as a tree (``mole models list``)."""
    raise NotImplementedError("Lineage registry is implemented in Phase 6.")


def show_model(models_root: str | Path, model_id: str):
    """Print full provenance of one checkpoint (``mole models show <id>``)."""
    raise NotImplementedError("Lineage registry is implemented in Phase 6.")
