"""Labeled-dataset ingestion for the supervised module (SCAFFOLD).

Consumes the same per-dataset ``labels.csv`` defined in
:mod:`mole.data.datasets` ("Data input"), handling PARTIAL coverage natively:
any subset of images may be labeled, the rest are ignored for supervised
purposes. No new data format is ever introduced.

The concrete triplet/pair sampling format is the one decision to settle when
this module is fleshed out (Phase 8).
"""

from __future__ import annotations

from pathlib import Path


def load_labeled_pairs(labels_root: str | Path):
    """Load labeled images grouped by hand_id for pair/triplet sampling."""
    raise NotImplementedError("Supervised dataset ingestion is scaffolded now.")
