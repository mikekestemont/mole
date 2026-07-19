"""Supervised module — SCAFFOLD ONLY (implemented in a later phase).

Interfaces are fixed now so the sklearn-style separation of learning paradigms
is visible from day one. Supervised finetunes are lineage BRANCHES like any
finetune; they never mutate the base, and metric-learned branches produce
embeddings through the same ``mole embed`` machinery.

* :mod:`mole.supervised.metric`   -- triplet-loss finetuning of the backbone.
* :mod:`mole.supervised.probe`    -- linear/classification probes on frozen embeddings.
* :mod:`mole.supervised.datasets` -- ingests per-dataset ``labels.csv`` (partial coverage).
"""

from __future__ import annotations

from mole.supervised.datasets import (
    FeatureCache,
    HandBatchSampler,
    SupervisedIndex,
    SupItem,
    build_feature_cache,
    load_labeled_pairs,
    pair_masks,
    window_descriptors,
)
from mole.supervised.metric import (
    build_head,
    holdout_macro_map,
    masked_supcon,
    train_head,
    train_metric,
)

__all__ = [
    "FeatureCache",
    "HandBatchSampler",
    "SupItem",
    "SupervisedIndex",
    "build_feature_cache",
    "build_head",
    "holdout_macro_map",
    "load_labeled_pairs",
    "masked_supcon",
    "pair_masks",
    "train_head",
    "train_metric",
    "window_descriptors",
]
