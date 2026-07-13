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

__all__: list[str] = []
