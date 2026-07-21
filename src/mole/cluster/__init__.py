"""Unsupervised clustering of page embeddings (parameter-free FINCH)."""

from __future__ import annotations

from mole.cluster.finch import FinchResult, cluster_agreement, finch

__all__ = ["FinchResult", "cluster_agreement", "finch"]
