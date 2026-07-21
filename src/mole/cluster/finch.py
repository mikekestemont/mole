"""FINCH — First Integer Neighbour Clustering Hierarchy (Sarfraz et al., CVPR 2019).

Parameter-free clustering: no K, no distance threshold, no stopping criterion to
tune. That is exactly why it fits mole — *how many hands are in the unlabeled pool*
is the thing we do not know, so any method requiring K would beg the question.

Each point is linked to its first (nearest) neighbour; the connected components of
that graph form the finest partition. Cluster means then become the points and the
step repeats, producing a short hierarchy from many small clusters up to a few large
ones. Typically 4-8 levels, and each level is a full clustering of every document.

**Adjacency note.** The paper defines ``A(i,j)=1`` when ``j == k_i`` or ``k_j == i``
or ``k_i == k_j`` (a shared first neighbour). The connected components of that graph
are *identical* to those of the plain undirected edge set ``{(i, k_i)}``: if
``k_i == k_j == m`` then edges ``i-m`` and ``j-m`` already put i and j in one
component. So we build only the n edges and never materialise the shared-neighbour
pairs — same partition, O(n) memory instead of O(n^2) in the worst case.

Use with page embeddings from ``mole embed``; ``cluster_agreement`` scores a partition
against whatever partial ground truth exists (purity / NMI / ARI over labeled points
only), which is how we tell "discovered structure" from "recovered known hands".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FinchResult:
    """One FINCH run: a hierarchy of partitions over the same N points."""

    partitions: list[np.ndarray]          # each [N] int labels, level 0 = finest
    n_clusters: list[int]
    metric: str
    agreement: list[dict] = field(default_factory=list)   # filled by the CLI when labels exist

    def __len__(self) -> int:
        return len(self.partitions)

    def partition(self, level: int) -> np.ndarray:
        return self.partitions[level]


def _first_neighbours(x: np.ndarray, metric: str, chunk: int = 1024) -> np.ndarray:
    """Index of each point's nearest OTHER point (pure NumPy, chunked).

    Brute force is O(n^2) in time but chunked to O(chunk*n) in memory, which is the
    right trade at mole's scale (a few thousand documents per archive). An ANN backend
    would be needed well beyond ~50k documents.
    """
    n = len(x)
    if n < 2:
        return np.zeros(n, dtype=np.int64)
    out = np.empty(n, dtype=np.int64)
    if metric == "cosine":
        xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sim = xn[s:e] @ xn.T                       # higher = nearer
            sim[np.arange(e - s), np.arange(s, e)] = -np.inf   # never pick self
            out[s:e] = sim.argmax(1)
    elif metric == "euclidean":
        sq = (x * x).sum(1)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            # |a-b|^2 = |a|^2 - 2ab + |b|^2; rounding can push near-zero entries
            # slightly negative, so clamp before use.
            d2 = sq[s:e, None] - 2.0 * (x[s:e] @ x.T) + sq[None, :]
            np.maximum(d2, 0.0, out=d2)
            d2[np.arange(e - s), np.arange(s, e)] = np.inf
            out[s:e] = d2.argmin(1)
    else:
        raise ValueError(f"unknown metric {metric!r} (cosine|euclidean)")
    return out


def _components(kappa: np.ndarray) -> np.ndarray:
    """Connected components of the undirected first-neighbour graph (union-find).

    The graph is exactly ``n`` edges ``(i, kappa[i])``, so union-find is both simpler
    and lighter than building a sparse matrix — and keeps this module NumPy-only.
    """
    n = len(kappa)
    parent = np.arange(n, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]          # path halving
            a = parent[a]
        return a

    for i in range(n):
        ra, rb = find(i), find(int(kappa[i]))
        if ra != rb:
            parent[ra] = rb
    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def _contingency(true: np.ndarray, pred: np.ndarray):
    """Contingency counts plus both marginals, for NMI/ARI."""
    _, ti = np.unique(true, return_inverse=True)
    _, pi = np.unique(pred, return_inverse=True)
    table = np.zeros((ti.max() + 1, pi.max() + 1), dtype=np.int64)
    np.add.at(table, (ti, pi), 1)
    return table, table.sum(1), table.sum(0)


def _nmi(table, a, b) -> float:
    """Normalized mutual information (arithmetic normalisation, as sklearn defaults)."""
    n = table.sum()
    nz = table > 0
    mi = float((table[nz] / n * np.log(table[nz] * n / np.outer(a, b)[nz])).sum())
    ha = -float((a[a > 0] / n * np.log(a[a > 0] / n)).sum())
    hb = -float((b[b > 0] / n * np.log(b[b > 0] / n)).sum())
    denom = (ha + hb) / 2.0
    return mi / denom if denom > 0 else 0.0


def _ari(table, a, b) -> float:
    """Adjusted Rand index from the contingency table."""
    def c2(v):
        v = np.asarray(v, dtype=np.float64)
        return (v * (v - 1) / 2).sum()

    n = table.sum()
    index = c2(table.ravel())
    exp = c2(a) * c2(b) / c2(np.array([n]))
    mx = 0.5 * (c2(a) + c2(b))
    return float((index - exp) / (mx - exp)) if mx != exp else 0.0


def _centroids(x: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Mean vector per cluster (ordered by label id)."""
    dim = x.shape[1]
    out = np.zeros((n_clusters, dim), dtype=np.float32)
    counts = np.zeros(n_clusters, dtype=np.int64)
    np.add.at(out, labels, x)
    np.add.at(counts, labels, 1)
    return out / np.maximum(counts, 1)[:, None]


def finch(x: np.ndarray, metric: str = "cosine", max_levels: int = 20) -> FinchResult:
    """Cluster ``x`` ``[N, D]`` into a parameter-free hierarchy of partitions.

    Returns partitions from finest (level 0) to coarsest. Recursion stops when a step
    stops merging or a single cluster remains, so the number of levels is data-driven.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected [N, D] embeddings, got shape {x.shape}")
    n = len(x)
    if n < 2:
        return FinchResult([np.zeros(n, dtype=np.int64)], [max(n, 1)], metric)

    labels = _components(_first_neighbours(x, metric))
    n_c = int(labels.max()) + 1
    partitions, counts = [labels], [n_c]

    while n_c > 1 and len(partitions) < max_levels:
        cent = _centroids(x, labels, n_c)
        merged = _components(_first_neighbours(cent, metric))
        new_n = int(merged.max()) + 1
        if new_n >= n_c:                     # no progress — stop rather than spin
            break
        labels = merged[labels]              # lift the cluster-level merge to points
        n_c = new_n
        partitions.append(labels)
        counts.append(n_c)

    return FinchResult(partitions, counts, metric)


def cluster_agreement(labels_true: list[str | None], labels_pred: np.ndarray) -> dict:
    """Score a partition against PARTIAL ground truth (labeled points only).

    ``labels_true`` may contain ``None`` for unlabeled documents; those are excluded
    from every statistic (they are not negatives — an unlabeled document may well
    belong to a labeled hand, so counting it as a mistake would be wrong).

    * **purity**   — fraction of labeled points whose cluster's majority hand is theirs.
                     Rises trivially as clusters get smaller; read with n_clusters.
    * **nmi/ari**  — standard clustering agreement, penalising over-fragmentation.
    """
    mask = np.array([t is not None for t in labels_true], dtype=bool)
    n_labeled = int(mask.sum())
    if n_labeled == 0:
        return {"n_labeled": 0, "purity": None, "nmi": None, "ari": None,
                "n_clusters_covering_labels": 0}

    true = np.array([t for t in labels_true if t is not None])
    pred = np.asarray(labels_pred)[mask]

    table, a, b = _contingency(true, pred)
    correct = int(table.max(axis=0).sum())      # majority hand within each cluster
    return {
        "n_labeled": n_labeled,
        "purity": correct / n_labeled,
        "nmi": _nmi(table, a, b),
        "ari": _ari(table, a, b),
        "n_clusters_covering_labels": int(len(np.unique(pred))),
    }
