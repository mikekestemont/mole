"""Post-hoc reranking of page descriptors — a retrieval stage, not an embedding.

Reranking exploits the structure of the *gallery*: a page borrows a little of its
neighbours' geometry, which pulls a scattered cluster together. That is a
different kind of claim from an embedding improvement — it is contingent on what
else is in the corpus, and it does not travel to a new archive on its own. So it
lives here, as an explicit opt-in at eval time, and never touches what
``mole embed`` writes. Every stored vector stays exactly as it was, and a
reranked number is always reported alongside the plain one rather than replacing
it.

SGR (Peer, Kleber & Sablatnig, *Towards Writer Retrieval for Historical
Datasets*, ICDAR 2023, arXiv:2305.05358) is the method implemented here. On
Historical-WI they report mAP 73.4 → 80.6 from reranking alone.

Why it is worth trying on this corpus specifically: FINCH on Flanders showed
**fragmentation, not confusion** — a hand's documents split across many
individually *pure* sub-clusters. Diffusion over a neighbour graph is close to a
direct remedy for that failure mode.
"""

from __future__ import annotations

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def sgr_rerank(X: np.ndarray, *, k: int = 2, layers: int = 1, gamma: float = 0.4,
               groups: np.ndarray | None = None) -> np.ndarray:
    """Similarity Graph Reranking: diffuse each descriptor over its k nearest.

    ``A[i,j] = exp(−(1−s[i,j])²/γ)`` weights the graph, and each vertex is
    updated as ``h_i ← h_i + Σ_{j∈N(i,k)} A[i,j]·h_j`` followed by L2
    normalisation, for ``layers`` rounds. Restricting the sum to the ``k``
    nearest neighbours is what keeps it from aggregating wrong matches; the
    published defaults (k=2, L=1, γ=0.4) are deliberately conservative.

    ``groups`` (length-N labels, e.g. document ids) excludes same-group
    neighbours from the graph. **This is not optional on this corpus**: 56
    charters here are multiple scans of one physical document, a page's nearest
    neighbour is very often its own sibling, and diffusing across near-duplicates
    would manufacture a gain out of the scan shortcut that ``--cross-doc-only``
    exists to suppress.

    Returns a new ``[N, D]`` matrix; the input is not modified.
    """
    if k < 1 or layers < 1:
        raise ValueError(f"k and layers must be >= 1, got k={k}, layers={layers}")
    n = len(X)
    if n < 2:
        return _l2(np.asarray(X, dtype=np.float32))

    H = _l2(np.asarray(X, dtype=np.float32))
    sim = H @ H.T
    weight = np.exp(-((1.0 - sim) ** 2) / gamma)

    # Neighbour eligibility: never self, never a sibling scan.
    eligible = ~np.eye(n, dtype=bool)
    if groups is not None:
        g = np.asarray(groups, dtype=object)
        eligible &= (g[:, None] != g[None, :])

    ranked = np.where(eligible, sim, -np.inf)
    kk = min(k, n - 1)
    nbrs = np.argpartition(-ranked, kk - 1, axis=1)[:, :kk]           # [n, k]
    rows = np.arange(n)[:, None]
    valid = np.isfinite(ranked[rows, nbrs])                           # empty-group guard
    w = np.where(valid, weight[rows, nbrs], 0.0).astype(np.float32)

    for _ in range(layers):
        H = _l2(H + np.einsum("ij,ijd->id", w, H[nbrs]))
    return H


RERANKERS = {"sgr": sgr_rerank}


def apply_rerank(X: np.ndarray, method: str, *, groups=None, **kw) -> np.ndarray:
    if method not in RERANKERS:
        raise ValueError(f"unknown reranker {method!r}; available: {sorted(RERANKERS)}")
    return RERANKERS[method](X, groups=groups, **kw)
