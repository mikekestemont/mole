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

MEASURED ON THE FIVE CHARTER ARCHIVES (2026-07-22, `outputs/pooled_final`,
cross-doc relevance, k=2, Δmacro vs no reranking):

    gamma  antwerp brackley flanders   leroy  utrecht |   mean
     0.05  +0.0029  +0.0067  +0.0003 -0.0014  +0.0009 | +0.0019
     0.10  +0.0110  +0.0105  +0.0061 +0.0079  -0.0104 | +0.0050   <- default
     0.20  +0.0193  -0.0007  +0.0038 +0.0181  -0.0270 | +0.0027
     0.40  +0.0233  -0.0007  -0.0083 +0.0277  -0.0383 | +0.0007   <- paper's value

⚠ THE GOVERNING QUANTITY IS NEIGHBOUR PRECISION, NOT FRAGMENTATION. The
plausible prediction — that the most fragmented archives have the most to gain —
is wrong, and measurably so: the two archives that gain are the two with the
*highest* baselines (Antwerp 0.817, Leroy 0.813), while Utrecht (0.621, 86 hands
in one gallery, Top-1 ~0.67) degrades monotonically with gamma. Diffusion
amplifies whatever the neighbourhood already says: where the nearest neighbours
are the right scribe it reinforces true structure, and where a third of them are
the wrong hand it blends a page with another writer and corrupts it.

So gamma buys gain on precise archives at the cost of damage on imprecise ones,
and no single value wins everywhere. The default here is 0.1 rather than the
paper's 0.4 because 0.4 is tuned on Historical-WI (5 pages per writer, Top-1
~0.88) and does not transfer to an 86-hand gallery. **Tune it per archive
against that archive's own held-out hands**, and expect the answer to track its
Top-1. On a low-precision archive the right setting may be "off".
"""

from __future__ import annotations

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def sgr_rerank(X: np.ndarray, *, k: int = 2, layers: int = 1, gamma: float = 0.1,
               groups: np.ndarray | None = None) -> np.ndarray:
    """Similarity Graph Reranking: diffuse each descriptor over its k nearest.

    ``A[i,j] = exp(−(1−s[i,j])²/γ)`` weights the graph, and each vertex is
    updated as ``h_i ← h_i + Σ_{j∈N(i,k)} A[i,j]·h_j`` followed by L2
    normalisation, for ``layers`` rounds. Restricting the sum to the ``k``
    nearest neighbours is what keeps it from aggregating wrong matches, and γ
    controls how much a weak neighbour still contributes: at γ=0.4 a neighbour at
    similarity 0.5 carries weight 0.535, nearly as much as one at 0.9 (0.975),
    which is close to indiscriminate. At γ=0.1 the same pair is 0.082 vs 0.905.
    Default γ=0.1, not the paper's 0.4 — see the module docstring for the measured
    per-archive trade.

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
