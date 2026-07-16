"""VLAD encoding with a reproducible, versioned codebook.

Fix (vs. original code): the k-means codebook uses a fixed, configurable seed;
fitting is reproducible and the fitted codebook is SAVED with the run (next to
the embeddings, stamped with the producing model ID in the sidecar) so
embeddings are stable across invocations.

k-means backend: ``scikit-learn`` when installed (a declared dependency); a
compact, seeded NumPy k-means++/Lloyd fallback otherwise, so VLAD never
hard-fails on a slim install and stays reproducible either way.
"""

from __future__ import annotations

import numpy as np


def fit_codebook(descriptors, n_clusters: int = 100, seed: int = 0,
                 max_iter: int = 100, max_descriptors: int = 0):
    """Fit a reproducible k-means codebook on patch descriptors.

    ``descriptors`` is a ``[N, dim]`` float array. Returns the ``[K, dim]``
    cluster centres (float32). Reproducible for a fixed ``seed``.

    By default the fit uses ALL descriptors, matching Raven et al., who "gather all
    foreground tokens from the entire training set" and cluster them with minibatch
    k-means. MiniBatchKMeans streams minibatches from the pool, so a large pool costs
    little beyond the memory already holding it. Set ``max_descriptors`` to a positive
    N to cap the pool at a seeded random subsample of N (a tractability escape hatch —
    note it *is* a deviation from the paper).
    """
    x = np.asarray(descriptors, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"descriptors must be 2-D [N, dim], got shape {x.shape}")
    if max_descriptors and len(x) > max_descriptors:  # seeded subsample for the fit
        rng = np.random.default_rng(seed)
        x = x[rng.choice(len(x), max_descriptors, replace=False)]
    k = int(min(n_clusters, len(x)))
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ModuleNotFoundError:
        return _numpy_kmeans(x, k, seed, max_iter)

    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=10_000,
                         n_init=3, max_iter=max_iter)
    _fit_with_heartbeat(km, x, f"Fitting VLAD codebook (K={k}, {len(x):,} pts)")
    return km.cluster_centers_.astype(np.float32)


def _fit_with_heartbeat(km, x, description: str) -> None:
    """Run ``km.fit(x)`` in a worker thread with a live elapsed-time bar.

    MiniBatchKMeans is a single blocking call with no per-iteration callback, so a
    true ``%`` bar isn't available without reimplementing the fit as a partial_fit
    loop — which would change the codebook numerically. To keep results comparable
    (we A/B retrieval scores against this codebook) we keep the exact fit and just
    show that it's alive and how long it's taken.
    """
    import threading

    from mole.progress import progress_bar

    err: dict = {}

    def _run():
        try:
            km.fit(x)
        except BaseException as e:                    # re-raise in the main thread
            err["e"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    bar = progress_bar(description=description, unit="s", bar_format="{desc} | {elapsed} elapsed")
    while t.is_alive():
        t.join(timeout=0.5)
        bar.refresh()                                 # advance the elapsed clock
    bar.close()
    if "e" in err:
        raise err["e"]


def _numpy_kmeans(x: np.ndarray, k: int, seed: int, max_iter: int) -> np.ndarray:
    """Seeded k-means++ init + Lloyd iterations (reproducible fallback)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    # k-means++ seeding.
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = x[rng.integers(n)]
    closest = ((x - centers[0]) ** 2).sum(1)
    for i in range(1, k):
        probs = closest / closest.sum() if closest.sum() > 0 else None
        idx = rng.choice(n, p=probs)
        centers[i] = x[idx]
        closest = np.minimum(closest, ((x - centers[i]) ** 2).sum(1))
    # Lloyd iterations.
    from mole.progress import track
    for _ in track(range(max_iter), "Fitting VLAD codebook (Lloyd)", unit="iter"):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        assign = d.argmin(1)
        moved = False
        for c in range(k):
            members = x[assign == c]
            if len(members):
                new = members.mean(0)
                if not np.allclose(new, centers[c]):
                    centers[c] = new
                    moved = True
        if not moved:
            break
    return centers.astype(np.float32)


def vlad_encode(descriptors, codebook, powernorm: bool = True,
                intra_norm: bool = True) -> np.ndarray:
    """VLAD-encode a page's descriptors against a fitted codebook.

    Aggregates residuals (descriptor - nearest centre) per cluster, optionally
    intra-normalises each cluster block, then optional signed power-norm and a
    final global L2. Returns a flat ``[K * dim]`` float32 vector.

    ``intra_norm=False`` reproduces Raven et al.'s plain VLAD (residual sum →
    power-norm → global L2, no per-cluster normalisation).
    """
    x = np.asarray(descriptors, dtype=np.float32)
    c = np.asarray(codebook, dtype=np.float32)
    k, dim = c.shape
    if len(x) == 0:
        return np.zeros(k * dim, dtype=np.float32)
    # Nearest centre via ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x.c^T, giving an [P,K]
    # matrix directly — avoids materialising the [P,K,dim] broadcast (GBs per page).
    d = (x * x).sum(1)[:, None] + (c * c).sum(1)[None, :] - 2.0 * (x @ c.T)
    assign = d.argmin(1)
    vlad = np.zeros((k, dim), dtype=np.float32)
    for i in range(k):
        members = x[assign == i]
        if len(members):
            vlad[i] = (members - c[i]).sum(0)
    # Intra-normalisation (per-cluster L2). Skipped for Raven-parity plain VLAD.
    if intra_norm:
        norms = np.linalg.norm(vlad, axis=1, keepdims=True)
        vlad = vlad / np.maximum(norms, 1e-12)
    vlad = vlad.reshape(-1)
    if powernorm:
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    vlad = vlad / max(np.linalg.norm(vlad), 1e-12)
    return vlad.astype(np.float32)
