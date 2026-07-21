"""Supervised metric learning ON the aggregated document descriptor.

The Tier-1 head shapes *window* descriptors — the mean of a window's foreground
tokens — and is then deployed under VLAD, which discards exactly that statistic
in favour of residuals to K centroids. Measured by leave-one-archive-out, that
mismatch is fatal: the same heads score **+0.066** macro-mAP when the readout is
mean pooling and **−0.013** when it is VLAD (2026-07-21). Supervision transfers
across collections; the aggregator throws it away.

This module puts the loss *after* aggregation instead. The unit of supervision is
the finished VLAD document vector, so the thing being optimised is the thing
retrieval actually ranks. It is deliberately cheap — no backbone, no GPU, no
token cache — and reuses the Tier-1 machinery wholesale: the doc vectors are
wrapped in a :class:`~mole.supervised.datasets.FeatureCache` (one "window" per
document), which makes :class:`~mole.supervised.datasets.HandBatchSampler`,
:func:`~mole.supervised.datasets.pair_masks` and
:func:`~mole.supervised.metric.train_head` apply unchanged — including the
cross-document positive rule and the confirmed-negatives denominator.

**One shared codebook is mandatory.** Per-archive transductive VLAD spaces are
not comparable — dimension *i* means a different centroid in each archive — so a
projection fitted on four archives is meaningless on the fifth. Feed this the
universal-codebook embeddings (``mole codebook`` / ``mole embed --codebook-from``),
never ``pooled_final``.

**PCA-whitening is the control, not a step to skip.** Whitening a VLAD descriptor
is a strong unsupervised trick in its own right (the writer-retrieval standard,
and Raven's own pipeline whitens to 384). Any supervised gain has to be measured
against whitening alone at the *same* output dimensionality, or it is just
whitening wearing a medal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


class PCAWhiten:
    """PCA-whitening fitted on one set of documents and applied to another.

    Fitted on the TRAINING archives only. Whitening is transductive by habit in
    this codebase (``mole embed --whiten``), but under leave-one-archive-out the
    held-out archive must not influence the transform any more than it influences
    the loss — otherwise the "a new collection arrives" claim is contaminated by
    the new collection.

    ⚠ **Truncation is the dangerous knob, not the whitening.** Writer identity is
    a LOW-variance property: nuisance factors (layout, ink density, page size)
    dominate the variance, so the discriminative directions sit far down the
    spectrum — on the synthetic corpus in ``tests/test_doc_metric.py`` they are
    components 40-43 of 44. Keeping the top-k by variance therefore *discards the
    signal*, which is precisely backwards. Whitening then rescues what survives by
    equalising variance. So keep ``dim`` generous (default: as close to full rank
    as the training set allows) and treat any reduction as a hyper-parameter to
    sweep, never a default to trust. :meth:`truncated` makes a sweep nearly free.
    """

    def __init__(self, dim: int | None = None, eps: float = 1e-6):
        self.dim = dim                 # None = keep as much rank as there is
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def truncated(self, k: int) -> "PCAWhiten":
        """A copy keeping only the top ``k`` components — for sweeping cheaply.

        Refitting per candidate dimensionality would repeat the SVD; slicing an
        already-fitted transform gives an identical result for free.
        """
        if self.components_ is None:
            raise RuntimeError("truncated() before fit")
        out = PCAWhiten(dim=k, eps=self.eps)
        out.mean_ = self.mean_
        out.components_ = self.components_[:k]
        out.scale_ = self.scale_[:k]
        return out

    def fit(self, X: np.ndarray) -> "PCAWhiten":
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(0, keepdims=True)
        Xc = X - self.mean_
        full = max(1, min(Xc.shape) - 1)
        k = full if self.dim is None else min(self.dim, full)
        # economy SVD: [n, d] with d >> n is fine, we only keep k components
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = vt[:k]
        var = (s[:k] ** 2) / max(len(Xc) - 1, 1)
        self.scale_ = 1.0 / np.sqrt(var + self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("PCAWhiten.transform before fit")
        Xc = np.asarray(X, dtype=np.float32) - self.mean_
        return _l2((Xc @ self.components_.T) * self.scale_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ------------------------------------------------------------------- data load
def load_archive_vectors(paths: list[str | Path]):
    """Load several ``mole embed`` outputs into one array with their labels.

    Every path must have been embedded with the SAME codebook; the caller is
    responsible for that (see the module docstring). Returns
    ``(X, names, hands, docs, archives)`` where ``hands``/``docs`` are namespaced
    by archive, and unlabeled documents carry ``hand == ""``.
    """
    from mole.data.datasets import load_labels
    from mole.data.docids import doc_id_resolver

    Xs, names, hands, docs, archives = [], [], [], [], []
    for p in paths:
        npy = Path(p)
        npy = npy if npy.suffix == ".npy" else npy.with_suffix(".npy")
        X = np.load(npy)
        meta = json.loads(npy.with_suffix(".mapping.json").read_text())
        rows = meta["rows"]
        if len(rows) != len(X):
            raise ValueError(f"{npy}: {len(rows)} rows vs {len(X)} vectors")
        Xs.append(X)
        cache: dict[Path, tuple] = {}
        for r in rows:
            img = Path(r["image"])
            if img.parent not in cache:
                cache[img.parent] = (load_labels(img.parent),
                                     doc_id_resolver(img.parent))
            table, resolve = cache[img.parent]
            arch = img.parent.name
            raw = table.hand_by_filename.get(img.name)
            names.append(img.name)
            archives.append(arch)
            hands.append(f"{arch}/{raw}" if raw else "")
            docs.append(f"{arch}/{resolve(img.name)}")
    dims = {x.shape[1] for x in Xs}
    if len(dims) != 1:
        raise ValueError(f"embeddings have different dimensionalities {dims} — "
                         "they cannot share a codebook, so a metric learned on "
                         "one archive is meaningless on another")
    return (np.concatenate(Xs).astype(np.float32), names,
            np.asarray(hands, dtype=object), np.asarray(docs, dtype=object),
            np.asarray(archives, dtype=object))


# --------------------------------------------------------------------- scoring
def archive_macro_map(X: np.ndarray, hands: np.ndarray, docs: np.ndarray,
                      keep: np.ndarray) -> tuple[float, dict]:
    """Cross-document macro-mAP within one archive (its own gallery).

    Relevance is same hand AND different document, matching
    ``mole eval --cross-doc-only``: a sibling scan of the query earns no credit.
    """
    from mole.eval.retrieval import _rank_metrics

    idx = np.where(keep)[0]
    labeled = idx[np.asarray([bool(hands[i]) for i in idx], dtype=bool)]
    if len(labeled) < 2:
        return float("nan"), {}
    Z = _l2(X[labeled])
    sim = (Z @ Z.T).astype(np.float64)
    d = docs[labeled]
    allow = d[:, None] != d[None, :]          # excludes self AND sibling scans
    scores = _rank_metrics(sim, hands[labeled], allow, (1,))
    if scores is None:
        return float("nan"), {}
    return float(scores.macro_map), {h: v["ap"] for h, v in scores.per_hand.items()}


# ------------------------------------------------------------------- the fitter
def fit_doc_metric(X: np.ndarray, hands: np.ndarray, docs: np.ndarray,
                   archives: np.ndarray, *, holdout_archive: str,
                   whiten_dim: int | None = None, out_dim: int = 128,
                   epochs: int = 60, lr: float = 1e-3, temperature: float = 0.07,
                   holdout_frac: float = 0.2, seed: int = 0,
                   batches_per_epoch: int = 100, progress: bool = False,
                   pca: "PCAWhiten | None" = None):
    """Fit whitening + a supervised projection on every archive EXCEPT one.

    Returns ``(transform, report)``. ``transform(X) -> Z`` maps raw document
    vectors into the learned space; it is fitted without ever seeing the held-out
    archive — not in the whitening, not in the loss, not in model selection
    (which uses a hand-slice of the training archives).
    """
    import torch

    from mole.supervised.datasets import FeatureCache
    from mole.supervised.metric import train_head

    train_rows = archives != holdout_archive
    if not train_rows.any():
        raise ValueError(f"no training rows left after holding out {holdout_archive!r}")

    if pca is None:
        pca = PCAWhiten(dim=whiten_dim).fit(X[train_rows])
    elif whiten_dim is not None:
        pca = pca.truncated(whiten_dim)     # reuse one SVD across a sweep
    Z = pca.transform(X)

    labeled = np.asarray([bool(h) for h in hands], dtype=bool)
    sel = np.where(train_rows & labeled)[0]
    # one "window" per document: the Tier-1 sampler, masks and trainer apply as-is
    cache = FeatureCache(
        descriptors=Z[sel].astype(np.float32),
        window_hand=[str(hands[i]) for i in sel],
        window_doc=[str(docs[i]) for i in sel],
        window_archive=[str(archives[i]) for i in sel],
        window_item=[str(i) for i in sel],
        meta={"model_id": "doc-metric",
              "whiten_dim": int(pca.components_.shape[0])})

    # model selection on a hand-slice of the TRAINING archives (never the test one)
    rng = np.random.RandomState(seed)
    train_hands = sorted({h for h in cache.window_hand if h})
    val_n = max(1, int(round(len(train_hands) * holdout_frac)))
    val_hands = set(rng.permutation(train_hands)[:val_n].tolist())

    head, report = train_head(
        cache, holdout_hands=val_hands, out_dim=out_dim, temperature=temperature,
        kind="linear", seed=seed, epochs=epochs, lr=lr, progress=progress,
        sampler_cfg={"hands_per_batch": 16, "docs_per_hand": 2, "windows_per_doc": 1,
                     "same_archive_frac": 0.5,
                     "batches_per_epoch": batches_per_epoch})
    head.eval()

    def transform(raw: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = head(torch.from_numpy(pca.transform(raw).astype(np.float32)))
        return _l2(z.numpy())

    report["holdout_archive"] = holdout_archive
    report["whiten_dim"] = int(pca.components_.shape[0])
    report["n_train_docs"] = int(len(sel))
    return transform, report
