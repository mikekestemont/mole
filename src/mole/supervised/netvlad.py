"""Tier 2 — differentiable VLAD (NetVLAD) over cached page tokens.

The Phase-2 result (SUPERVISED_PLAN.md §0a) is the whole motivation: supervision
transfers across archives (+0.066 macro under mean pooling, all five positive),
but it dies at the aggregator (−0.013 under VLAD), and a linear map on the
finished VLAD vector recovers nothing (F3). What the head optimises — a window
mean — is precisely what hard-assignment VLAD discards. NetVLAD replaces the
``argmin`` with a softmax so the loss runs *through* the aggregation: what is
optimised is what retrieval ranks.

The design property that makes the experiment readable is
:meth:`NetVLAD.from_codebook`. Initialised from a k-means codebook ``C`` with
``w = 2αC`` and ``b = −α‖C‖²``, the softmax assignment converges to the hard
nearest-centre assignment as ``α → ∞``::

    argmax_k (2α·c_k·x − α‖c_k‖²) = argmin_k ‖x − c_k‖²   (‖x‖² is common to all k)

so a sufficiently sharp untrained NetVLAD reproduces the frozen-codebook
baseline (asserted in ``tests/test_netvlad.py``) and Δ measures learning rather
than a changed aggregator.

⚠ How sharp is "sufficiently" is not obvious and must be *measured*, never
assumed. Assignment entropy is a misleading proxy: on real geometry a softmax
can be almost one-hot (entropy ≈ 0) while the resulting descriptor is nearly
orthogonal to hard VLAD, because the residual to a *distant* centre is large and
does not cancel, whereas the correct cluster's residuals mostly do. A 1%
assignment leak onto a far centre can therefore outweigh the entire correct
cluster. :func:`vlad_fidelity` measures the thing that matters (cosine to hard
VLAD) and :func:`alpha_for_codebook` calibrates α against it; the trainer records
``init_fidelity`` in its report and warns when it drops below 0.99. The LOAO
driver additionally evaluates the *untrained* module as its own reference point,
so ``trained − init`` stays confound-free at any α.

The α trade-off is real and is the crux of the experiment: high α keeps the
aggregator faithful but nearly saturates the softmax, so the assignment gradient
shrinks while the centroids — which enter the residual linearly — keep training.
``grad_assign`` / ``grad_centroids`` are logged per epoch for exactly this
reason, and the ``learn`` ablation separates the two effects.

Output dim is unchanged (``K*dim`` = 38,400 at K=100), power-norm + global L2,
``intra_norm`` off — i.e. deployed Raven-plain VLAD — so every existing
``mole eval`` / ``mole eval-compare`` invocation works on the result untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mole.progress import track
from mole.supervised.tokens import TokenCache


# --------------------------------------------------------------------- alpha
def assignment_gap(codebook: np.ndarray, descriptors: np.ndarray) -> float:
    """Mean squared-distance margin between the best and second-best centre.

    ``1/gap`` is the natural scale for α — it is the α at which the runner-up
    centre carries ``1/e`` of the softmax mass — so it makes a good unit for the
    search in :func:`alpha_for_codebook`.
    """
    x = np.asarray(descriptors, dtype=np.float32)
    c = np.asarray(codebook, dtype=np.float32)
    d = ((x * x).sum(1)[:, None] + (c * c).sum(1)[None, :] - 2.0 * (x @ c.T))
    part = np.partition(d, 1, axis=1)[:, :2]
    return float(np.mean(np.maximum(part[:, 1] - part[:, 0], 1e-6)))


def vlad_fidelity(pages, codebook: np.ndarray, alpha: float, *,
                  intra_norm: bool = False) -> float:
    """Mean cosine between soft-assignment VLAD at ``alpha`` and hard VLAD.

    This — not the assignment entropy — is the number that says whether an
    untrained NetVLAD *is* the frozen-codebook baseline. The two come apart
    badly: a softmax can be almost one-hot (entropy ≈ 0) while the descriptor is
    nearly orthogonal to hard VLAD, because the residual to a *distant* centre is
    large and, unlike the true cluster's residuals, does not cancel. A 1%
    assignment leak onto a far centre therefore contributes more to the
    descriptor than the entire correct cluster does.
    """
    from mole.embed.vlad import vlad_encode

    model = NetVLAD.from_codebook(codebook, alpha, intra_norm=intra_norm)
    cos = []
    with torch.no_grad():
        for x in pages:
            x = np.asarray(x, dtype=np.float32)
            if len(x) == 0:
                continue
            hard = vlad_encode(x, codebook, intra_norm=intra_norm)
            soft = model(torch.from_numpy(x)).numpy()
            n = np.linalg.norm(hard) * np.linalg.norm(soft)
            if n > 0:
                cos.append(float(hard @ soft / n))
    return float(np.mean(cos)) if cos else 0.0


def alpha_for_codebook(codebook: np.ndarray, pages, *, target_cos: float = 0.999,
                       intra_norm: bool = False, max_steps: int = 24) -> float:
    """Smallest α on a geometric ladder whose descriptor matches hard VLAD.

    α trades faithfulness against gradient. Calibrating it on *assignment* mass
    (the obvious reading of Arandjelović et al.) is wrong here — see
    :func:`vlad_fidelity` — so we calibrate on the descriptor itself: walk α up
    from the ``1/gap`` scale until mean cosine to hard VLAD reaches
    ``target_cos``, and take the smallest such α, which is the most trainable
    aggregator that is still the baseline.

    ``pages`` is a sequence of ``[N, dim]`` token blocks (one per page); fidelity
    is a per-page property, so passing one big pooled array would measure the
    wrong thing.
    """
    pages = [np.asarray(p, dtype=np.float32) for p in pages]
    pooled = np.vstack([p for p in pages if len(p)]) if pages else np.zeros((0, 1))
    base = 1.0 / max(assignment_gap(codebook, pooled), 1e-12)
    alpha = base
    for _ in range(max_steps):
        if vlad_fidelity(pages, codebook, alpha, intra_norm=intra_norm) >= target_cos:
            return float(alpha)
        alpha *= 2.0
    return float(alpha)


# --------------------------------------------------------------------- module
class NetVLAD(nn.Module):
    """Soft-assignment VLAD with learnable centres and assignment.

    ``forward(x)`` takes ``[P, T, dim]`` (a batch of pages, ``T`` tokens each) or
    ``[T, dim]`` (one page) and returns L2-normalised ``[P, K*dim]`` descriptors,
    identical in layout and normalisation to
    :func:`mole.embed.vlad.vlad_encode`.

    ``learn`` selects what is trainable: ``both`` (default), ``assign`` (the
    routing only — centres stay at the k-means solution), or ``centroids``. The
    ablation is the diagnostic for *where* aggregation headroom actually lives.
    """

    def __init__(self, num_clusters: int, dim: int, alpha: float = 100.0,
                 intra_norm: bool = False, powernorm: bool = True,
                 learn: str = "both"):
        super().__init__()
        if learn not in ("both", "assign", "centroids"):
            raise ValueError(
                f"learn must be 'both' | 'assign' | 'centroids', got {learn!r}")
        self.num_clusters, self.dim = int(num_clusters), int(dim)
        self.alpha, self.intra_norm, self.powernorm = float(alpha), intra_norm, powernorm
        self.learn = learn
        self.centroids = nn.Parameter(torch.zeros(num_clusters, dim))
        self.assign_w = nn.Parameter(torch.zeros(num_clusters, dim))
        self.assign_b = nn.Parameter(torch.zeros(num_clusters))
        self.centroids.requires_grad_(learn in ("both", "centroids"))
        self.assign_w.requires_grad_(learn in ("both", "assign"))
        self.assign_b.requires_grad_(learn in ("both", "assign"))

    # ----------------------------------------------------------------- init
    @classmethod
    def from_codebook(cls, codebook, alpha: float, *, intra_norm: bool = False,
                      powernorm: bool = True, learn: str = "both") -> "NetVLAD":
        """Initialise so that soft assignment ≈ hard assignment against ``codebook``.

        ``w = 2αC``, ``b = −α‖C‖²`` ⇒ ``argmax_k (w_k·x + b_k) = argmin_k ‖x−c_k‖²``.
        At large α the module therefore *is* the frozen-codebook baseline.
        """
        c = torch.as_tensor(np.asarray(codebook, dtype=np.float32))
        k, d = c.shape
        m = cls(k, d, alpha=alpha, intra_norm=intra_norm, powernorm=powernorm, learn=learn)
        with torch.no_grad():
            m.centroids.copy_(c)
            m.assign_w.copy_(2.0 * alpha * c)
            m.assign_b.copy_(-alpha * (c * c).sum(dim=1))
        return m

    def codebook(self) -> np.ndarray:
        """The learned centres — the artifact a hard-VLAD deploy would use."""
        return self.centroids.detach().cpu().numpy().astype(np.float32)

    # -------------------------------------------------------------- forward
    def assignments(self, x):
        return torch.softmax(x @ self.assign_w.t() + self.assign_b, dim=-1)

    def forward(self, x):
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)                              # [1, T, d]
        a = self.assignments(x)                             # [P, T, K]
        # V[p,k,:] = Σ_t a[p,t,k]·x[p,t,:] − c[k]·Σ_t a[p,t,k]
        v = a.transpose(1, 2) @ x                           # [P, K, d]
        v = v - a.sum(dim=1).unsqueeze(-1) * self.centroids.unsqueeze(0)
        if self.intra_norm:
            v = torch.nn.functional.normalize(v, dim=2)
        v = v.reshape(v.shape[0], -1)                       # [P, K*d]
        if self.powernorm:
            v = torch.sign(v) * torch.sqrt(torch.abs(v) + 1e-12)
        v = torch.nn.functional.normalize(v, dim=1)
        return v.squeeze(0) if single else v

    def assignment_entropy(self, x) -> float:
        """Mean softmax entropy in nats — the α saturation tripwire.

        ~0 means the assignment has gone one-hot and gradients have vanished (α
        too large); ~ln(K) means it has blurred toward a mean, the statistic we
        already know fails. Logged every epoch so a null result can be attributed
        to one or the other rather than left ambiguous.
        """
        with torch.no_grad():
            a = self.assignments(x if x.dim() == 3 else x.unsqueeze(0))
            return float(-(a * torch.log(a + 1e-12)).sum(-1).mean())


def _grad_norm(p) -> float:
    return float(p.grad.norm()) if p.grad is not None else 0.0


# ------------------------------------------------------------- page vectors
def vlad_page_vectors(cache: TokenCache, codebook: np.ndarray,
                      rows: list[int] | None = None, *, intra_norm: bool = False,
                      progress: bool = True) -> np.ndarray:
    """Plain hard-assignment VLAD from the cache — the baseline, via production code.

    Calls :func:`mole.embed.vlad.vlad_encode` itself rather than reimplementing
    it, so the fold baseline is the same function that produced
    ``outputs/pooled_final``. Run this with an archive's own transductive
    codebook to check the cache's token cap reproduces the known numbers.
    """
    from mole.embed import vlad as _vlad

    rows = list(range(cache.n_pages)) if rows is None else rows
    k, d = np.asarray(codebook).shape
    out = np.zeros((len(rows), k * d), np.float32)
    for j, i in enumerate(track(rows, "VLAD (hard)", unit="page", disable=not progress)):
        out[j] = _vlad.vlad_encode(cache.page_tokens(i), codebook, intra_norm=intra_norm)
    return out


def netvlad_page_vectors(model, cache: TokenCache, rows: list[int] | None = None, *,
                         device=None, max_tokens: int = 0,
                         progress: bool = True) -> np.ndarray:
    """Page embeddings through the (trained) soft aggregator, all tokens by default."""
    rows = list(range(cache.n_pages)) if rows is None else rows
    dev = device or torch.device("cpu")
    model = model.to(dev).eval()
    out = np.zeros((len(rows), model.num_clusters * model.dim), np.float32)
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for j, i in enumerate(track(rows, "VLAD (learned)", unit="page",
                                    disable=not progress)):
            x = (cache.sample_tokens(i, max_tokens, rng) if max_tokens
                 else cache.page_tokens(i))
            if len(x) == 0:
                continue                                  # blank page → zero vector
            out[j] = model(torch.from_numpy(x).to(dev)).cpu().numpy()
    return out


# ------------------------------------------------------- eval-compatible output
def write_embeddings(output: str | Path, matrix: np.ndarray, cache: TokenCache,
                     rows: list[int], meta: dict, *,
                     dataset_dir: str | Path | None = None) -> Path:
    """Write ``matrix`` + a ``.mapping.json`` that ``mole eval`` reads.

    ``dataset_dir`` rewrites each row's image path into that folder (the token
    cache is built over a pooled symlink tree, while evals run against the
    per-archive directory). Basenames are preserved, so label matching is
    identical to a real `mole embed` on that folder.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output if output.suffix == ".npy" else output.with_suffix(".npy"), matrix)

    dd = Path(dataset_dir) if dataset_dir else None
    row_records = []
    for r, i in enumerate(rows):
        p = Path(cache.pages[i]["item"])
        row_records.append({"row": r, "image": str(dd / p.name if dd else p),
                            "n_tokens": cache.pages[i]["count"]})
    sidecar = {
        **cache.meta, **meta,
        "embed_matrix_shape": list(matrix.shape),
        "n_rows": len(row_records),
        "foreground_filter": True,
        "foreground_method": cache.meta.get("fg_method"),
        "foreground_threshold": cache.meta.get("fg_threshold"),
        "source": "token-cache",
        "rows": row_records,
    }
    output.with_suffix(".mapping.json").write_text(json.dumps(sidecar, indent=2))
    print(f"[mole] ✓ wrote {tuple(matrix.shape)} embeddings → {output}")
    return output


# ------------------------------------------------------------------ training
class _PageView:
    """Duck-typed :class:`~mole.supervised.datasets.FeatureCache` whose rows are PAGES.

    :class:`~mole.supervised.datasets.HandBatchSampler` only reads
    ``window_hand`` / ``window_doc`` / ``window_archive``, so pointing it at one
    entry per page (with ``windows_per_doc=1``) reuses its tested structural
    guarantees verbatim: only hands with ≥``docs_per_hand`` distinct documents
    are anchors, unlabeled rows are never drawn, and ``same_archive_frac``
    keeps negatives from collapsing to the trivial cross-archive contrast.
    """

    def __init__(self, cache: TokenCache, rows: list[int]):
        self.rows = rows
        self.window_hand = [cache.pages[i]["hand"] for i in rows]
        self.window_doc = [cache.pages[i]["doc"] for i in rows]
        self.window_archive = [cache.pages[i]["archive"] for i in rows]


def _holdout_macro_map(model, cache: TokenCache, rows: list[int], device,
                       *, max_tokens: int = 0) -> float:
    """Cross-document macro-mAP over ``rows`` — the model-selection metric.

    Page-level with same-hand-different-document relevance: the same rule as
    ``mole eval --cross-doc-only``, so the number the trainer selects on and the
    number the experiment reports measure the same thing.
    """
    from mole.eval.retrieval import _rank_metrics, _similarity

    if len(rows) < 2:
        return 0.0
    emb = netvlad_page_vectors(model, cache, rows, device=device,
                               max_tokens=max_tokens, progress=False)
    labels = np.asarray([cache.pages[i]["hand"] for i in rows], dtype=object)
    docs = np.asarray([cache.pages[i]["doc"] for i in rows], dtype=object)
    sim = _similarity(emb.astype(np.float64), "cosine")
    allow = (docs[:, None] != docs[None, :])      # excludes self AND sibling scans
    scores = _rank_metrics(sim, labels, allow, (1,))
    return float(scores.macro_map) if scores else 0.0


def train_netvlad(cache: TokenCache, codebook: np.ndarray, *,
                  holdout_hands: set[str],
                  exclude_hands: set[str] | frozenset = frozenset(),
                  alpha: float | None = None, learn: str = "both",
                  temperature: float = 0.07, tokens_per_page: int = 512,
                  sampler_cfg: dict | None = None, seed: int = 0, epochs: int = 20,
                  lr: float = 1e-3, weight_decay: float = 1e-4,
                  select_max_tokens: int = 0,
                  device: str | None = None, progress: bool = True):
    """Train NetVLAD on cached page tokens; return ``(model, report)``.

    Mirrors :func:`mole.supervised.metric.train_head` — masked SupCon, train on
    train-hand pages only, model-select on held-out hands, ``exclude_hands`` is
    the leave-one-archive-out fold (dropped from training *and* selection so
    train / select / test stay three disjoint sets of hands).

    The unit of the loss is a **page**, not a window: retrieval ranks pages, so a
    training sample must be one. That is the correction Tier 1 needed.
    """
    from mole.supervised.datasets import HandBatchSampler, pair_masks
    from mole.supervised.metric import masked_supcon

    dev = torch.device(device) if device else torch.device("cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    holdout_hands, exclude_hands = set(holdout_hands), set(exclude_hands)
    all_hands = {p["hand"] for p in cache.pages if p["hand"]}
    train_hands = all_hands - holdout_hands - exclude_hands

    train_rows = cache.rows_for(hands=train_hands)
    holdout_rows = cache.rows_for(hands=holdout_hands)

    # α is calibrated on descriptor fidelity, not assignment mass: the point of
    # initialising from `codebook` is that the untrained module IS the frozen
    # baseline, so Δ measures learning and nothing else.
    probe = [cache.sample_tokens(i, 256, rng) for i in train_rows[:32]]
    if alpha is None:
        alpha = alpha_for_codebook(codebook, probe)
    fidelity = vlad_fidelity(probe, codebook, alpha)
    if fidelity < 0.99:
        print(f"[mole] WARNING: at α={alpha:g} the untrained aggregator matches hard "
              f"VLAD with cosine {fidelity:.4f} — it is NOT the baseline, so read Δ "
              f"against the @init evaluation, not against hard VLAD.")
    model = NetVLAD.from_codebook(codebook, alpha, learn=learn).to(dev)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    view = _PageView(cache, train_rows)
    cfg = {**(sampler_cfg or {})}
    cfg["windows_per_doc"] = 1                     # one page per (hand, document)
    sampler = HandBatchSampler(view, seed=seed, **cfg)

    best_macro, best_epoch, best_state, history = -1.0, -1, None, []
    for ep in track(range(epochs), "Training NetVLAD", unit="epoch", disable=not progress):
        model.train()
        losses, ents, grads = [], [], []
        for local_rows, hands, docs in sampler:
            pages = [view.rows[r] for r in local_rows]
            x = torch.from_numpy(np.stack(
                [cache.sample_tokens(i, tokens_per_page, rng) for i in pages])).to(dev)
            z = model(x)
            pos, neg = pair_masks(hands, docs)
            loss = masked_supcon(z, torch.from_numpy(pos).to(dev),
                                 torch.from_numpy(neg).to(dev), temperature)
            opt.zero_grad()
            loss.backward()
            # Which half of the module is actually moving? At the α that keeps the
            # aggregator faithful the softmax is nearly one-hot, so the assignment
            # gradient can vanish while the centroids (which enter the residual
            # linearly) keep training. Recording both makes a null result
            # attributable instead of ambiguous.
            grads.append((_grad_norm(model.assign_w), _grad_norm(model.centroids)))
            opt.step()
            losses.append(float(loss.item()))
            ents.append(model.assignment_entropy(x))
        sched.step()
        model.eval()
        macro = _holdout_macro_map(model, cache, holdout_rows, dev,
                                   max_tokens=select_max_tokens)
        g = np.mean(grads, axis=0) if grads else (0.0, 0.0)
        history.append({"epoch": ep, "loss": float(np.mean(losses)) if losses else 0.0,
                        "assign_entropy": float(np.mean(ents)) if ents else 0.0,
                        "grad_assign": float(g[0]), "grad_centroids": float(g[1]),
                        "holdout_macro": macro})
        if macro > best_macro:
            best_macro, best_epoch = macro, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Fit vs transfer, the distinction F4 was learned the hard way: measure the
    # SEEN hands too. A large train-side gain with a flat held-out side is
    # overfitting; both flat means the aggregator simply is not the bottleneck.
    # Capped, because this is a diagnostic and not the headline.
    fit_rows = train_rows[:: max(1, len(train_rows) // 400)] if train_rows else []
    init_model = NetVLAD.from_codebook(codebook, alpha, learn=learn).to(dev)
    fit_before = _holdout_macro_map(init_model, cache, fit_rows, dev,
                                    max_tokens=select_max_tokens)
    fit_after = _holdout_macro_map(model, cache, fit_rows, dev,
                                   max_tokens=select_max_tokens)

    report = {
        "best_holdout_macro": best_macro, "best_epoch": best_epoch,
        "epochs": epochs, "steps_per_epoch": len(sampler),
        "num_clusters": int(model.num_clusters), "dim": int(model.dim),
        "alpha": float(alpha), "init_fidelity": float(fidelity),
        "learn": learn, "temperature": temperature,
        "tokens_per_page": int(tokens_per_page), "seed": seed, "lr": lr,
        "base_model_id": cache.meta.get("model_id"),
        "n_train_hands": len(train_hands), "n_train_pages": len(train_rows),
        # seen hands, seen archives: the fit side of the fit-vs-transfer read
        "train_fit_before": fit_before, "train_fit_after": fit_after,
        "train_fit_delta": fit_after - fit_before, "n_fit_pages": len(fit_rows),
        "n_holdout_hands": len(holdout_hands), "n_holdout_pages": len(holdout_rows),
        "n_excluded_hands": len(exclude_hands),
        "train_hands": sorted(train_hands), "holdout_hands": sorted(holdout_hands),
        "excluded_hands": sorted(exclude_hands),
        "history": history,
    }
    return model, report


def save_netvlad(path: str | Path, model, report: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_clusters": model.num_clusters,
                "dim": model.dim, "alpha": model.alpha, "learn": model.learn,
                "intra_norm": model.intra_norm, "powernorm": model.powernorm,
                "base_model_id": report.get("base_model_id")}, path)
    return path


def load_netvlad(path: str | Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = NetVLAD(blob["num_clusters"], blob["dim"], alpha=blob["alpha"],
                    intra_norm=blob.get("intra_norm", False),
                    powernorm=blob.get("powernorm", True),
                    learn=blob.get("learn", "both"))
    model.load_state_dict(blob["state_dict"])
    return model.eval(), blob
