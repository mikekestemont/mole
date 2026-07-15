"""Retrieval benchmark from partial labels — writer-identification style.

Each labeled document queries a **leave-one-out** gallery of all other labeled
documents; relevance = same ``hand_id``. Reports mean Average Precision (mAP)
and Top-k accuracy — the standard Historical-WI / writer-retrieval protocol — so
a mole run can be scored directly against the literature (e.g. Raven et al.).

Two robustness extras beyond the headline mAP:

* **macro-mAP** averages AP per hand first, so a corpus dominated by one hand
  (Antwerp is ~48% hand ``R``) cannot flatter the score.
* a **cross-dataset breakdown** (same hand matched across *different*
  digitizations vs. within one) is produced whenever the labels span more than
  one dataset — the confound detector for repository / scan-quality shortcuts.

Metrics are also written to a ``<embeddings>.eval.json`` sidecar. (Wiring scores
into the lineage registry follows once that registry lands — it is still a stub.)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from mole.data.datasets import discover_datasets
from mole.progress import track


@dataclass
class RetrievalScores:
    """One retrieval measurement over some query/gallery configuration."""

    n_queries: int
    mean_ap: float          # micro mAP: mean AP over all valid queries
    macro_map: float        # per-hand-averaged AP (robust to class skew)
    top1: float
    topk: dict[int, float]  # k -> soft Top-k accuracy (any relevant in top k)


@dataclass
class EvalResult:
    model_id: str | None
    metric: str
    n_embeddings: int
    n_labeled: int
    n_hands: int
    coverage: float
    datasets: list[str]
    overall: RetrievalScores
    within_dataset: RetrievalScores | None = None
    cross_dataset: RetrievalScores | None = None
    per_dataset: dict[str, RetrievalScores] = field(default_factory=dict)


# --------------------------------------------------------------------- loading
def _load_embeddings(path: str | Path):
    """Return (matrix, per-row image paths, model_id) from an embed output."""
    path = Path(path)
    npy = path if path.suffix == ".npy" else path.with_suffix(".npy")
    X = np.load(npy).astype(np.float64)
    meta = json.loads(path.with_suffix(".mapping.json").read_text())
    rows = meta.get("rows") or []
    images = [str(r["image"]) for r in rows]
    return X, images, meta.get("model_id")


def _label_tables(datasets_root: str | Path) -> dict[str, dict[str, str]]:
    """dataset_name -> {basename: hand_id} for every labeled dataset under root."""
    tables: dict[str, dict[str, str]] = {}
    for m in discover_datasets(datasets_root):
        if m.labels and m.labels.hand_by_filename:
            tables[m.name] = m.labels.hand_by_filename
    return tables


# --------------------------------------------------------------------- metrics
def _similarity(X: np.ndarray, metric: str) -> np.ndarray:
    """Pairwise similarity where higher = more similar (rank descending)."""
    if metric == "cosine":
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return Xn @ Xn.T
    if metric == "euclidean":
        sq = (X * X).sum(1)
        return -(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T))  # negative sq-dist
    raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric!r}")


def _rank_metrics(sim: np.ndarray, labels: np.ndarray, allow: np.ndarray,
                  ks: tuple[int, ...], *, progress: bool = False,
                  desc: str = "Scoring") -> RetrievalScores | None:
    """Leave-one-out retrieval metrics over the gallery defined by ``allow``.

    ``allow[i, j]`` marks j as an eligible gallery item for query i (self is
    excluded by the caller). Queries with no eligible relevant item are skipped
    — standard for mAP when a writer has no other document in the gallery.
    """
    n = len(labels)
    aps: list[float] = []
    t1: list[float] = []
    hitk: dict[int, list[float]] = {k: [] for k in ks}
    per_hand: dict[str, list[float]] = {}

    for i in track(range(n), desc, unit="query", disable=not progress):
        gal = np.where(allow[i])[0]
        if gal.size == 0:
            continue
        order = np.argsort(-sim[i, gal], kind="stable")
        rel = labels[gal][order] == labels[i]
        R = int(rel.sum())
        if R == 0:
            continue
        cum = np.cumsum(rel)
        prec = cum / np.arange(1, rel.size + 1)
        ap = float((prec * rel).sum() / R)
        aps.append(ap)
        t1.append(float(rel[0]))
        for k in ks:
            hitk[k].append(float(rel[:k].any()))
        per_hand.setdefault(str(labels[i]), []).append(ap)

    if not aps:
        return None
    macro = float(np.mean([float(np.mean(v)) for v in per_hand.values()]))
    return RetrievalScores(
        n_queries=len(aps),
        mean_ap=float(np.mean(aps)),
        macro_map=macro,
        top1=float(np.mean(t1)),
        topk={k: float(np.mean(hitk[k])) for k in ks},
    )


# ------------------------------------------------------------------ public API
def evaluate(embeddings_path: str | Path, datasets_root: str | Path,
             *, metric: str = "cosine", topk: tuple[int, ...] = (1, 5, 10),
             out: str | Path | None = None) -> EvalResult:
    """Run the retrieval benchmark and write a JSON report sidecar.

    ``datasets_root`` may be a single dataset folder (its ``labels.csv``) or a
    root of several; labels are matched to embeddings by image basename, exactly
    as ``mole embed``/``viz`` match them.
    """
    X, images, model_id = _load_embeddings(embeddings_path)
    if len(images) != len(X):
        raise ValueError(
            f"{len(images)} mapping rows vs {len(X)} vectors — this looks like a "
            "patch-level embedding; eval needs page-level (mean/cls/vlad) output")

    tables = _label_tables(datasets_root)
    # When only one dataset carries labels, attribute every embedding to it even
    # if the mapping's folder name differs (embedded elsewhere, evaluated here).
    # With several datasets we key on the image's parent-folder name so shared
    # basenames across digitizations don't collide.
    solo = next(iter(tables)) if len(tables) == 1 else None
    hands: list[str] = []
    dsets: list[str] = []
    keep: list[int] = []
    for i, path in enumerate(images):
        p = Path(path)
        ds = p.parent.name
        if ds in tables and p.name in tables[ds]:
            hand = tables[ds][p.name]
        elif solo and p.name in tables[solo]:
            hand, ds = tables[solo][p.name], solo
        else:
            continue
        hands.append(hand)
        dsets.append(ds)
        keep.append(i)

    if len(keep) < 2:
        raise ValueError(
            f"only {len(keep)} of {len(images)} embeddings are labeled — need ≥2 "
            f"(is labels.csv under {datasets_root} and do basenames match?)")

    Xk = X[keep]
    labels = np.asarray(hands, dtype=object)
    datasets = np.asarray(dsets, dtype=object)
    dataset_names = sorted(set(dsets))
    n = len(keep)

    sim = _similarity(Xk, metric)
    off_diag = ~np.eye(n, dtype=bool)

    overall = _rank_metrics(sim, labels, off_diag, topk, progress=True,
                            desc="Retrieval mAP")
    if overall is None:
        raise ValueError("no hand has ≥2 labeled documents — nothing to retrieve")

    result = EvalResult(
        model_id=model_id, metric=metric, n_embeddings=len(X), n_labeled=n,
        n_hands=len(set(hands)), coverage=n / len(X), datasets=dataset_names,
        overall=overall,
    )

    if len(dataset_names) > 1:
        same = datasets[:, None] == datasets[None, :]
        result.within_dataset = _rank_metrics(sim, labels, off_diag & same, topk)
        result.cross_dataset = _rank_metrics(sim, labels, off_diag & ~same, topk)
        for d in dataset_names:
            idx = np.where(datasets == d)[0]
            if idx.size < 2:
                continue
            sub = _rank_metrics(sim[np.ix_(idx, idx)], labels[idx],
                                ~np.eye(idx.size, dtype=bool), topk)
            if sub is not None:
                result.per_dataset[d] = sub

    out_path = Path(out) if out else Path(embeddings_path).with_suffix(".eval.json")
    out_path.write_text(json.dumps(_to_jsonable(result), indent=2))
    print(f"[mole] ✓ eval report → {out_path}")
    return result


def _to_jsonable(result: EvalResult) -> dict:
    d = asdict(result)
    # asdict keeps topk dict keys as ints; JSON needs str keys — normalise.
    def fix(scores: dict | None):
        if scores and isinstance(scores.get("topk"), dict):
            scores["topk"] = {str(k): v for k, v in scores["topk"].items()}
    fix(d.get("overall"))
    fix(d.get("within_dataset"))
    fix(d.get("cross_dataset"))
    for s in (d.get("per_dataset") or {}).values():
        fix(s)
    return d


# ------------------------------------------------------------------ reporting
def _fmt_scores(s: RetrievalScores, indent: str = "    ") -> str:
    lines = [
        f"{indent}mAP           {s.mean_ap:.4f}",
        f"{indent}macro-mAP     {s.macro_map:.4f}   (per-hand averaged)",
        f"{indent}Top-1         {s.top1:.4f}",
    ]
    lines += [f"{indent}Top-{k:<9d}{v:.4f}" for k, v in s.topk.items() if k != 1]
    return "\n".join(lines)


def format_report(r: EvalResult) -> str:
    ds = ", ".join(r.datasets)
    out = [
        f"Retrieval eval — {r.model_id or '?'}",
        f"  metric: {r.metric} | {r.n_embeddings} embeddings, "
        f"{r.n_labeled} labeled ({r.coverage:.1%}), {r.n_hands} hands, "
        f"{len(r.datasets)} dataset(s): {ds}",
        "",
        f"  Overall (leave-one-out, {r.overall.n_queries} queries)",
        _fmt_scores(r.overall, "    "),
    ]
    if r.within_dataset and r.cross_dataset:
        out += ["", "  Within-dataset (same digitization)",
                _fmt_scores(r.within_dataset, "    "),
                "", "  Cross-dataset (same hand across digitizations — the real signal)",
                _fmt_scores(r.cross_dataset, "    ")]
    if r.per_dataset:
        out += ["", "  Per dataset:"]
        for name, s in r.per_dataset.items():
            out.append(f"    {name}: mAP {s.mean_ap:.4f}  Top-1 {s.top1:.4f}  "
                       f"({s.n_queries} queries)")
    return "\n".join(out)
