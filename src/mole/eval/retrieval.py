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

from mole.data.datasets import LabelTable, discover_datasets
from mole.progress import track


@dataclass
class RetrievalScores:
    """One retrieval measurement over some query/gallery configuration."""

    n_queries: int
    mean_ap: float          # micro mAP: mean AP over all valid queries
    macro_map: float        # per-hand-averaged AP (robust to class skew)
    top1: float
    topk: dict[int, float]  # k -> soft Top-k accuracy (any relevant in top k)
    # hand -> {"ap": mean AP over that hand's queries, "n_queries": how many}.
    # macro_map == mean of the per-hand "ap" values; serialised so downstream
    # tools (eval-compare's paired per-hand bootstrap, the "which hands are
    # hopeless" view) can consume it without re-running retrieval.
    per_hand: dict[str, dict] = field(default_factory=dict)


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
    min_confidence: float | None = None
    cross_doc_only: bool = False
    n_holdout_hands: int | None = None  # queries restricted to N held-out hands
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


def _label_tables(datasets_root: str | Path) -> dict[str, LabelTable]:
    """dataset_name -> LabelTable for every labeled dataset under root."""
    tables: dict[str, LabelTable] = {}
    for m in discover_datasets(datasets_root):
        if m.labels and m.labels.hand_by_filename:
            tables[m.name] = m.labels
    return tables


def load_hand_set(path: str | Path, key: str = "holdout_hands") -> set[str]:
    """Load a set of hand ids from a split file.

    Accepts either a bare JSON list of hands, or an object with a ``key`` list
    (default ``holdout_hands``) — the format written by the Phase-1 hand
    splitter. Hands may be namespaced (``archive/hand``) or raw; :func:`evaluate`
    matches a query against both forms.
    """
    data = json.loads(Path(path).read_text())
    hands = data if isinstance(data, list) else data.get(key, [])
    return {str(h) for h in hands}


def _doc_resolvers(datasets_root: str | Path):
    """dataset_name -> (basename -> doc_id) resolver, for cross-doc grouping."""
    from mole.data.docids import doc_id_resolver
    return {
        m.name: doc_id_resolver(m.root)
        for m in discover_datasets(datasets_root)
        if m.labels and m.labels.hand_by_filename
    }


def _hand_if_confident(table: LabelTable, fname: str,
                       min_confidence: float | None) -> str | None:
    """hand_id for ``fname`` in ``table``, or None if unlabeled / below floor.

    A row below ``min_confidence`` is demoted to *unlabeled* (it becomes neither
    a query nor a gallery item) rather than kept as a low-trust label. Rows with
    no confidence value are treated as confident (the floor only bites where a
    ``confidence`` column exists — e.g. Leroy's auto-matched labels)."""
    hand = table.hand_by_filename.get(fname)
    if hand is None:
        return None
    if min_confidence is not None:
        conf = table.confidence.get(fname)
        if conf is not None and conf < min_confidence:
            return None
    return hand


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
                  ks: tuple[int, ...], *, query_mask: np.ndarray | None = None,
                  progress: bool = False,
                  desc: str = "Scoring") -> RetrievalScores | None:
    """Leave-one-out retrieval metrics over the gallery defined by ``allow``.

    ``allow[i, j]`` marks j as an eligible gallery item for query i (self is
    excluded by the caller). Queries with no eligible relevant item are skipped
    — standard for mAP when a writer has no other document in the gallery.

    ``query_mask`` (optional bool array) restricts which rows may act as
    *queries* while leaving the gallery untouched — used for held-out-hand eval
    (queries = held-out-hand docs, gallery = the full archive).
    """
    n = len(labels)
    aps: list[float] = []
    t1: list[float] = []
    hitk: dict[int, list[float]] = {k: [] for k in ks}
    per_hand: dict[str, list[float]] = {}

    for i in track(range(n), desc, unit="query", disable=not progress):
        if query_mask is not None and not query_mask[i]:
            continue
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
    per_hand_summary = {
        h: {"ap": float(np.mean(v)), "n_queries": len(v)}
        for h, v in per_hand.items()
    }
    macro = float(np.mean([s["ap"] for s in per_hand_summary.values()]))
    return RetrievalScores(
        n_queries=len(aps),
        mean_ap=float(np.mean(aps)),
        macro_map=macro,
        top1=float(np.mean(t1)),
        topk={k: float(np.mean(hitk[k])) for k in ks},
        per_hand=per_hand_summary,
    )


# ------------------------------------------------------------------ public API
def evaluate(embeddings_path: str | Path, datasets_root: str | Path,
             *, metric: str = "cosine", topk: tuple[int, ...] = (1, 5, 10),
             min_confidence: float | None = None, cross_doc_only: bool = False,
             holdout_hands: set[str] | None = None,
             out: str | Path | None = None) -> EvalResult:
    """Run the retrieval benchmark and write a JSON report sidecar.

    ``datasets_root`` may be a single dataset folder (its ``labels.csv``) or a
    root of several; labels are matched to embeddings by image basename, exactly
    as ``mole embed``/``viz`` match them.

    ``min_confidence`` demotes any label whose ``confidence`` column value is
    below the floor to *unlabeled* (drops it from both queries and gallery) —
    the honest way to read auto-matched label sets like Leroy. Labels without a
    confidence value are unaffected.

    ``cross_doc_only`` redefines relevance as *same hand AND different document*:
    sibling scans of one physical charter (grouped by :mod:`mole.data.docids`)
    are removed from both the relevant set and the ranking, so a model earns no
    credit for re-finding a sibling scan. This is the honest metric for any
    label-trained claim. For archives that are one-image-per-charter it is a
    no-op (every doc id is unique).

    ``holdout_hands`` (a set of hand ids, possibly namespaced ``archive/hand``)
    restricts which docs may act as *queries* — the gallery stays the full
    archive. This is the §4.2 held-out-hand protocol: it measures whether the
    geometry improved for *unseen* hands, not memorization.
    """
    X, images, model_id = _load_embeddings(embeddings_path)
    if len(images) != len(X):
        raise ValueError(
            f"{len(images)} mapping rows vs {len(X)} vectors — this looks like a "
            "patch-level embedding; eval needs page-level (mean/cls/vlad) output")

    tables = _label_tables(datasets_root)
    resolvers = _doc_resolvers(datasets_root) if cross_doc_only else {}
    # When only one dataset carries labels, attribute every embedding to it even
    # if the mapping's folder name differs (embedded elsewhere, evaluated here).
    # With several datasets we key on the image's parent-folder name so shared
    # basenames across digitizations don't collide.
    solo = next(iter(tables)) if len(tables) == 1 else None
    hands: list[str] = []
    dsets: list[str] = []
    docs: list[str] = []
    keep: list[int] = []
    for i, path in enumerate(images):
        p = Path(path)
        ds = p.parent.name
        if ds in tables:
            hand = _hand_if_confident(tables[ds], p.name, min_confidence)
        elif solo:
            hand, ds = _hand_if_confident(tables[solo], p.name, min_confidence), solo
        else:
            hand = None
        if hand is None:
            continue
        hands.append(hand)
        dsets.append(ds)
        if cross_doc_only:
            resolve = resolvers.get(ds)
            # doc ids are namespaced by dataset so they never collide across
            # archives; fall back to the basename if the dataset has no resolver.
            docs.append(f"{ds}/{resolve(p.name)}" if resolve else f"{ds}/{p.name}")
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
    # cross-doc-only: also forbid same-document (sibling-scan) gallery items.
    if cross_doc_only:
        doc_arr = np.asarray(docs, dtype=object)
        base_allow = off_diag & (doc_arr[:, None] != doc_arr[None, :])
    else:
        base_allow = off_diag

    # held-out-hand protocol: restrict queries (not the gallery) to those hands.
    n_holdout = None
    query_mask = None
    if holdout_hands is not None:
        query_mask = np.asarray(
            [(h in holdout_hands) or (f"{d}/{h}" in holdout_hands)
             for h, d in zip(hands, dsets)], dtype=bool)
        n_holdout = len({h for h, m in zip(hands, query_mask) if m})
        if not query_mask.any():
            raise ValueError(
                "no labeled doc belongs to a held-out hand — check the split file "
                "(namespaced 'archive/hand' or raw 'hand') against these datasets")

    overall = _rank_metrics(sim, labels, base_allow, topk, query_mask=query_mask,
                            progress=True, desc="Retrieval mAP")
    if overall is None:
        raise ValueError(
            "no hand has ≥2 labeled documents"
            + (" in *different* charters (cross-doc-only)" if cross_doc_only else "")
            + (" among the held-out hands" if holdout_hands is not None else "")
            + " — nothing to retrieve")

    result = EvalResult(
        model_id=model_id, metric=metric, n_embeddings=len(X), n_labeled=n,
        n_hands=len(set(hands)), coverage=n / len(X), datasets=dataset_names,
        overall=overall, min_confidence=min_confidence, cross_doc_only=cross_doc_only,
        n_holdout_hands=n_holdout,
    )

    if len(dataset_names) > 1:
        same = datasets[:, None] == datasets[None, :]
        result.within_dataset = _rank_metrics(sim, labels, base_allow & same, topk,
                                               query_mask=query_mask)
        result.cross_dataset = _rank_metrics(sim, labels, base_allow & ~same, topk,
                                              query_mask=query_mask)
        for d in dataset_names:
            idx = np.where(datasets == d)[0]
            if idx.size < 2:
                continue
            sub = _rank_metrics(sim[np.ix_(idx, idx)], labels[idx],
                                base_allow[np.ix_(idx, idx)], topk,
                                query_mask=query_mask[idx] if query_mask is not None
                                else None)
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


def format_per_hand(scores: RetrievalScores, *, title: str = "Per-hand AP",
                    indent: str = "    ") -> str:
    """Worst-first per-hand AP table — the 'which hands are hopeless' view."""
    if not scores.per_hand:
        return f"{indent}(no per-hand breakdown)"
    rows = sorted(scores.per_hand.items(), key=lambda kv: kv[1]["ap"])
    width = max((len(h) for h in scores.per_hand), default=4)
    lines = [f"{indent}{title} (worst first):",
             f"{indent}  {'hand'.ljust(width)}   AP      n"]
    for hand, s in rows:
        lines.append(f"{indent}  {hand.ljust(width)}   {s['ap']:.4f}  {s['n_queries']:>3d}")
    return "\n".join(lines)


def format_report(r: EvalResult, *, per_hand: bool = False) -> str:
    ds = ", ".join(r.datasets)
    out = [
        f"Retrieval eval — {r.model_id or '?'}",
        f"  metric: {r.metric} | {r.n_embeddings} embeddings, "
        f"{r.n_labeled} labeled ({r.coverage:.1%}), {r.n_hands} hands, "
        f"{len(r.datasets)} dataset(s): {ds}",
        *([f"  min-confidence: ≥{r.min_confidence:g} (lower-confidence labels dropped)"]
          if r.min_confidence is not None else []),
        *(["  relevance: cross-document only (sibling scans excluded)"]
          if r.cross_doc_only else []),
        *([f"  queries: held-out hands only ({r.n_holdout_hands} unseen hands)"]
          if r.n_holdout_hands is not None else []),
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
    if per_hand:
        out += ["", "  " + format_per_hand(r.overall, indent="  ")]
    return "\n".join(out)
