"""Label-review suggestions from a partially labeled embedding.

Archives arrive with *some* hands identified and most documents untouched. This
module turns one ``mole embed`` output plus whatever ``labels.csv`` exists into
ranked, human-checkable hypotheses of six kinds:

* **attributions** — an unlabeled document that sits with a known hand;
* **merges**       — two labeled hands that may be one scribe;
* **splits**       — one labeled hand whose documents form two separate groups;
* **new hands**    — a tight group of unlabeled documents unlike any known hand;
* **doubts**       — a labeled document that sits with a *different* hand;
* **duplicates**   — two documents that are near-identical (same charter twice).

Nothing here writes labels: the output is a report for a human to read
(``SUPERVISED_PLAN.md`` D5). Two rules keep the numbers honest:

**Sibling scans are not evidence.** Every score excludes documents sharing the
query's ``doc_id`` (:mod:`mole.data.docids`), so "this page looks like hand B"
can never rest on another scan of the very same charter — the scan-shortcut that
``mole eval --cross-doc-only`` exists to kill.

**Only attributions are calibrated.** Hiding each labeled document in turn and
scoring it as if unlabeled gives real ground truth, so attribution scores are
mapped through isotonic regression to an empirical P(top-1 correct). The other
five kinds have no ground truth available and carry a *relative* strength
instead (a percentile against a null), which the UI must word as a question
rather than a claim.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# a hand needs this many documents before "is it really two hands?" is askable
MIN_DOCS_FOR_SPLIT = 4
# cosine above which two different documents are treated as the same image
DUPLICATE_SIM = 0.98


@dataclass
class ReviewReport:
    """Ranked suggestions of every kind, plus what they were computed from."""

    n_documents: int = 0
    n_labeled: int = 0
    n_hands: int = 0
    datasets: list[str] = field(default_factory=list)
    model_id: str | None = None
    attributions: list[dict] = field(default_factory=list)
    merges: list[dict] = field(default_factory=list)
    splits: list[dict] = field(default_factory=list)
    new_hands: list[dict] = field(default_factory=list)
    doubts: list[dict] = field(default_factory=list)
    duplicates: list[dict] = field(default_factory=list)
    isolated: list[dict] = field(default_factory=list)
    calibration: dict = field(default_factory=dict)
    # FINCH's whole hierarchy, kept so the renderer can offer one colour scheme
    # per level without paying for the clustering a second time. Each entry is
    # {level, n_clusters, labels, silhouette}; `silhouette` is None where it is
    # undefined (fewer than 2 clusters, or one cluster per document).
    cluster_levels: list[dict] = field(default_factory=list)

    @property
    def cluster_labels(self) -> list[int]:
        """The finest partition — what the 'possible new hand' list is built on."""
        return self.cluster_levels[0]["labels"] if self.cluster_levels else []

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(asdict(self), indent=2))
        return p


# ------------------------------------------------------------------ primitives
def _l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _top2_mean(vals: np.ndarray) -> float:
    """Mean of the two largest values — 'two documents agree', not one lucky match.

    A single high similarity is the classic false friend (one shared formula, one
    similar layout). Requiring the top TWO to be high asks for corroboration from
    a second, independent document. With only one candidate its own value stands.
    """
    if vals.size == 0:
        return float("-inf")
    if vals.size == 1:
        return float(vals[0])
    top = np.partition(vals, -2)[-2:]
    return float(top.mean())


def hand_score_matrix(sim: np.ndarray, members: dict[str, np.ndarray],
                      doc_ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """``[N, H]`` score of every document against every hand.

    ``score[i, h]`` = mean of the top-2 similarities from document ``i`` to hand
    ``h``'s documents, **excluding any document sharing i's doc_id** (so sibling
    scans of the same charter cannot vouch for each other). Documents of a hand
    that are all siblings of the query simply yield ``-inf``: no evidence.
    """
    hands = sorted(members)
    out = np.full((sim.shape[0], len(hands)), -np.inf, dtype=np.float32)
    for j, h in enumerate(hands):
        cols = members[h]
        sub = sim[:, cols]                                  # [N, |h|]
        same_doc = doc_ids[:, None] == doc_ids[None, cols]   # incl. the diagonal
        sub = np.where(same_doc, -np.inf, sub)
        for i in range(sub.shape[0]):
            row = sub[i]
            row = row[np.isfinite(row)]
            out[i, j] = _top2_mean(row)
    return out, hands


def _cohesion(sim: np.ndarray, idx: np.ndarray, doc_ids: np.ndarray) -> float:
    """Mean cross-document similarity within a group (its internal tightness)."""
    if len(idx) < 2:
        return float("nan")
    sub = sim[np.ix_(idx, idx)]
    mask = doc_ids[idx][:, None] != doc_ids[idx][None, :]
    return float(sub[mask].mean()) if mask.any() else float("nan")


# -------------------------------------------------------------- the six lists
def _attributions(scores, hands, unlabeled, names, hand_names, limit):
    """Unlabeled document -> the known hand it sits with (ranked by score)."""
    out = []
    for i in unlabeled:
        row = scores[i]
        if not np.isfinite(row).any():
            continue
        order = np.argsort(-row)
        best = int(order[0])
        second = float(row[order[1]]) if len(order) > 1 and np.isfinite(row[order[1]]) else float("nan")
        out.append({
            "row": int(i), "document": names[i], "hand": hands[best],
            "score": float(row[best]),
            "margin": float(row[best] - second) if np.isfinite(second) else None,
            "runner_up": hands[int(order[1])] if len(order) > 1 else None,
            "n_support": int(len(hand_names[hands[best]])),
        })
    out.sort(key=lambda d: -d["score"])
    return out[:limit]


def _doubts(scores, hands, labeled, hand_of, names, limit):
    """Labeled document that resembles some OTHER hand more than its own."""
    idx = {h: j for j, h in enumerate(hands)}
    out = []
    for i in labeled:
        own = hand_of[i]
        if own not in idx:
            continue
        own_score = float(scores[i, idx[own]])
        row = scores[i].copy()
        row[idx[own]] = -np.inf
        if not np.isfinite(row).any():
            continue
        best = int(np.argmax(row))
        gap = float(row[best]) - own_score
        if not np.isfinite(own_score):
            # its hand has no other document to compare against: not a doubt,
            # just an unverifiable label. Leave it out rather than cry wolf.
            continue
        if gap > 0:
            out.append({"row": int(i), "document": names[i], "hand": own,
                        "own_score": own_score, "closer_hand": hands[best],
                        "closer_score": float(row[best]), "gap": gap})
    out.sort(key=lambda d: -d["gap"])
    return out[:limit]


def _merges(sim, members, doc_ids, limit):
    """Hand pairs whose documents mingle as much as each hand mingles with itself.

    Scored as a DIFFERENCE, not a ratio: ``cross - mean(cohesion_a, cohesion_b)``.
    Cosine similarities are freely negative (mean-pooled and whitened spaces
    routinely are), and a ratio of two possibly-negative quantities flips sign and
    explodes — the difference degrades gracefully and reads plainly: at 0 the two
    hands are exactly as alike as each is to itself, which is what "these may be
    one scribe" means.
    """
    hands = sorted(members)
    coh = {h: _cohesion(sim, members[h], doc_ids) for h in hands}
    out = []
    for a_i, a in enumerate(hands):
        for b in hands[a_i + 1:]:
            ca, cb = coh[a], coh[b]
            if not (np.isfinite(ca) and np.isfinite(cb)):
                continue                      # a 1-doc hand has no cohesion to match
            cross = sim[np.ix_(members[a], members[b])]
            if cross.size == 0:
                continue
            own = 0.5 * (ca + cb)
            out.append({"hand_a": a, "hand_b": b,
                        "closeness": float(cross.mean()) - own,
                        "cross_similarity": float(cross.mean()),
                        "own_similarity": float(own),
                        "cohesion_a": ca, "cohesion_b": cb,
                        "n_a": int(len(members[a])), "n_b": int(len(members[b]))})
    out.sort(key=lambda d: -d["closeness"])
    return out[:limit]


def _split_strength(sub_sim: np.ndarray, seed: int = 0, n_perm: int = 200):
    """Best 2-way split of one hand, scored against random splits of the SAME docs.

    Returns ``(labels, separation, percentile)``. The permutation null is what
    makes this readable to a non-specialist: "this division is sharper than 95% of
    random divisions of the same documents" needs no notion of cosine distance.
    """
    from sklearn.cluster import AgglomerativeClustering

    n = len(sub_sim)
    dist = 1.0 - sub_sim
    np.fill_diagonal(dist, 0.0)
    labels = AgglomerativeClustering(
        n_clusters=2, metric="precomputed", linkage="average").fit_predict(dist)

    def separation(lab):
        a, b = lab == 0, lab == 1
        if a.sum() < 2 or b.sum() < 2:
            return float("nan")
        within = np.concatenate([sub_sim[np.ix_(a, a)][np.triu_indices(a.sum(), 1)],
                                 sub_sim[np.ix_(b, b)][np.triu_indices(b.sum(), 1)]])
        return float(within.mean() - sub_sim[np.ix_(a, b)].mean())

    obs = separation(labels)
    if not np.isfinite(obs):
        return labels, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = labels.copy()
        rng.shuffle(perm)
        s = separation(perm)
        if np.isfinite(s):
            null.append(s)
    pct = float((np.asarray(null) < obs).mean() * 100) if null else float("nan")
    return labels, obs, pct


def _splits(sim, members, doc_ids, names, seed, limit):
    """Labeled hands that look like two hands wearing one name."""
    out = []
    for h, idx in sorted(members.items()):
        if len(idx) < MIN_DOCS_FOR_SPLIT:
            continue
        labels, sep, pct = _split_strength(sim[np.ix_(idx, idx)], seed=seed)
        if not np.isfinite(sep) or not np.isfinite(pct):
            continue
        out.append({
            "hand": h, "separation": sep, "percentile": pct, "n_docs": int(len(idx)),
            "group_a": [names[i] for i, v in zip(idx, labels) if v == 0],
            "group_b": [names[i] for i, v in zip(idx, labels) if v == 1],
            "rows_a": [int(i) for i, v in zip(idx, labels) if v == 0],
            "rows_b": [int(i) for i, v in zip(idx, labels) if v == 1],
        })
    out.sort(key=lambda d: (-d["percentile"], -d["separation"]))
    return out[:limit]


def _new_hands(sim, cluster_labels, is_labeled, scores, doc_ids, names,
               hand_cohesions, limit):
    """Tight clusters of mostly-unlabeled documents that match no known hand."""
    ref = float(np.nanmedian(hand_cohesions)) if len(hand_cohesions) else 0.0
    out = []
    for c in sorted(set(int(v) for v in cluster_labels)):
        idx = np.where(cluster_labels == c)[0]
        if len(idx) < 3:
            continue
        frac_unlabeled = float((~is_labeled[idx]).mean())
        if frac_unlabeled < 0.8:
            continue                                  # mostly known: not new
        coh = _cohesion(sim, idx, doc_ids)
        if not np.isfinite(coh) or coh < ref:
            continue                                  # looser than a typical hand
        best_known = float(np.nanmax(scores[idx])) if scores.size else float("-inf")
        out.append({"cluster": c, "n_docs": int(len(idx)), "cohesion": coh,
                    "reference_cohesion": ref, "closest_known_score": best_known,
                    "documents": [names[i] for i in idx],
                    "rows": [int(i) for i in idx]})
    out.sort(key=lambda d: (-d["n_docs"], -d["cohesion"]))
    return out[:limit]


def _duplicates(sim, doc_ids, names, limit):
    """Near-identical documents that are NOT already grouped as one charter."""
    n = len(sim)
    iu = np.triu_indices(n, 1)
    hot = np.where(sim[iu] >= DUPLICATE_SIM)[0]
    out = []
    for k in hot:
        i, j = int(iu[0][k]), int(iu[1][k])
        if doc_ids[i] == doc_ids[j]:
            continue                                   # known siblings: fine
        out.append({"row_a": i, "row_b": j, "document_a": names[i],
                    "document_b": names[j], "similarity": float(sim[i, j])})
    out.sort(key=lambda d: -d["similarity"])
    return out[:limit]


def _isolated(sim, doc_ids, names, limit, quantile=0.02):
    """Documents whose best match is far below everyone else's — blanks, covers."""
    n = len(sim)
    best = np.full(n, -np.inf, dtype=np.float32)
    for i in range(n):
        row = np.where(doc_ids == doc_ids[i], -np.inf, sim[i])
        if np.isfinite(row).any():
            best[i] = row.max()
    finite = best[np.isfinite(best)]
    if finite.size == 0:
        return []
    cut = float(np.quantile(finite, quantile))
    order = np.argsort(best)
    return [{"row": int(i), "document": names[i], "best_match": float(best[i])}
            for i in order[:limit] if np.isfinite(best[i]) and best[i] <= cut]


# ------------------------------------------------------------------ calibration
def _calibrate(scores, hands, labeled, hand_of):
    """Score -> empirical P(top-1 hand correct), by hiding each labeled doc in turn.

    The hand-score matrix already excludes the query itself and its siblings, so
    reading a labeled row off it *is* the leave-one-out prediction — no refit
    needed. Isotonic regression keeps the mapping monotone without assuming a
    shape, and we hand back the raw points too so the UI can show honest counts
    ("of 40 suggestions this strong, 36 were right") rather than a bare number.
    """
    idx = {h: j for j, h in enumerate(hands)}
    xs, ys = [], []
    for i in labeled:
        row = scores[i]
        if not np.isfinite(row).any() or hand_of[i] not in idx:
            continue
        best = int(np.argmax(row))
        xs.append(float(row[best]))
        ys.append(1.0 if hands[best] == hand_of[i] else 0.0)
    if len(xs) < 8:
        return {"n": len(xs), "fitted": False, "scores": xs, "correct": ys}

    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(xs, ys)
    grid = np.linspace(float(min(xs)), float(max(xs)), 25)
    return {"n": len(xs), "fitted": True,
            "accuracy": float(np.mean(ys)),
            "grid": [float(g) for g in grid],
            "precision": [float(p) for p in iso.predict(grid)],
            "scores": xs, "correct": ys}


def _apply_calibration(cal, score: float) -> float | None:
    if not cal.get("fitted"):
        return None
    return float(np.interp(score, cal["grid"], cal["precision"]))


def _silhouette(Xn: np.ndarray, labels: np.ndarray) -> float | None:
    """Mean silhouette of one partition (cosine), or None where it is undefined.

    A partition needs at least 2 clusters and at least one cluster with more than
    one member; FINCH's coarsest levels often fail both. Reported per level so the
    reader has a principled way to pick one instead of guessing.
    """
    n_lab = len(set(labels.tolist()))
    if n_lab < 2 or n_lab >= len(labels):
        return None
    try:
        from sklearn.metrics import silhouette_score

        return float(silhouette_score(Xn, labels, metric="cosine"))
    except Exception:
        return None


# ------------------------------------------------------------------- the driver
def _load(embeddings: str | Path):
    path = Path(embeddings)
    npy = path if path.suffix == ".npy" else path.with_suffix(".npy")
    X = np.load(npy)
    sidecar = npy.with_suffix(".mapping.json")
    meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    rows = meta.get("rows") or [{"row": i, "image": str(i)} for i in range(len(X))]
    if len(rows) != len(X):
        rows = [{"row": i, "image": str(i)} for i in range(len(X))]
    return X, meta, rows


def document_table(embeddings: str | Path):
    """``(X, meta, rows, names, paths, hands, docs)`` for one embedding file.

    Hands and doc ids are namespaced by dataset folder. Shared with the renderer
    so the picture and the lists can never disagree about what a row is.
    """
    from mole.data.datasets import load_labels
    from mole.data.docids import doc_id_resolver

    X, meta, rows = _load(embeddings)
    paths = [Path(r["image"]) for r in rows]
    names = [p.name for p in paths]
    cache: dict[Path, tuple] = {}
    hands, docs = [], []
    for p in paths:
        if p.parent not in cache:
            cache[p.parent] = (load_labels(p.parent), doc_id_resolver(p.parent))
        table, resolve = cache[p.parent]
        raw = table.hand_by_filename.get(p.name)
        hands.append(f"{p.parent.name}/{raw}" if raw else "")
        docs.append(f"{p.parent.name}/{resolve(p.name)}")
    return X, meta, rows, names, paths, hands, docs


def build_review(embeddings: str | Path, *, clusters: str | Path | None = None,
                 limit: int = 100, seed: int = 0) -> ReviewReport:
    """Build every suggestion list for one embedding file.

    ``clusters`` is an optional ``mole cluster`` report; without one, FINCH's
    finest partition is computed here so the "possible new hand" list always
    exists. ``limit`` caps each list — these are for human review, and a list
    nobody can finish reading is a list nobody reads.
    """
    X, meta, rows, names, paths, hand_of, doc_ids = document_table(embeddings)
    parents = [p.parent for p in paths]

    hand_of_arr = np.asarray(hand_of, dtype=object)
    doc_arr = np.asarray(doc_ids, dtype=object)
    is_labeled = np.asarray([bool(h) for h in hand_of], dtype=bool)
    labeled = np.where(is_labeled)[0]
    unlabeled = np.where(~is_labeled)[0]

    members: dict[str, np.ndarray] = {}
    for h in sorted({h for h in hand_of if h}):
        members[h] = np.where(hand_of_arr == h)[0]

    Xn = _l2(X)
    sim = (Xn @ Xn.T).astype(np.float32)

    report = ReviewReport(
        n_documents=len(rows), n_labeled=int(is_labeled.sum()), n_hands=len(members),
        datasets=sorted({p.name for p in parents}), model_id=meta.get("model_id"))
    if not members:
        return report                       # nothing labeled: no reference to reason from

    scores, hands = hand_score_matrix(sim, members, doc_arr)
    cal = _calibrate(scores, hands, labeled, hand_of_arr)
    report.calibration = cal

    report.attributions = _attributions(scores, hands, unlabeled, names, members, limit)
    for a in report.attributions:
        a["calibrated_p"] = _apply_calibration(cal, a["score"])
    report.doubts = _doubts(scores, hands, labeled, hand_of_arr, names, limit)
    report.merges = _merges(sim, members, doc_arr, limit)
    report.splits = _splits(sim, members, doc_arr, names, seed, limit)
    report.duplicates = _duplicates(sim, doc_arr, names, limit)
    report.isolated = _isolated(sim, doc_arr, names, limit)

    if clusters is not None:
        rep = json.loads(Path(clusters).read_text())
        levels = [(lv["level"], np.asarray(lv["labels"], dtype=int))
                  for lv in rep.get("levels", [])]
    else:
        from mole.cluster.finch import finch
        res = finch(Xn, metric="cosine")
        levels = [(i, np.asarray(lab, dtype=int))
                  for i, lab in enumerate(res.partitions)]

    report.cluster_levels = [
        {"level": int(i), "n_clusters": int(len(set(lab.tolist()))),
         "labels": [int(v) for v in lab],
         "silhouette": _silhouette(Xn, lab)}
        for i, lab in levels
        if len(lab) == len(rows) and len(set(lab.tolist())) > 1
    ]

    cl = np.asarray(report.cluster_labels, dtype=int) if report.cluster_levels \
        else np.zeros(0, dtype=int)
    if len(cl) == len(rows):
        cohesions = [_cohesion(sim, idx, doc_arr) for idx in members.values()]
        report.new_hands = _new_hands(sim, cl, is_labeled, scores, doc_arr, names,
                                      [c for c in cohesions if np.isfinite(c)], limit)
    return report
