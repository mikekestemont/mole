"""Label-review suggestions from a partially labeled embedding.

Archives arrive with *some* hands identified and most documents untouched. This
module turns one ``mole embed`` output plus the existing ``labels.csv`` into
ranked, human-checkable hypotheses. ``labels.csv`` is the sole reference and is
never written here — accepted suggestions leave as a CSV the reviewer downloads.

Three review tasks (built in consecutive stages):

* **outliers**       — a labeled document that does not sit with its recorded
                       hand (stage 1: false positives to reject);
* **attributions**   — an unlabeled document that sits with a known hand
                       (stage 2: false negatives; never overwrites a recorded hand);
* **new hands**      — unlabeled FINCH clusters at the ARI-best cut vs recorded
                       hands (stage 3: a scribe who is not yet named);

Parked (still scored, not shown in the review sheet): merges, splits,
duplicates, isolated.

Two rules keep the numbers honest:

**Sibling scans are not evidence.** Every score excludes documents sharing the
query's ``doc_id`` (:mod:`mole.data.docids`), so "this page looks like hand B"
can never rest on another scan of the very same charter — the scan-shortcut that
``mole eval --cross-doc-only`` exists to kill.

**Only attributions are calibrated.** Hiding each labeled document in turn and
scoring it as if unlabeled gives real ground truth, so attribution scores are
mapped through isotonic regression to an empirical P(top-1 correct). Outliers
have no such ground truth; they enter only when both isolation and a closer
hand clear a high bar (recorded labels are mostly right). New hands carry a
*relative* strength, which the UI must word as a question rather than a claim.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# a hand needs this many documents before "is it really two hands?" is askable
MIN_DOCS_FOR_SPLIT = 4
# isolation z needs a within-hand distribution (leave-one-out median/MAD)
MIN_DOCS_FOR_OUTLIER = 3
# cosine above which two different documents are treated as the same image
DUPLICATE_SIM = 0.98
# a dominant hand (KA_8) must not flood a review tab
OUTLIER_PER_HAND_CAP = 8
ATTRIB_PER_HAND_CAP = 8
# Stage 1 (false positives): recorded labels are mostly right, so a
# misidentification needs both a real alternative hand and a real isolation
# from the recorded one. z=1 / any positive gap is expected noise.
OUTLIER_Z_MIN = 3.0
OUTLIER_GAP_MIN = 0.05          # cosine; a hair closer to another hand is not a misID
# Stage 2 (false negatives): unlabeled pages are the common case. Skip a
# proposal only if it would already look like a mild outlier inside that hand.
JOIN_Z_MAX = 1.0


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
    outliers: list[dict] = field(default_factory=list)
    duplicates: list[dict] = field(default_factory=list)
    isolated: list[dict] = field(default_factory=list)
    calibration: dict = field(default_factory=dict)
    # FINCH's whole hierarchy, kept so the renderer can offer one colour scheme
    # per level without paying for the clustering a second time. Each entry is
    # {level, n_clusters, labels, silhouette}; `silhouette` is None where it is
    # undefined (fewer than 2 clusters, or one cluster per document).
    cluster_levels: list[dict] = field(default_factory=list)
    new_hand_level: dict = field(default_factory=dict)

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
def _per_hand_take(items: list[dict], cap: int, key: str = "hand") -> list[dict]:
    """Keep ranking, but at most ``cap`` items per hand."""
    counts: dict[str, int] = {}
    out = []
    for d in items:
        h = d[key]
        if counts.get(h, 0) >= cap:
            continue
        counts[h] = counts.get(h, 0) + 1
        out.append(d)
    return out


def _attributions(scores, hands, unlabeled, names, members, limit,
                  z_min: float = JOIN_Z_MAX):
    """Unlabeled document -> the known hand it sits with.

    A page that would immediately be an outlier *inside* the proposed hand is
    not offered: Keep would just recreate a false positive. Ranked later by
    calibrated probability; ``limit`` is applied by the caller after that.
    """
    idx = {h: j for j, h in enumerate(hands)}
    own_by_hand: dict[str, np.ndarray] = {}
    for h, mems in members.items():
        j = idx.get(h)
        if j is None:
            continue
        vals = [float(scores[int(i), j]) for i in mems if np.isfinite(scores[int(i), j])]
        own_by_hand[h] = np.asarray(vals, dtype=np.float64)

    out = []
    for i in unlabeled:
        row = scores[i]
        if not np.isfinite(row).any():
            continue
        order = np.argsort(-row)
        best = int(order[0])
        hand = hands[best]
        score = float(row[best])
        second = float(row[order[1]]) if len(order) > 1 and np.isfinite(row[order[1]]) else float("nan")
        others = own_by_hand.get(hand, np.zeros(0, dtype=np.float64))
        z = _mad_z(score, others) if others.size >= MIN_DOCS_FOR_OUTLIER else 0.0
        if others.size >= MIN_DOCS_FOR_OUTLIER and z >= z_min:
            continue
        out.append({
            "row": int(i), "document": names[i], "hand": hand,
            "score": score,
            "margin": float(score - second) if np.isfinite(second) else None,
            "runner_up": hands[int(order[1])] if len(order) > 1 else None,
            "runner_up_score": float(second) if np.isfinite(second) else None,
            "n_support": int(len(members.get(hand, []))),
            "join_z": float(z),
        })
    out.sort(key=lambda d: -d["score"])
    return out[:limit] if limit else out


def _mad_z(value: float, others: np.ndarray) -> float:
    """Positive z = ``value`` is below the median of ``others`` (worse own-hand match).

    MAD scale; a degenerate hand (everyone identical) yields 0 unless ``value``
    itself is the odd one out, in which case it is a large positive z.
    """
    if others.size < 2:
        return 0.0
    med = float(np.median(others))
    mad = float(np.median(np.abs(others - med)))
    if mad < 1e-8:
        if abs(value - med) < 1e-6:
            return 0.0
        return 10.0 if value < med else -10.0
    return float((med - value) / mad)


def _outliers(scores, hands, labeled, hand_of, names, members, limit,
              per_hand_cap: int = OUTLIER_PER_HAND_CAP,
              z_min: float = OUTLIER_Z_MIN):
    """Labeled documents that should perhaps be rejected from their recorded hand.

    Two leave-one-out signals, both read off the existing ``labels.csv`` (never
    modified): isolation within the recorded hand (robust z of own-hand score)
    and a closer *other* known hand. Both are required, and each has a high
    bar: the prior that a recorded identification is wrong is much lower than
    the prior that an unlabeled page belongs to a known hand. Ranked by z then
    gap, then capped per hand so a dominant scribe cannot flood the list.
    """
    idx = {h: j for j, h in enumerate(hands)}
    own: dict[int, float] = {}
    for i in labeled:
        h = hand_of[i]
        if h not in idx:
            continue
        s = float(scores[i, idx[h]])
        if np.isfinite(s):
            own[i] = s

    by_hand: dict[str, list[int]] = {}
    for i in own:
        by_hand.setdefault(hand_of[i], []).append(i)

    cands = []
    for h, idxs in by_hand.items():
        scores_h = np.array([own[i] for i in idxs], dtype=np.float64)
        can_z = len(idxs) >= MIN_DOCS_FOR_OUTLIER
        for k, i in enumerate(idxs):
            others = np.delete(scores_h, k) if can_z else scores_h
            z = _mad_z(own[i], others) if can_z else 0.0
            row = scores[i].copy()
            row[idx[h]] = -np.inf
            closer_hand = None
            closer_score = None
            gap = 0.0
            if np.isfinite(row).any():
                best = int(np.argmax(row))
                closer_score = float(row[best])
                gap = closer_score - own[i]
                if gap > 0:
                    closer_hand = hands[best]
                else:
                    gap = 0.0
                    closer_score = None
            if (closer_hand is None or gap < OUTLIER_GAP_MIN or z < z_min):
                continue
            cands.append({
                "row": int(i), "document": names[i], "hand": h,
                "own_score": own[i], "z": z,
                "closer_hand": closer_hand, "closer_score": closer_score,
                "gap": float(gap), "n_support": int(len(members[h])),
            })

    by: dict[str, list[dict]] = {}
    for d in cands:
        by.setdefault(d["hand"], []).append(d)
    out = []
    for items in by.values():
        items.sort(key=lambda d: (-d["z"], -d["gap"]))
        out.extend(items[:per_hand_cap])
    out.sort(key=lambda d: (-d["z"], -d["gap"]))
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


def _medoid(sim: np.ndarray, idx: np.ndarray, doc_ids: np.ndarray) -> int:
    """Most central document in ``idx`` (mean cosine to the others, siblings out)."""
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 1:
        return int(idx[0])
    sub = sim[np.ix_(idx, idx)].astype(np.float64).copy()
    np.fill_diagonal(sub, np.nan)
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            if a < b and doc_ids[i] == doc_ids[j]:
                sub[a, b] = sub[b, a] = np.nan
    means = np.nanmean(sub, axis=1)
    if not np.isfinite(means).any():
        return int(idx[0])
    return int(idx[int(np.nanargmax(means))])


def _pick_partition_for_new_hands(levels: list[dict]
                                  ) -> tuple[np.ndarray | None, dict]:
    """FINCH level whose partition best recovers the recorded hands (ARI).

    That granularity is what "a hand" means in this archive. New-hand proposals
    are then the clusters at that cut which contain *no* labeled document.
    Silhouette is the fallback when nothing is labeled. Ties go to the finer
    cut so a two-document unnamed hand is not merged away.
    """
    finch = [lv for lv in levels if str(lv.get("level", "")).startswith("FINCH")]
    pool = finch or list(levels)
    if not pool:
        return None, {}
    use_ari = any(lv.get("ari") is not None for lv in pool)

    def key(lv):
        score = lv.get("ari") if use_ari else lv.get("silhouette")
        if score is None:
            score = -2.0
        return (float(score), int(lv.get("n_clusters") or 0))

    best = max(pool, key=key)
    return np.asarray(best["labels"], dtype=int), {
        "level": best.get("level"),
        "n_clusters": best.get("n_clusters"),
        "ari": best.get("ari"),
        "silhouette": best.get("silhouette"),
        "criterion": "ari" if use_ari else "silhouette",
    }


def _new_hands(sim, cluster_labels, is_labeled, scores, doc_ids, names,
               hand_cohesions, limit, *, level: dict | None = None):
    """Unlabeled FINCH clusters at the ARI-chosen cut — candidate unnamed hands.

    A cluster that shares even one recorded label is not "missed": it already
    overlaps a known scribe (stage 2's problem). A cluster closer to a named
    hand than to itself is the same. Ranked by size, then cohesion.
    """
    ref = float(np.nanmedian(hand_cohesions)) if len(hand_cohesions) else 0.0
    meta = level or {}
    out = []
    for c in sorted(set(int(v) for v in cluster_labels) - {NOISE}):
        idx = np.where(cluster_labels == c)[0]
        if len(idx) < 3:
            continue
        if bool(is_labeled[idx].any()):
            continue                                  # overlaps a recorded hand
        coh = _cohesion(sim, idx, doc_ids)
        if not np.isfinite(coh) or (np.isfinite(ref) and coh < ref):
            continue                                  # looser than a typical hand
        if scores.ndim == 2 and scores.shape[1]:
            per = []
            for i in idx:
                row = scores[int(i)]
                per.append(float(np.nanmax(row)) if np.isfinite(row).any() else np.nan)
            best_known = float(np.nanmedian(np.asarray(per, dtype=np.float64)))
        else:
            best_known = float("-inf")
        if np.isfinite(best_known) and best_known >= coh:
            continue                                  # sits with a named hand
        q = _medoid(sim, idx, doc_ids)
        out.append({"cluster": c, "n_docs": int(len(idx)), "cohesion": coh,
                    "reference_cohesion": ref, "closest_known_score": best_known,
                    "row": q, "document": names[q],
                    "documents": [names[i] for i in idx],
                    "rows": [int(i) for i in idx],
                    "level": meta.get("level"), "ari": meta.get("ari")})
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


NOISE = -1          # HDBSCAN's "belongs to no cluster" label


def _silhouette(Xn: np.ndarray, labels: np.ndarray) -> float | None:
    """Mean silhouette of one partition (cosine), or None where it is undefined.

    Noise points are EXCLUDED. Scoring them as if they were a cluster would
    reward a method for dumping every awkward charter into one bag — the opposite
    of what we want to know. A partition also needs at least 2 clusters and fewer
    clusters than documents.
    """
    keep = labels != NOISE
    lab = labels[keep]
    n_lab = len(set(lab.tolist()))
    if n_lab < 2 or n_lab >= len(lab):
        return None
    try:
        from sklearn.metrics import silhouette_score

        return float(silhouette_score(Xn[keep], lab, metric="cosine"))
    except Exception:
        return None


def _agreement(hands: np.ndarray, labels: np.ndarray) -> dict:
    """Score a partition against the hands that ARE recorded.

    This is the better selector when labels exist: silhouette only asks whether a
    partition is geometrically tidy, while agreement asks whether it recovers what
    the archivist already established. Both are reported — silhouette still works
    where nothing is labeled.

    Unlabeled documents are excluded (an unlabeled charter may well belong to a
    recorded hand, so counting it as an error would be wrong), and so are HDBSCAN's
    noise points — a method should not be scored on the charters it declined to
    place.
    """
    from mole.cluster.finch import cluster_agreement

    truth = [h if h and lab != NOISE else None
             for h, lab in zip(hands.tolist(), labels.tolist())]
    if not any(t is not None for t in truth):
        return {"ari": None, "nmi": None, "purity": None, "n_scored": 0}
    a = cluster_agreement(truth, labels)
    return {"ari": a["ari"], "nmi": a["nmi"], "purity": a["purity"],
            "n_scored": a["n_labeled"]}


def _hdbscan_levels(Xn: np.ndarray, sizes=(2, 3, 5)) -> list[tuple[str, np.ndarray]]:
    """Density clustering at a few minimum cluster sizes.

    Run on a PCA-reduced space, not the raw descriptor: density estimation in
    38,400 dimensions is close to meaningless, and reducing first is the standard
    remedy. ``min_cluster_size=2`` is included deliberately — this corpus has 69
    two-document hands, and a floor of 5 cannot represent them at all — with the
    understanding that it is also the noisiest setting.
    """
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:                      # scikit-learn < 1.3
        return []

    from mole.viz.scatter import _pca

    Z = _pca(Xn, min(50, max(2, min(Xn.shape) - 1)))
    out = []
    for mcs in sizes:
        if mcs > len(Z):
            continue
        # sklearn 1.10 flips the default of `copy`; set it explicitly so the
        # FutureWarning stays out of the user's terminal, but only if this version
        # actually accepts the argument.
        import inspect

        kw = {"min_cluster_size": int(mcs)}
        if "copy" in inspect.signature(HDBSCAN.__init__).parameters:
            kw["copy"] = True
        try:
            lab = HDBSCAN(**kw).fit_predict(Z)
        except Exception:
            continue
        if len(set(lab.tolist()) - {NOISE}) > 1:
            out.append((f"HDBSCAN min-size {mcs}", np.asarray(lab, dtype=int)))
    return out


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
                 limit: int = 100, seed: int = 0,
                 cluster_method: str = "both", lists: bool = True,
                 per_hand_cap: int = OUTLIER_PER_HAND_CAP) -> ReviewReport:
    """Build suggestion lists for one embedding file from the existing labels.csv.

    ``labels.csv`` is only read. ``clusters`` is an optional ``mole cluster``
    report; without one, FINCH's finest partition is computed here so the
    "possible new hand" list always exists. ``lists=False`` (``mole viz``)
    still computes colour-scheme partitions and skips the suggestion engine.
    ``limit`` caps each list — these are for human review, and a list nobody
    can finish reading is a list nobody reads.
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

    report = ReviewReport(
        n_documents=len(rows), n_labeled=int(is_labeled.sum()), n_hands=len(members),
        datasets=sorted({p.name for p in parents}), model_id=meta.get("model_id"))
    # Clustering does not need labels, and an archive with NONE is exactly when
    # discovered clusters matter most — so compute them before the early return.

    levels: list[tuple[str, np.ndarray]] = []
    if clusters is not None:
        rep = json.loads(Path(clusters).read_text())
        levels += [(f"FINCH L{lv['level']}", np.asarray(lv["labels"], dtype=int))
                   for lv in rep.get("levels", [])]
    elif cluster_method in ("finch", "both"):
        from mole.cluster.finch import finch
        res = finch(Xn, metric="cosine")
        levels += [(f"FINCH L{i}", np.asarray(lab, dtype=int))
                   for i, lab in enumerate(res.partitions)]
    if cluster_method in ("hdbscan", "both"):
        levels += _hdbscan_levels(Xn)

    report.cluster_levels = [
        {"level": name,
         "n_clusters": int(len(set(lab.tolist()) - {NOISE})),
         "n_noise": int((lab == NOISE).sum()),
         "labels": [int(v) for v in lab],
         "silhouette": _silhouette(Xn, lab),
         **_agreement(hand_of_arr, lab)}
        for name, lab in levels
        if len(lab) == len(rows) and len(set(lab.tolist()) - {NOISE}) > 1
    ]

    if not members:
        return report      # nothing labeled: no hand to reason from, clusters only
    if not lists:
        return report      # viz: colour schemes only; no suggestion lists

    sim = (Xn @ Xn.T).astype(np.float32)
    scores, hands = hand_score_matrix(sim, members, doc_arr)
    cal = _calibrate(scores, hands, labeled, hand_of_arr)
    report.calibration = cal

    report.outliers = _outliers(scores, hands, labeled, hand_of_arr, names, members,
                                limit, per_hand_cap=per_hand_cap)
    attrib = _attributions(scores, hands, unlabeled, names, members, limit=None)
    for a in attrib:
        a["calibrated_p"] = _apply_calibration(cal, a["score"])
    attrib.sort(key=lambda d: (
        -(d["calibrated_p"] if d["calibrated_p"] is not None else -1.0),
        -d["score"]))
    report.attributions = _per_hand_take(attrib, ATTRIB_PER_HAND_CAP)[:limit]
    report.merges = _merges(sim, members, doc_arr, limit)
    report.splits = _splits(sim, members, doc_arr, names, seed, limit)
    report.duplicates = _duplicates(sim, doc_arr, names, limit)
    report.isolated = _isolated(sim, doc_arr, names, limit)


    cl, meta = _pick_partition_for_new_hands(report.cluster_levels)
    report.new_hand_level = meta
    if cl is not None and len(cl) == len(rows):
        cohesions = [_cohesion(sim, idx, doc_arr) for idx in members.values()]
        report.new_hands = _new_hands(sim, cl, is_labeled, scores, doc_arr, names,
                                      [c for c in cohesions if np.isfinite(c)], limit,
                                      level=meta)
    return report
