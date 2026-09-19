"""Cross-archive hand candidates from one pooled embedding.

Several collections embedded in ONE space (one checkpoint, one codebook, one
intra-norm setting — `mole embed` over a symlink pool, or several `.npy` files
concatenated here) are searched for scribes that appear in more than one of them.
There is no ground truth for that, so the module does three things:

1. **Measures the domain gap** before any match is believed: how often a page's
   nearest neighbours come from its own archive (kNN archive purity vs chance),
   and how within- and cross-archive cosines compare.
2. **Normalizes it away as far as arithmetic can**: per-archive mean-centering of
   the descriptors (the standard cross-collection fix for VLAD), and CSLS for the
   label-free page pairs (Conneau et al. 2018 — the hubness correction from
   cross-lingual embedding alignment: a page that is everybody's neighbour, e.g.
   a formulaic layout, is pushed down).
3. **Gives the reviewer a yardstick that exists in every archive**: the
   within-archive labeled pairs. Next to every cross-archive score stands the
   share of within-archive SAME-hand pairs it exceeds and the share of
   DIFFERENT-hand pairs it exceeds. No dates, no controls, no cross-archive
   truth are needed — the reviewer is told "this pair scores like the median
   same-hand pair inside an archive", which is a statement the data can support.
   It is context, never a gate: nothing is filtered or re-ranked by it.

**The document pair is the unit.** Two pages in different archives that are
close to each other are a candidate whatever their own archives make of them —
a page from a loose, heterogeneous hand is as eligible as one from a tight
hand. Hand-level lists are built from their strongest document pairs, not from
averages over the hand, and within-archive cohesion is shown, not scored.

Three lists come out, all excluding sibling scans (:mod:`mole.data.docids`) and
near-duplicate images:

* **page pairs** — mutual cross-archive nearest neighbours by CSLS (label-free,
  every page is a query; the headline list);
* **page → foreign hand** — a page whose best-matching hand in ANOTHER archive
  (top-2 mean: two pages of that hand agree) outscores anything at home;
* **hand pairs** — two recorded hands in different archives, ranked by their
  two strongest cross-document matches, with reciprocal ranks and the
  mean/cohesion numbers as context.

``labels.csv`` is only read.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from mole.progress import track

# cosine above which two images are treated as the same charter photographed twice
DUPLICATE_SIM = 0.98
# CSLS neighbourhood (mean of the k nearest cross-archive similarities)
CSLS_K = 10
# a dominant hand must not flood a list
PER_HAND_CAP = 8
# hand pairs: how many times one hand may appear
HAND_PAIR_CAP = 3
# different-hand reference pairs are plentiful; this many is enough for percentiles
REFERENCE_MAX_PAIRS = 200_000


def _pair_key(a: str, b: str) -> str:
    """Canonical 'archiveA|archiveB' name for an archive pair."""
    return "|".join(sorted((str(a), str(b))))


def _take_per_pair(items: list[dict], limit: int, *, cap_key: str | None = None,
                   cap: int = 0) -> list[dict]:
    """Keep the top ``limit`` items of EVERY archive pair (items already ranked).

    One dominant pair (comital ↔ Utrecht in Phase A: 19 of 50 page pairs) must
    not crowd the others out — the reviewer reads the sheet per pair. An optional
    per-``cap_key`` cap (per hand) applies on top.
    """
    n_pair: dict[str, int] = {}
    n_cap: dict[str, int] = {}
    out = []
    for d in items:
        if n_pair.get(d["pair"], 0) >= limit:
            continue
        if cap_key and n_cap.get(d[cap_key], 0) >= cap:
            continue
        n_pair[d["pair"]] = n_pair.get(d["pair"], 0) + 1
        if cap_key:
            n_cap[d[cap_key]] = n_cap.get(d[cap_key], 0) + 1
        out.append(d)
    return out


@dataclass
class CrossReport:
    """Everything `mole cross` computed, JSON-serialisable."""

    n_documents: int = 0
    n_labeled: int = 0
    archives: dict[str, int] = field(default_factory=dict)
    model_id: str | None = None
    centered: bool = True
    gap_raw: dict = field(default_factory=dict)
    gap_centered: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)
    page_pairs: list[dict] = field(default_factory=list)
    page_to_hand: list[dict] = field(default_factory=list)
    hand_pairs: list[dict] = field(default_factory=list)
    duplicates: list[dict] = field(default_factory=list)


# ------------------------------------------------------------------ loading


def _l2(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _hand_if_confident(table, fname: str, min_confidence: float | None) -> str | None:
    hand = table.hand_by_filename.get(fname)
    if hand is None:
        return None
    if min_confidence is not None:
        conf = table.confidence.get(fname)
        if conf is not None and conf < min_confidence:
            return None
    return hand


def load_pool(embeddings: list[str | Path], *, min_confidence: float | None = None):
    """Stack one or more embedding files into one table.

    Returns ``(X, meta, paths, archives, hands, docs)``. ``archives`` is the
    image's parent-folder name; ``hands`` are ``archive/hand`` (``""`` when
    unlabeled or below ``min_confidence``); ``docs`` are ``archive/doc_id``.
    Files must share the embedding width; a differing ``model_id`` is refused
    because the vectors would not live in one space.
    """
    from mole.data.datasets import load_labels
    from mole.data.docids import doc_id_resolver

    Xs, rows, model_ids = [], [], set()
    for e in embeddings:
        npy = Path(e)
        npy = npy if npy.suffix == ".npy" else npy.with_suffix(".npy")
        X = np.load(npy)
        sidecar = npy.with_suffix(".mapping.json")
        meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        r = meta.get("rows") or [{"row": i, "image": str(i)} for i in range(len(X))]
        if len(r) != len(X):
            raise ValueError(f"{npy}: mapping has {len(r)} rows, embedding has {len(X)}")
        if Xs and X.shape[1] != Xs[0].shape[1]:
            raise ValueError(f"{npy}: {X.shape[1]}-d, expected {Xs[0].shape[1]}-d")
        Xs.append(np.asarray(X, dtype=np.float32))
        rows.extend(r)
        model_ids.add(meta.get("model_id"))
    if len(model_ids) > 1:
        raise ValueError(f"embeddings come from different models: {sorted(map(str, model_ids))}"
                         " — one space needs one checkpoint (and one codebook)")
    X = np.concatenate(Xs, axis=0)
    meta = {"model_id": next(iter(model_ids)), "sources": [str(e) for e in embeddings]}

    paths = [Path(r["image"]) for r in rows]
    cache: dict[Path, tuple] = {}
    archives, hands, docs = [], [], []
    for p in paths:
        if p.parent not in cache:
            cache[p.parent] = (load_labels(p.parent), doc_id_resolver(p.parent))
        table, resolve = cache[p.parent]
        a = p.parent.name
        raw = _hand_if_confident(table, p.name, min_confidence)
        archives.append(a)
        hands.append(f"{a}/{raw}" if raw else "")
        docs.append(f"{a}/{resolve(p.name)}")
    return X, meta, paths, archives, hands, docs


# ------------------------------------------------------------- normalisation


def center_by_archive(X: np.ndarray, archives: list[str]) -> np.ndarray:
    """Subtract each archive's mean vector, then re-normalise.

    Removes the component every page of an archive shares — how that repository
    scans, binarizes and crops — which is exactly what a cross-archive cosine
    must not reward. Costs a little within-archive structure when one hand
    dominates an archive (its signature is part of the mean).
    """
    Xc = np.asarray(X, dtype=np.float32).copy()
    arch = np.asarray(archives, dtype=object)
    for a in sorted(set(archives)):
        idx = np.where(arch == a)[0]
        Xc[idx] -= Xc[idx].mean(axis=0, keepdims=True)
    return _l2(Xc)


def _masked_sim(Xn: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Cosine matrix with self and sibling-scan entries set to -inf."""
    sim = Xn @ Xn.T
    same_doc = docs[:, None] == docs[None, :]
    sim[same_doc] = -np.inf
    return sim


def _purity(sim: np.ndarray, arch: np.ndarray, rows: np.ndarray, ks) -> dict:
    """Share of each query row's k nearest neighbours that share its archive."""
    kmax = min(max(ks), sim.shape[1] - 1)
    sub = sim[rows]
    top = np.argpartition(-sub, kmax, axis=1)[:, :kmax]
    order = np.argsort(-np.take_along_axis(sub, top, axis=1), axis=1)   # argpartition is unordered
    top = np.take_along_axis(top, order, axis=1)
    same = arch[top] == arch[rows][:, None]
    return {k: float(same[:, :min(k, kmax)].mean()) for k in ks}


def gap_diagnostics(Xn: np.ndarray, archives: list[str], docs: list[str],
                    hands: list[str] | None = None,
                    ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """How much the space ranks archives rather than hands.

    ``purity@k``: share of a page's k nearest (cross-document) neighbours that
    come from its own archive, averaged over pages, next to ``chance`` (what a
    neighbour-blind draw would give). Much of that is legitimate — a labeled
    page's nearest neighbour SHOULD be its own hand at home — so
    ``purity_other_hand@k`` repeats it over labeled pages with every same-hand
    pair masked: when a page cannot find its own scribe, does it still stay in
    its archive? That is the archive signature proper. Also the median cosine
    of within- vs cross-archive pairs and per archive pair.
    """
    arch = np.asarray(archives, dtype=object)
    doc_arr = np.asarray(docs, dtype=object)
    sim = _masked_sim(Xn, doc_arr)
    n = len(arch)
    purity = {f"purity@{k}": v for k, v in _purity(sim, arch, np.arange(n), ks).items()}
    if hands is not None:
        hand_arr = np.asarray(hands, dtype=object)
        lab = np.where(hand_arr != "")[0]
        if len(lab) >= 2:
            masked = sim.copy()
            same_hand = (hand_arr[:, None] == hand_arr[None, :]) & (hand_arr[:, None] != "")
            masked[same_hand] = -np.inf
            purity.update({f"purity_other_hand@{k}": v
                           for k, v in _purity(masked, arch, lab, ks).items()})
    counts = {a: int((arch == a).sum()) for a in sorted(set(archives))}
    chance = float(np.mean([(counts[a] - 1) / max(n - 1, 1) for a in arch]))

    iu = np.triu_indices(n, k=1)
    vals = sim[iu]
    ok = np.isfinite(vals)
    within = (arch[iu[0]] == arch[iu[1]]) & ok
    cross = (arch[iu[0]] != arch[iu[1]]) & ok
    per_pair = {}
    names = sorted(counts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = ok & (((arch[iu[0]] == a) & (arch[iu[1]] == b))
                      | ((arch[iu[0]] == b) & (arch[iu[1]] == a)))
            if m.any():
                per_pair[f"{a}|{b}"] = float(np.median(vals[m]))
    return {**purity, "chance": chance,
            "within_median": float(np.median(vals[within])) if within.any() else None,
            "cross_median": float(np.median(vals[cross])) if cross.any() else None,
            "per_archive_pair_median": per_pair}


# ------------------------------------------------------------ reference


def reference_distributions(sim: np.ndarray, archives: np.ndarray, hands: np.ndarray,
                            seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Within-archive labeled pair cosines: ``(same_hand, different_hand)``, sorted.

    Both are cross-document by construction (``sim`` already masks siblings).
    The different-hand set is subsampled to :data:`REFERENCE_MAX_PAIRS`.
    """
    lab = np.where(hands != "")[0]
    if len(lab) < 2:
        return np.zeros(0), np.zeros(0)
    sub = sim[np.ix_(lab, lab)]
    iu = np.triu_indices(len(lab), k=1)
    v = sub[iu]
    ok = np.isfinite(v)
    same_arch = archives[lab][iu[0]] == archives[lab][iu[1]]
    same_hand = hands[lab][iu[0]] == hands[lab][iu[1]]
    same = np.sort(v[ok & same_arch & same_hand])
    diff = v[ok & same_arch & ~same_hand]
    if len(diff) > REFERENCE_MAX_PAIRS:
        rng = np.random.default_rng(seed)
        diff = rng.choice(diff, REFERENCE_MAX_PAIRS, replace=False)
    return same, np.sort(diff)


def _pct(sorted_vals: np.ndarray, s: float) -> float | None:
    """Share of ``sorted_vals`` strictly below ``s`` (None if no reference)."""
    if sorted_vals.size == 0 or not np.isfinite(s):
        return None
    return float(np.searchsorted(sorted_vals, s, side="left") / sorted_vals.size)


def _anchor(same: np.ndarray, diff: np.ndarray, s: float) -> dict:
    return {"same_hand_pct": _pct(same, s), "diff_hand_pct": _pct(diff, s)}


# ------------------------------------------------------------------ lists


def _top2_mean_matrix(sim: np.ndarray, members: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """``[N, H]`` mean of the top-2 similarities from every page to every hand.

    Vectorised twin of :func:`mole.review.suggest.hand_score_matrix`; ``sim``
    already carries -inf on sibling scans, so a hand whose pages are all
    siblings of the query scores -inf (no evidence). With one usable page its
    own value stands.
    """
    hands = sorted(members)
    out = np.full((sim.shape[0], len(hands)), -np.inf, dtype=np.float32)
    for j, h in enumerate(hands):
        sub = sim[:, members[h]]
        if sub.shape[1] == 1:
            out[:, j] = sub[:, 0]
            continue
        part = np.partition(sub, -2, axis=1)[:, -2:]
        best, second = part[:, 1], part[:, 0]
        one = ~np.isfinite(second) & np.isfinite(best)
        out[:, j] = np.where(one, best, 0.5 * (best + second))
    return out, hands


def _cohesion(sim: np.ndarray, idx: np.ndarray) -> float:
    if len(idx) < 2:
        return float("nan")
    sub = sim[np.ix_(idx, idx)]
    v = sub[np.isfinite(sub)]
    return float(v.mean()) if v.size else float("nan")


def hand_pairs(sim: np.ndarray, members: dict[str, np.ndarray], archive_of_hand: dict[str, str],
               same: np.ndarray, diff: np.ndarray, limit: int) -> list[dict]:
    """Cross-archive hand pairs, ranked by their two strongest document matches.

    ``score`` is the mean of the two highest page-to-page cosines between the
    hands (two documents agree — one lucky pair is the classic false friend;
    with a single usable pair its value stands). The mean over all pairs and
    each hand's own cohesion are reported as context, as is
    ``closeness = cross_mean - mean(cohesion)`` from the merges list, but none of
    them ranks anything: a loose hand may still hold a real match. ``rank_ab``
    is hand b's rank among all hands of ITS archive by ``score`` from a;
    ``rank_ba`` the reverse — 1/1 means each is the other's first choice there.
    """
    hands = sorted(h for h in members if len(members[h]) >= 1)
    coh = {h: _cohesion(sim, members[h]) for h in hands}
    cross: dict[tuple[str, str], float] = {}
    mean: dict[tuple[str, str], float] = {}
    best: dict[tuple[str, str], tuple[float, int, int]] = {}
    for a in track(hands, "Scoring hand pairs", unit="hand"):
        for b in hands:
            if b <= a or archive_of_hand[a] == archive_of_hand[b]:
                continue
            blk = sim[np.ix_(members[a], members[b])]
            v = blk[np.isfinite(blk)]
            if v.size == 0:
                continue
            top = np.sort(v)[-2:]
            cross[(a, b)] = float(top.mean())
            mean[(a, b)] = float(v.mean())
            k = int(np.argmax(np.where(np.isfinite(blk), blk, -np.inf)))
            ia, ib = divmod(k, blk.shape[1])
            best[(a, b)] = (float(blk[ia, ib]), int(members[a][ia]), int(members[b][ib]))

    def cross_of(x, y):
        return cross.get((x, y) if x < y else (y, x))

    def rank_in(archive: str, src: str, target: str) -> int:
        cands = [(cross_of(src, h), h) for h in hands
                 if archive_of_hand[h] == archive and cross_of(src, h) is not None]
        cands.sort(key=lambda t: -t[0])
        return 1 + [h for _, h in cands].index(target)

    out = []
    for (a, b), c in cross.items():
        ca, cb = coh[a], coh[b]
        own = 0.5 * (ca + cb) if (np.isfinite(ca) and np.isfinite(cb)) else None
        bp, ra, rb = best[(a, b)]
        out.append({"hand_a": a, "hand_b": b,
                    "archive_a": archive_of_hand[a], "archive_b": archive_of_hand[b],
                    "pair": _pair_key(archive_of_hand[a], archive_of_hand[b]),
                    "score": c, "cross_mean": mean[(a, b)],
                    "own_similarity": own,
                    "closeness": (mean[(a, b)] - own) if own is not None else None,
                    "cohesion_a": ca if np.isfinite(ca) else None,
                    "cohesion_b": cb if np.isfinite(cb) else None,
                    "n_a": int(len(members[a])), "n_b": int(len(members[b])),
                    "n_pairs": int(len(members[a]) * len(members[b])),
                    "best_pair_similarity": bp, "best_row_a": ra, "best_row_b": rb,
                    "rank_ab": rank_in(archive_of_hand[b], a, b),
                    "rank_ba": rank_in(archive_of_hand[a], b, a),
                    **_anchor(same, diff, c)})
    out.sort(key=lambda d: -d["score"])
    taken: dict[str, int] = {}
    n_pair: dict[str, int] = {}
    kept = []
    for d in out:
        if n_pair.get(d["pair"], 0) >= limit:
            continue
        if taken.get(d["hand_a"], 0) >= HAND_PAIR_CAP or taken.get(d["hand_b"], 0) >= HAND_PAIR_CAP:
            continue
        taken[d["hand_a"]] = taken.get(d["hand_a"], 0) + 1
        taken[d["hand_b"]] = taken.get(d["hand_b"], 0) + 1
        n_pair[d["pair"]] = n_pair.get(d["pair"], 0) + 1
        kept.append(d)
    return kept


def page_to_hand(sim: np.ndarray, members: dict[str, np.ndarray], archive_of_hand: dict[str, str],
                 archives: np.ndarray, hands: np.ndarray, names: list[str],
                 same: np.ndarray, diff: np.ndarray, limit: int) -> list[dict]:
    """Pages whose best hand in another archive outscores every hand at home.

    Every page is a query (labeled ones too — a recorded Antwerp page sitting
    with a comital hand is evidence). ``own_score`` is the page's own recorded
    hand when it has one, else the best hand of its archive; ``delta`` is
    foreign minus that. Ranked by the foreign score, capped per foreign hand.
    """
    scores, hand_list = _top2_mean_matrix(sim, members)
    hand_arch = np.asarray([archive_of_hand[h] for h in hand_list], dtype=object)
    hand_idx = {h: j for j, h in enumerate(hand_list)}
    support = {h: len(members[h]) for h in hand_list}
    out = []
    for i in track(range(sim.shape[0]), "Matching pages to foreign hands", unit="page"):
        foreign = hand_arch != archives[i]
        if not foreign.any():
            continue
        fs = np.where(foreign, scores[i], -np.inf)
        j = int(np.argmax(fs))
        s = float(fs[j])
        if not np.isfinite(s):
            continue
        fs2 = fs.copy()
        fs2[j] = -np.inf
        runner = int(np.argmax(fs2))
        runner_s = float(fs2[runner])
        if hands[i]:
            own_hand, own_s = hands[i], float(scores[i, hand_idx[hands[i]]])
        else:
            hs = np.where(~foreign, scores[i], -np.inf)
            k = int(np.argmax(hs))
            own_hand, own_s = (hand_list[k], float(hs[k])) if np.isfinite(hs[k]) else ("", float("nan"))
        out.append({"row": i, "document": names[i], "archive": str(archives[i]),
                    "pair": _pair_key(archives[i], hand_arch[j]),
                    "recorded_hand": hands[i] or "",
                    "hand": hand_list[j], "score": s,
                    "runner_up": hand_list[runner] if np.isfinite(runner_s) else None,
                    "runner_up_score": runner_s if np.isfinite(runner_s) else None,
                    "margin": (s - runner_s) if np.isfinite(runner_s) else None,
                    "own_hand": own_hand, "own_score": own_s if np.isfinite(own_s) else None,
                    "delta": (s - own_s) if np.isfinite(own_s) else None,
                    "n_support": int(support[hand_list[j]]),
                    **_anchor(same, diff, s)})
    out.sort(key=lambda d: -d["score"])
    return _take_per_pair(out, limit, cap_key="hand", cap=PER_HAND_CAP)


def csls_matrix(sim: np.ndarray, archives: np.ndarray, k: int = CSLS_K) -> np.ndarray:
    """CSLS over cross-archive pairs: ``2·cos(i,j) − r_B(i) − r_A(j)``.

    ``r_B(i)`` is the mean of i's k highest similarities into archive B (j's
    archive). Same-archive entries are set to -inf; so are sibling scans (they
    arrive as -inf in ``sim``).
    """
    n = sim.shape[0]
    names = sorted(set(archives.tolist()))
    r = np.zeros((n, len(names)), dtype=np.float32)
    for b_i, b in enumerate(names):
        cols = np.where(archives == b)[0]
        sub = sim[:, cols]
        kk = min(k, sub.shape[1])
        top = np.partition(sub, -kk, axis=1)[:, -kk:]
        top = np.where(np.isfinite(top), top, np.nan)
        r[:, b_i] = np.nan_to_num(np.nanmean(top, axis=1), nan=0.0)
    arch_idx = np.asarray([names.index(a) for a in archives])
    r_to = r[:, arch_idx]                       # r_to[i, j] = r_B(i), B = archive of j
    out = 2.0 * sim - r_to - r_to.T             # r_to.T[i, j] = r_A(j), A = archive of i
    out[archives[:, None] == archives[None, :]] = -np.inf
    return out


def page_pairs(sim: np.ndarray, archives: np.ndarray, hands: np.ndarray, names: list[str],
               same: np.ndarray, diff: np.ndarray, limit: int,
               k: int = CSLS_K) -> tuple[list[dict], list[dict], np.ndarray]:
    """Mutual cross-archive nearest neighbours by CSLS, and near-duplicates.

    Returns ``(pairs, duplicates, csls)``. Each pair carries the raw cosine,
    the CSLS score, both pages' best cosine at home (the anchor: "at home its
    closest page is 0.41; this foreign page is 0.52") and the reference shares.
    Pairs at or above :data:`DUPLICATE_SIM` are reported as duplicates instead.
    """
    cs = csls_matrix(sim, archives, k=k)
    nn = np.argmax(cs, axis=1)
    home = sim.copy()
    home[archives[:, None] != archives[None, :]] = -np.inf
    home_best = np.max(home, axis=1)
    pairs, dups = [], []
    for i in range(len(nn)):
        j = int(nn[i])
        if not np.isfinite(cs[i, j]) or int(nn[j]) != i or j < i:
            continue
        cos = float(sim[i, j])
        rec = {"row_a": i, "row_b": j, "document_a": names[i], "document_b": names[j],
               "archive_a": str(archives[i]), "archive_b": str(archives[j]),
               "pair": _pair_key(archives[i], archives[j]),
               "hand_a": hands[i] or "", "hand_b": hands[j] or "",
               "similarity": cos, "csls": float(cs[i, j]),
               "home_best_a": float(home_best[i]) if np.isfinite(home_best[i]) else None,
               "home_best_b": float(home_best[j]) if np.isfinite(home_best[j]) else None,
               **_anchor(same, diff, cos)}
        (dups if cos >= DUPLICATE_SIM else pairs).append(rec)
    pairs.sort(key=lambda d: -d["csls"])
    dups.sort(key=lambda d: -d["similarity"])
    return _take_per_pair(pairs, limit), dups, cs


# ------------------------------------------------------------------ driver


def build_cross(embeddings: list[str | Path], *, limit: int = 12, center: bool = True,
                min_confidence: float | None = None, seed: int = 0,
                out: str | Path | None = None) -> tuple[CrossReport, dict]:
    """Run the whole analysis. Returns ``(report, table)``.

    ``limit`` is per list AND per archive pair: every pair of archives keeps its
    own top ``limit`` candidates, so the sheet can be read pair by pair.

    ``table`` holds what the renderer needs beyond the report: the centered,
    normalised vectors ``Xn``, the masked similarity, the CSLS matrix, paths,
    archives, hands, docs and members.
    """
    X, meta, paths, archive_list, hand_list, doc_list = load_pool(
        embeddings, min_confidence=min_confidence)
    if len(set(archive_list)) < 2:
        raise ValueError("cross-archive review needs pages from at least two dataset "
                         "folders; this embedding holds only "
                         f"{sorted(set(archive_list))}")
    archives = np.asarray(archive_list, dtype=object)
    hands = np.asarray(hand_list, dtype=object)
    docs = np.asarray(doc_list, dtype=object)
    names = [f"{p.parent.name}/{p.name}" for p in paths]

    report = CrossReport(
        n_documents=len(X), n_labeled=int((hands != "").sum()),
        archives={a: int((archives == a).sum()) for a in sorted(set(archive_list))},
        model_id=meta.get("model_id"), centered=center)

    Xraw = _l2(X)
    report.gap_raw = gap_diagnostics(Xraw, archive_list, doc_list, hand_list)
    Xn = center_by_archive(X, archive_list) if center else Xraw
    report.gap_centered = (gap_diagnostics(Xn, archive_list, doc_list, hand_list) if center
                           else dict(report.gap_raw))

    sim = _masked_sim(Xn, docs)
    same, diff = reference_distributions(sim, archives, hands, seed=seed)
    report.reference = {
        "n_same_hand_pairs": int(same.size), "n_diff_hand_pairs": int(diff.size),
        "same_hand_quartiles": ([float(np.quantile(same, q)) for q in (0.25, 0.5, 0.75)]
                                if same.size else None),
        "diff_hand_quartiles": ([float(np.quantile(diff, q)) for q in (0.25, 0.5, 0.75)]
                                if diff.size else None),
    }

    members: dict[str, np.ndarray] = {}
    for h in sorted({h for h in hand_list if h}):
        members[h] = np.where(hands == h)[0]
    archive_of_hand = {h: h.split("/", 1)[0] for h in members}

    report.hand_pairs = hand_pairs(sim, members, archive_of_hand, same, diff, limit)
    report.page_to_hand = page_to_hand(sim, members, archive_of_hand, archives, hands,
                                       names, same, diff, limit)
    report.page_pairs, report.duplicates, cs = page_pairs(
        sim, archives, hands, names, same, diff, limit)

    table = {"Xn": Xn, "sim": sim, "csls": cs, "paths": paths, "names": names,
             "archives": archives, "hands": hands, "docs": docs, "members": members}
    if out is not None:
        Path(out).write_text(json.dumps(asdict(report), indent=1))
    return report, table


def format_report(r: CrossReport) -> str:
    """Terminal summary: the gap, the anchors, and the head of each list."""
    def gap(g: dict) -> str:
        p = ", ".join(f"{k} {g[k]:.2f}" for k in g if k.startswith("purity@"))
        o = ", ".join(f"@{k.split('@')[1]} {g[k]:.2f}" for k in g if k.startswith("purity_other"))
        return (f"{p} (chance {g['chance']:.2f})"
                + (f"; other-hand-only {o}" if o else "")
                + f"; median cosine within {g['within_median']:.3f} vs cross {g['cross_median']:.3f}")

    arch = ", ".join(f"{a} {n}" for a, n in r.archives.items())
    lines = [f"cross-archive review — {r.n_documents} pages ({arch}); {r.n_labeled} labeled",
             f"  archive purity of neighbours, raw:      {gap(r.gap_raw)}"]
    if r.centered:
        lines.append(f"  archive purity of neighbours, centered: {gap(r.gap_centered)}")
    ref = r.reference
    if ref.get("same_hand_quartiles"):
        s, d = ref["same_hand_quartiles"], ref["diff_hand_quartiles"]
        lines.append(f"  within-archive reference: same-hand pairs Q1/med/Q3 "
                     f"{s[0]:.3f}/{s[1]:.3f}/{s[2]:.3f} (n={ref['n_same_hand_pairs']}), "
                     f"different-hand {d[0]:.3f}/{d[1]:.3f}/{d[2]:.3f} (n={ref['n_diff_hand_pairs']})")
    lines.append(f"  page pairs {len(r.page_pairs)} · page→foreign hand {len(r.page_to_hand)} · "
                 f"hand pairs {len(r.hand_pairs)} · possible duplicates {len(r.duplicates)}")
    pairs = sorted({d["pair"] for d in r.page_pairs + r.page_to_hand + r.hand_pairs})
    for pk in pairs:
        n = [sum(1 for d in lst if d["pair"] == pk)
             for lst in (r.page_pairs, r.page_to_hand, r.hand_pairs)]
        lines.append(f"    {pk.replace('|', ' ↔ ')}: {n[0]} / {n[1]} / {n[2]}")
    for d in r.page_pairs[:5]:
        lines.append(f"    {d['document_a']} ↔ {d['document_b']}: cos {d['similarity']:.3f}, "
                     f"csls {d['csls']:.3f}, home {d['home_best_a']:.3f}/{d['home_best_b']:.3f}, "
                     f"> {100 * (d['same_hand_pct'] or 0):.0f}% of same-hand pairs")
    for d in r.hand_pairs[:5]:
        own = f"{d['own_similarity']:.3f}" if d['own_similarity'] is not None else "—"
        lines.append(f"    {d['hand_a']} ~ {d['hand_b']}: top-2 {d['score']:.3f} "
                     f"(mean {d['cross_mean']:.3f} vs own {own}), "
                     f"ranks {d['rank_ab']}/{d['rank_ba']}, "
                     f"> {100 * (d['same_hand_pct'] or 0):.0f}% of same-hand pairs")
    for d in r.duplicates[:5]:
        lines.append(f"    DUPLICATE? {d['document_a']} ≈ {d['document_b']} ({d['similarity']:.4f})")
    return "\n".join(lines)
