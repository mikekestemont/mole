"""The review sheet and the viz map: one self-contained HTML each.

``mole viz`` is the map + charter viewer. ``mole review`` is a separate file
with a three-tab case queue: false positives (recorded page vs its hand), false
negatives (unattributed page vs a known hand) and new hands (an unlabeled
cluster shown as a grid of pages with a checkbox each — tick the pages that are
one scribe; nothing ticked rejects the hand). Decisions are Keep / Reject /
Unsure and leave as a CSV; ``labels.csv`` is never written.

Design rules, all of which have a reason:

* **Plain language by default.** No cosine appears unless "show the numbers" is
  ticked. Uncalibrated lists ask a QUESTION, because they have no ground truth.
* **Whole pages, losslessly.** See :mod:`mole.review.images` — thumbnails are too
  fuzzy to judge letterforms and lossy coding is *bigger* on bilevel scans.
* **One file.** Images are inlined, so there is no folder to keep alongside it and
  nothing to break when it is emailed.
* **labels.csv is the reference.** Suggested attributions never overwrite it.
* **Cosine distances are always on screen.** Similarity is the retrieval metric;
  the reviewer sees ``1 - cosine`` to the visible neighbour and to the hand
  (mean of the two closest pages).
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np

# how many other pages of the recorded hand the reviewer can flip through
CLASS_NEIGHBOR_CAP = 12
DEFAULT_LIMIT = 25
DEFAULT_MAX_MB = 10.0

_SECTIONS = [
    ("outliers", "False positives",
     "Does this charter belong with the scribe it is recorded under?"),
    ("attributions", "False negatives",
     "Does this unattributed charter belong with a known scribe?"),
    ("new_hands", "New hands",
     "Do these unattributed charters form a scribe who is not yet named?"),
]


def _short(hand: str) -> str:
    """Display form of a namespaced hand: drop the archive when it is obvious."""
    return hand.split("/", 1)[1] if "/" in hand else hand


def _confidence_sentence(p: float | None, cal: dict) -> str:
    """Turn a calibrated probability into a sentence with real counts behind it."""
    if p is None:
        return "No confidence estimate is available for this collection."
    scores = cal.get("scores") or []
    correct = cal.get("correct") or []
    band = [c for s, c in zip(scores, correct) if abs(_safe(s) - _safe(s)) < 1e9]
    n = len(band)
    pct = int(round(p * 100))
    if n:
        return (f"About {pct} out of 100 suggestions this confident turned out to be "
                f"correct, judged on the {n} charters whose scribe is already known.")
    return f"Roughly {pct}% confident."


def _safe(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, 1e-12)


def _class_neighbors(Xn: np.ndarray, query: int, members: list[int],
                     doc_ids: list | np.ndarray,
                     cap: int = CLASS_NEIGHBOR_CAP
                     ) -> tuple[list[int], list[int], list[float], list[float]]:
    """Other pages of the hand, closest-to-query and furthest-to-query.

    Sibling scans of the query are not comparison material (they are the same
    charter). Returns indices plus the cosine *similarity* of each to the query.
    """
    qdoc = str(doc_ids[query])
    others = [j for j in members if j != query and str(doc_ids[j]) != qdoc]
    if not others:
        others = [j for j in members if j != query]
    if not others:
        return [], [], [], []
    sims = np.asarray(Xn[query] @ Xn[np.asarray(others)].T, dtype=np.float64)
    order = np.argsort(-sims)
    ranked = [int(others[i]) for i in order]
    ranked_sims = [float(sims[i]) for i in order]
    close, far = ranked[:cap], list(reversed(ranked))[:cap]
    csim, fsim = ranked_sims[:cap], list(reversed(ranked_sims))[:cap]
    return close, far, csim, fsim


def _neighbor_pack(Xn, query: int, member_idx: list[int], doc_ids) -> dict:
    """Closest/furthest pages of a hand, with cosine similarity and distance."""
    closest, furthest, csim, fsim = [], [], [], []
    if Xn is not None and doc_ids is not None and member_idx:
        closest, furthest, csim, fsim = _class_neighbors(
            Xn, query, [int(j) for j in member_idx], doc_ids)
    if not closest:
        closest = [int(j) for j in member_idx if int(j) != query][:CLASS_NEIGHBOR_CAP]
        furthest = list(reversed(closest))
        if Xn is not None and closest:
            q = Xn[query]
            csim = [float(q @ Xn[j]) for j in closest]
            fsim = list(reversed(csim))
        else:
            csim = fsim = [0.0] * len(closest)
    closest = [int(j) for j in closest]
    furthest = [int(j) for j in furthest]
    return {
        "closest": closest,
        "furthest": furthest,
        "closest_cos": [float(x) for x in csim],
        "furthest_cos": [float(x) for x in fsim],
        "closest_dist": [float(1.0 - x) for x in csim],
        "furthest_dist": [float(1.0 - x) for x in fsim],
        "n_other": len(closest),
    }


def _rows_for(kind: str, items: list[dict], members: dict[str, list[int]],
              cal: dict, name_of: list[str], *, Xn=None, doc_ids=None) -> list[dict]:
    """One UI row per suggestion: what to say, what to light up, what to show."""
    rows = []
    for n, it in enumerate(items):
        r = {"kind": kind, "id": f"{kind}-{n}", "numbers": ""}
        if kind == "outliers":
            hand = it["hand"]
            short = _short(hand)
            pack = _neighbor_pack(Xn, int(it["row"]), members.get(hand, []), doc_ids)
            closer = it.get("closer_hand")
            closer_score = it.get("closer_score")
            hand_cos = float(it["own_score"])
            text = (f"Compare the left page with other pages already recorded as {short}. "
                    f"Keep if it is the same scribe. Reject if that recorded name should "
                    f"not stand (the charter is then unattributed; nothing is reassigned). "
                    f"Not sure if you cannot decide.")
            qrow = int(it["row"])
            r.update(title=escape(it["document"]),
                     text=text,
                     ask=f"Does this charter belong with hand {short}?",
                     keep_hint="Same scribe · 1",
                     reject_hint="Unattribute · 2",
                     query_kicker="Charter in question",
                     hand_role="recorded",
                     csv_kind="false_positive",
                     focus=[qrow], docs=[qrow, *pack["closest"], *pack["furthest"]],
                     document=it["document"], hand=short,
                     n_class=int(len(members.get(hand, []))),
                     hand_cos=hand_cos, hand_dist=float(1.0 - hand_cos),
                     z=float(it["z"]),
                     closer_hand=_short(closer) if closer else None,
                     closer_dist=(None if closer_score is None
                                  else float(1.0 - float(closer_score))),
                     calibrated_p=None, runner_up=None, runner_dist=None,
                     numbers=f"z {it['z']:.2f}, own {it['own_score']:.3f}"
                             + (f", closer {_short(closer)} gap {it['gap']:.3f}"
                                if closer else ""),
                     **pack)
        elif kind == "attributions":
            hand = it["hand"]
            short = _short(hand)
            pack = _neighbor_pack(Xn, int(it["row"]), members.get(hand, []), doc_ids)
            hand_cos = float(it["score"])
            p = it.get("calibrated_p")
            conf = _confidence_sentence(p, cal)
            text = (f"This charter has no recorded scribe. Compare it with pages of "
                    f"hand {short}. Keep to attribute it to {short}. Reject to leave "
                    f"it unattributed (nothing else is assigned). Not sure if you "
                    f"cannot decide. {conf}")
            qrow = int(it["row"])
            runner = it.get("runner_up")
            runner_score = it.get("runner_up_score")
            r.update(title=f"{it['document']} → {escape(short)}",
                     text=text,
                     ask=f"Does this unattributed charter belong with hand {short}?",
                     keep_hint="Attribute to this hand · 1",
                     reject_hint="Leave unattributed · 2",
                     query_kicker="Unattributed charter",
                     hand_role="proposed",
                     csv_kind="false_negative",
                     focus=[qrow], docs=[qrow, *pack["closest"], *pack["furthest"]],
                     document=it["document"], hand=short,
                     n_class=int(len(members.get(hand, []))),
                     hand_cos=hand_cos, hand_dist=float(1.0 - hand_cos),
                     z=float(it.get("join_z") or 0.0),
                     closer_hand=None, closer_dist=None,
                     calibrated_p=(None if p is None else float(p)),
                     runner_up=_short(runner) if runner else None,
                     runner_dist=(None if runner_score is None
                                  else float(1.0 - float(runner_score))),
                     numbers=f"score {it['score']:.3f}, margin "
                             f"{(it['margin'] or 0):.3f}, {it['n_support']} charters "
                             f"under this hand",
                     **pack)
        elif kind == "merges":
            a, b = it["hand_a"], it["hand_b"]
            r.update(title=f"<b>{escape(_short(a))}</b> and <b>{escape(_short(b))}</b>",
                     text=f"Could these be one scribe? Their charters "
                          f"({it['n_a']} and {it['n_b']}) are about as alike as each "
                          f"name is to itself.",
                     focus=members.get(a, [])[:2] + members.get(b, [])[:2],
                     docs=members.get(a, []) + members.get(b, []),
                     numbers=f"between {it['cross_similarity']:.3f} vs within "
                             f"{it['own_similarity']:.3f} (closeness {it['closeness']:+.3f})")
        elif kind == "splits":
            pct = it["percentile"]
            r.update(title=f"<b>{escape(_short(it['hand']))}</b> — {it['n_docs']} charters "
                           f"in two groups",
                     text=f"Could this be two scribes under one name? The division is "
                          f"sharper than {pct:.0f}% of random divisions of the same "
                          f"charters.",
                     focus=it["rows_a"][:2] + it["rows_b"][:2],
                     docs=it["rows_a"] + it["rows_b"],
                     groups=[it["rows_a"], it["rows_b"]],
                     numbers=f"separation {it['separation']:.3f}, percentile {pct:.0f}")
        elif kind == "new_hands":
            # The cluster is a grid of pages with a checkbox each, ordered by
            # distance to the medoid, so EVERY member ships (with the pairwise
            # cosines, so the sheet can show each page's distance to the medoid).
            cluster_rows = [int(i) for i in it["rows"]]
            qrow = int(it.get("row", cluster_rows[0]))
            pack = _neighbor_pack(Xn, qrow, cluster_rows, doc_ids)
            if Xn is not None:
                order = sorted(cluster_rows,
                               key=lambda j: -float(Xn[qrow] @ Xn[j]))
                sub = Xn[order] @ Xn[order].T          # indexed like ``members``
                member_sim = [[float(v) for v in rowv] for rowv in sub]
            else:
                order = list(cluster_rows)
                member_sim = [[1.0 if a == b else 0.0 for b in order]
                              for a in order]
            coh = float(it["cohesion"])
            lvl = it.get("level") or "FINCH"
            ari = it.get("ari")
            text = (f"None of these {it['n_docs']} charters has a recorded scribe, and "
                    f"the group does not overlap any named hand. Untick any page that "
                    f"is not by the same scribe as the rest, then Confirm. Deselect all "
                    f"and Confirm (or Reject) if this is not a hand at all. Click a page "
                    f"to inspect it in the zoomable pane on the right.")
            r.update(title=escape(it["document"]),
                     text=text,
                     ask=f"Which of these {it['n_docs']} unattributed charters are "
                         f"one unnamed scribe?",
                     keep_hint="Ticked = same scribe · 1",
                     reject_hint="Not a hand: none of them · 2",
                     query_kicker="Possible unnamed hand",
                     hand_role="cluster",
                     csv_kind="new_hand",
                     focus=[qrow], docs=list(order),
                     members=list(order), member_sim=member_sim,
                     reference=qrow, group=f"new_hand_{n + 1}",
                     document=it["document"], hand=f"new_hand_{n + 1}",
                     n_class=int(it["n_docs"]),
                     hand_cos=coh, hand_dist=float(1.0 - coh),
                     z=0.0, closer_hand=None, closer_dist=None,
                     calibrated_p=None, runner_up=None, runner_dist=None,
                     numbers=(f"{it['n_docs']} pages, cohesion {coh:.3f} vs typical "
                              f"{it['reference_cohesion']:.3f}, {lvl}"
                              + (f" ARI {ari:.3f}" if ari is not None else "")),
                     **pack)
        elif kind == "duplicates":
            r.update(title=f"{it['document_a']} ≈ {it['document_b']}",
                     text="These two images are nearly identical — probably the same "
                          "charter photographed twice.",
                     focus=[it["row_a"], it["row_b"]],
                     docs=[it["row_a"], it["row_b"]],
                     numbers=f"similarity {it['similarity']:.4f}")
        elif kind == "isolated":
            r.update(title=it["document"],
                     text="Nothing in the collection resembles this.",
                     focus=[it["row"]], docs=[it["row"]],
                     numbers=f"best match {it['best_match']:.3f}")
        rows.append(r)
    return rows


def _nearest_neighbors(X: np.ndarray, k: int = 5) -> list[list[int]]:
    """Top-``k`` cosine neighbours per document (self excluded), as index lists.

    Cosine because retrieval is scored cosine (see ``mole eval``): the panel then
    shows exactly the pages the metric considers closest, so a click doubles as a
    read-out of what the model thinks looks alike.
    """
    if k <= 0 or X.shape[0] < 2:
        return [[] for _ in range(X.shape[0])]
    Xf = np.asarray(X, dtype=np.float32)
    norm = np.linalg.norm(Xf, axis=1, keepdims=True)
    Xn = Xf / np.clip(norm, 1e-12, None)
    out: list[list[int]] = []
    kk = min(k, Xf.shape[0] - 1)
    # block the matrix to keep peak memory sane on large corpora
    for lo in range(0, Xn.shape[0], 512):
        hi = min(lo + 512, Xn.shape[0])
        sims = Xn[lo:hi] @ Xn.T
        for r, gi in enumerate(range(lo, hi)):
            sims[r, gi] = -np.inf
            top = np.argpartition(sims[r], -kk)[-kk:]
            out.append([int(j) for j in top[np.argsort(sims[r, top])[::-1]]])
    return out


def _svg(coords: np.ndarray, first_colors: list[str], base_cats: list[str],
         names: list[str], size: int = 620, highlight_idx=None,
         highlight_labels: bool = True) -> str:
    """The map, with unlabeled documents crossed through as in ``mole viz``.

    The cross is a property of the DOCUMENT, not of the active colouring, so it is
    fixed to the ground-truth (hand) scheme and stays put while fills change —
    under a cluster scheme an unlabeled point is still coloured by its cluster,
    which is exactly the attribution question ("which cluster did it join?").
    """
    from mole.viz.scatter import _HIGHLIGHT_STROKE, _UNLABELED_CROSS, _is_unlabeled

    xs, ys = coords[:, 0].astype(float), coords[:, 1].astype(float)

    def norm(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo or 1.0)

    pad = 18
    nx = norm(xs) * (size - 2 * pad) + pad
    ny = (1.0 - norm(ys)) * (size - 2 * pad) + pad
    hi = set(highlight_idx or [])
    out = []
    for i, (x, y) in enumerate(zip(nx, ny)):
        dot = (f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="3.6" '
               f'fill="{first_colors[i]}" data-i="{i}">'
               f'<title>{escape(names[i])}</title></circle>')
        if _is_unlabeled(base_cats[i]):
            a = 2.0
            dot = (f'<g data-unl="1">{dot}'
                   f'<path d="M{x - a:.1f} {y - a:.1f}L{x + a:.1f} {y + a:.1f}'
                   f'M{x - a:.1f} {y + a:.1f}L{x + a:.1f} {y - a:.1f}" '
                   f'stroke="{_UNLABELED_CROSS}" stroke-width="1" '
                   f'stroke-linecap="round" pointer-events="none"/></g>')
        out.append(dot)
    for i in hi:
        x, y = nx[i], ny[i]
        out.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7.5" fill="none" '
            f'stroke="{_HIGHLIGHT_STROKE}" stroke-width="2" pointer-events="none"/>')
        if highlight_labels:
            out.append(
                f'<text x="{x + 9:.1f}" y="{y:.1f}" dominant-baseline="central" '
                f'fill="{_HIGHLIGHT_STROKE}" font-size="10" font-weight="700" '
                f'pointer-events="none">{escape(names[i])}</text>')
    return (f'<svg id="map" viewBox="0 0 {size} {size}" width="{size}" '
            f'height="{size}">{"".join(out)}</svg>')


def _schemes(report, hands: list[str], paths, clusters) -> list[tuple[str, list[str]]]:
    """Colour schemes offered in the picker: hand, dataset, then FINCH levels."""
    out: list[tuple[str, list[str]]] = [
        ("hand", [_short(h) if h else "unlabeled" for h in hands])]
    datasets = [p.parent.name or "root" for p in paths]
    if len(set(datasets)) > 1:
        out.append(("dataset", datasets))
    # One scheme per FINCH level, so the hierarchy can be walked from fine to
    # coarse against the recorded hands. Levels with a single cluster are dropped
    # (they colour everything identically and say nothing), and the level with the
    # best silhouette is marked — a principled default rather than a guess.
    levels = [lv for lv in getattr(report, "cluster_levels", [])
              if lv["n_clusters"] > 1 and len(lv["labels"]) == len(hands)]
    # compare like with like: FINCH's silhouette counts every document, HDBSCAN's
    # excludes noise, so a method that discards more looks better for free. The
    # star therefore ranks WITHIN a method, never across them.
    def _family(name):
        return name.split()[0]

    # Prefer AGREEMENT WITH THE RECORDED HANDS when there is any: a partition that
    # recovers what the archivist established beats one that is merely tidy.
    # Silhouette remains the fallback where nothing is labeled.
    key = "ari" if any(lv.get("ari") is not None for lv in levels) else "silhouette"
    best_by: dict[str, float] = {}
    for lv in levels:
        if lv.get(key) is None:
            continue
        fam = _family(lv["level"])
        best_by[fam] = max(best_by.get(fam, -2.0), lv[key])
    # exactly one star per family: ties go to the FINER partition, which is the
    # one that can still represent a two-document hand
    starred: set[str] = set()
    for lv in levels:
        tag = f"{lv['level']} · {lv['n_clusters']} clusters"
        if lv.get("n_noise"):
            tag += f" · {lv['n_noise']} unclustered"
        if lv.get("ari") is not None:
            tag += f" · agreement {lv['ari']:.3f}"
        elif lv["silhouette"] is not None:
            tag += f" · silhouette {lv['silhouette']:.3f}"
        val, fam = lv.get(key), _family(lv["level"])
        if val is not None and val == best_by.get(fam) and fam not in starred:
            starred.add(fam)
            tag += " ★"
        # HDBSCAN's noise label is -1, which viz/scatter already treats as
        # "no ground truth": those points stay neutral grey instead of being
        # coloured as if they were a discovered hand.
        out.append((tag, ["-1" if v == -1 else f"c{v}" for v in lv["labels"]]))
    return out


def _picker(schemes, scheme_data, hands) -> str:
    """Scheme dropdown + the show/hide-unlabeled toggle (both from `mole viz`)."""
    from mole.viz.scatter import _is_unlabeled

    n_unl = sum(1 for h in hands if _is_unlabeled(h or "unlabeled"))
    bits = []
    if len(schemes) > 1:
        # Count only REAL categories: "unlabeled" (and HDBSCAN's noise) is the
        # absence of one, and counting it made the dropdown say 14 where the title
        # said 13 scribes. Cluster schemes already carry their own counts.
        opts = []
        for name, cats in schemes:
            label = escape(name)
            if "·" not in name:
                n_real = len({c for c in cats if not _is_unlabeled(c)})
                label += f" ({n_real})"
            opts.append(f'<option value="{escape(name, quote=True)}">{label}</option>')
        bits.append('<label>colour by <select id="scheme">'
                    + "".join(opts) + "</select></label>")
    if n_unl:
        bits.append(f'<label><input type="checkbox" id="unl" checked> '
                    f'show unattributed <b>{n_unl}</b></label>')
    return "".join(bits)


def _render_cases(embeddings: Path, *, out, clusters, limit, max_mb, image_cache,
                  image_url, images, cluster_method, seed,
                  false_positives: bool = True) -> tuple[Path, str]:
    """Case-by-case review sheet: no map, query vs recorded-hand exemplars.

    ``false_positives=False`` drops that tab (a recorded identification is
    rarely wrong in these archives, so some reviewers skip it).
    """
    from mole.review.images import ImageBudget
    from mole.review.suggest import build_review, document_table

    report = build_review(embeddings, clusters=clusters, limit=limit, seed=seed,
                          cluster_method=cluster_method, lists=True)
    X, meta, rows_meta, names, paths, hands, docs = document_table(embeddings)
    Xn = _l2(X)
    members: dict[str, list[int]] = {}
    for i, h in enumerate(hands):
        if h:
            members.setdefault(h, []).append(i)

    sections = []
    for kind, heading, blurb in _SECTIONS:
        if kind == "outliers" and not false_positives:
            continue
        items = getattr(report, kind, [])[:limit]
        rows = _rows_for(kind, items, members, report.calibration, names,
                         Xn=Xn, doc_ids=docs)
        sections.append((kind, heading, blurb, rows))

    room = int(max_mb * 1024 * 1024) if max_mb else 0
    budget = ImageBudget(room, cache_dir=image_cache)
    if images:
        wanted: list[int] = []
        for _k, _h, _b, rws in sections:
            for r in rws:
                wanted.append(r["focus"][0] if r.get("focus") else r.get("docs", [None])[0])
                wanted.extend(r.get("members") or [])     # new hands: every page
                wanted.extend(r.get("closest") or [])
                wanted.extend(r.get("furthest") or [])
        seen = set()
        for i in wanted:
            if i is None or i in seen:
                continue
            seen.add(i)
            budget.add(str(i), paths[i])

    payload = {
        "mode": "review",
        "dims": {k: list(v) for k, v in budget.dims.items()},
        "sections": [{"kind": k, "heading": h, "blurb": b, "rows": r}
                     for k, h, b, r in sections],
        "images": budget.uris,
        "names": names,
        "hands": [_short(h) for h in hands],
        "urls": ([image_url.replace("{filename}", n) for n in names] if image_url
                 else [p.resolve().as_uri() if p.is_file() else "" for p in paths]),
        "n_documents": report.n_documents,
        "n_labeled": report.n_labeled,
        "n_hands": report.n_hands,
    }
    title = escape(", ".join(report.datasets) or "archive")
    html = _CASE_HTML.replace("__TITLE__", title) \
                     .replace("__ZOOM_CSS__", _ZOOM_CSS) \
                     .replace("__ZOOM_JS__", _ZOOM_JS) \
                     .replace("__PAYLOAD__", json.dumps(payload))
    out_path = Path(out) if out else embeddings.with_suffix(".review.html")
    out_path.write_text(html, encoding="utf-8")
    mb = out_path.stat().st_size / (1024 * 1024)
    return out_path, f"{budget.summary()} · {mb:.1f} MB total"


def render_review(embeddings: str | Path, *, out: str | Path | None = None,
                  clusters: str | Path | None = None, limit: int = DEFAULT_LIMIT,
                  max_mb: float = DEFAULT_MAX_MB, image_cache: str | Path | None = None,
                  image_url: str | None = None, images: bool = True,
                  image_scope: str = "listed", map_backend: str = "auto",
                  mode: str = "review", cluster_method: str = "both",
                  method: str = "auto", seed: int = 0,
                  highlight: list[str] | None = None,
                  highlight_file: str | Path | None = None,
                  point_size: float = 9.0, pca_whiten: bool = True,
                  pca_dim: int = 150,
                  umap_neighbors: int = 15, umap_min_dist: float = 0.1,
                  theme: str = "dark", show_labels: bool = False,
                  neighbors: int = 5, false_positives: bool = True,
                  highlight_labels: bool = True,
                  neighbor_lines: bool = False) -> tuple[Path, str]:
    """Build the review sheet or the viz map. Returns ``(path, summary_line)``.

    ``mode="viz"`` is the map + charter viewer only (``mole viz``). ``mode="review"``
    is a separate case-by-case file with no map (``mole review``). ``theme`` is ``dark``
    (review) or ``light`` (publication figure), toggleable live. ``show_labels``
    prints the active category id in each circle. ``neighbors`` is how many nearest
    charters to list under the viewer when a document is selected.
    ``false_positives=False`` drops the false-positive tab from the review.
    ``highlight_labels=False`` rings the highlighted charters without printing
    their names on the map. ``neighbor_lines`` sets the initial state of the
    "neighbour lines" toggle (connectors from a selected charter to its nearest
    neighbours); off by default, the reader can switch it on in the page.
    """
    from mole.review.images import ImageBudget
    from mole.review.suggest import build_review, document_table
    from mole.viz.scatter import _is_highlighted, _parse_highlights, reduce_2d

    mode = "viz" if str(mode).lower() == "viz" else "review"
    embeddings = Path(embeddings)
    if mode == "review":
        return _render_cases(
            embeddings, out=out, clusters=clusters, limit=limit, max_mb=max_mb,
            image_cache=image_cache, image_url=image_url, images=images,
            cluster_method=cluster_method, seed=seed, false_positives=false_positives)

    X, meta, rows_meta, names, paths, hands, _docs = document_table(embeddings)
    coords, used_method = reduce_2d(X, method, seed, pca_dim=pca_dim,
                                    pca_whiten=pca_whiten,
                                    umap_neighbors=umap_neighbors, umap_min_dist=umap_min_dist)
    report = build_review(
        embeddings, clusters=clusters, limit=limit, lists=False,
        cluster_method=cluster_method)
    schemes = _schemes(report, hands, paths, clusters)

    hl = _parse_highlights(highlight, highlight_file)
    highlight_idx = ([i for i, p in enumerate(paths) if _is_highlighted(str(p), hl)]
                     if hl else [])

    nn = _nearest_neighbors(X, k=neighbors)
    theme = "light" if str(theme).lower() == "light" else "dark"

    members: dict[str, list[int]] = {}
    for i, h in enumerate(hands):
        if h:
            members.setdefault(h, []).append(i)

    sections = []
    if mode == "review":
        for kind, heading, blurb in _SECTIONS:
            items = getattr(report, kind, [])[:limit]
            sections.append((kind, heading, blurb,
                             _rows_for(kind, items, members, report.calibration, names)))

    # images, most-important-first, until the budget is spent
    from mole.review import bokeh_map

    use_bokeh = (map_backend == "bokeh"
                 or (map_backend == "auto" and bokeh_map.available()))
    if map_backend == "bokeh" and not bokeh_map.available():
        raise RuntimeError("--map bokeh needs bokeh: pip install 'mole[viz]'")

    # BokehJS is inlined, so it competes with the charters for the size cap.
    # Charging it to the budget is what keeps --max-mb honest.
    overhead = bokeh_map.bokehjs_bytes() if use_bokeh else 0
    room = int(max_mb * 1024 * 1024) - overhead if max_mb else 0
    if max_mb and room < 0:
        raise RuntimeError(
            f"--max-mb {max_mb} cannot hold BokehJS alone ({overhead / 1e6:.1f} MB); "
            f"raise it or pass --map svg")
    budget = ImageBudget(room, cache_dir=image_cache)
    if images:
        # The UI shows at most 4 images per row, so only the FOCUS documents plus a
        # couple of supporting ones are ever displayed. Enqueuing every member of a
        # hand (Antwerp's hand R alone has 217) would encode hundreds of pages that
        # nothing can show.
        wanted: list[int] = []
        # ring-highlighted charters are the reason the map was built: they get
        # a page before anything else, so they open under any --max-mb
        wanted.extend(highlight_idx)
        for _kind, _h, _b, rws in sections:
            for r in rws:
                wanted.extend(r.get("focus", [])[:4])
        for _kind, _h, _b, rws in sections:
            for r in rws:
                wanted.extend(r.get("docs", [])[:4])
        if image_scope == "all" or mode == "viz":
            # viz clicks arbitrary dots, so every document needs a page —
            # still budget-capped, and still listed-documents-first.
            wanted.extend(range(len(names)))
        seen = set()
        for i in wanted:
            if i in seen:
                continue
            seen.add(i)
            budget.add(str(i), paths[i])

    from mole.viz.scatter import _scheme_payload

    scheme_data = {n: _scheme_payload(c) for n, c in schemes}
    first = scheme_data[schemes[0][0]]
    payload = {
        "mode": mode,
        "dims": {k: list(v) for k, v in budget.dims.items()},
        "schemes": {n: {"colors": p["colors"], "cats": p["cats"],
                        "legend": p["legend"]} for n, p in scheme_data.items()},
        "first": schemes[0][0],
        "sections": [{"kind": k, "heading": h, "blurb": b, "rows": r}
                     for k, h, b, r in sections],
        "images": budget.uris,
        "names": names,
        "hands": [_short(h) for h in hands],
        "nn": nn,
        "urls": ([image_url.replace("{filename}", n) for n in names] if image_url
                 else [p.resolve().as_uri() if p.is_file() else "" for p in paths]),
    }
    subtitle = (f"{report.n_documents} charters · {report.n_labeled} with a recorded "
                f"scribe · {report.n_hands} scribes · map: {used_method}")

    if use_bokeh:
        bk_script, map_div, bk_css, bk_js = bokeh_map.build(
            coords, names, [_short(h) for h in hands], first["colors"],
            highlight_idx=highlight_idx, point_size=point_size,
            show_labels=show_labels, label_cats=schemes[0][1], theme=theme,
            highlight_labels=highlight_labels)
        glue = bokeh_map.glue_js()
        viewer_html = _ZOOM_VIEWER
    else:
        bk_script = map_div = bk_css = bk_js = ""
        map_div = _svg(coords, first["colors"], schemes[0][1], names,
                       highlight_idx=highlight_idx, highlight_labels=highlight_labels)
        glue = _svg_glue_js()
        viewer_html = _ZOOM_VIEWER
    kind_title = "Embeddings" if mode == "viz" else "Hand review"
    body_class = f"{theme}{'' if mode == 'review' else ' viz'}"
    html = _HTML.replace("__BODYCLASS__", body_class) \
                .replace("__KIND__", kind_title) \
                .replace("__THEMECHK__", " checked" if theme == "light" else "") \
                .replace("__LABELCHK__", " checked" if show_labels else "") \
                .replace("__NNCHK__", " checked" if neighbor_lines else "") \
                .replace("__PSIZE__", f"{float(point_size):.1f}") \
                .replace("__TITLE__", escape(", ".join(report.datasets) or "archive")) \
                .replace("__SUBTITLE__", subtitle) \
                .replace("__BOKEH_CSS__", bk_css) \
                .replace("__MAP__", map_div) \
                .replace("__VIEWER__", viewer_html) \
                .replace("__PICKER__", _picker(schemes, scheme_data, hands)) \
                .replace("__LEGEND__", first["legend"]) \
                .replace("__PAYLOAD__", json.dumps(payload)) \
                .replace("__BOKEH_JS__", bk_js) \
                .replace("__BOKEH_SCRIPT__", bk_script) \
                .replace("__MOLE_JS__", glue) \
                .replace("__ZOOM_CSS__", _ZOOM_CSS) \
                .replace("__ZOOM_JS__", _ZOOM_JS) \
                .replace("__REVIEW_CHROME__", _REVIEW_CHROME if mode == "review" else "") \
                .replace("__REVIEW_JS__", _REVIEW_JS if mode == "review" else "")

    out_path = Path(out) if out else embeddings.with_suffix(
        ".viz.html" if mode == "viz" else ".review.html")
    out_path.write_text(html, encoding="utf-8")
    mb = out_path.stat().st_size / (1024 * 1024)
    return out_path, f"{budget.summary()} · {mb:.1f} MB total"




def _svg_glue_js() -> str:
    """`window.MOLE` over the inline SVG — same three calls the page makes of Bokeh.

    Keeping one interface means the review panel, the colour picker and the
    inspector are written once and neither backend is privileged.
    """
    return r"""
window.MOLE = (function(){
  var svg = document.getElementById('map');
  var dots = svg ? svg.querySelectorAll('.dot') : [];
  var tapcb = null;
  if(svg) svg.addEventListener('click', function(e){
    var c = e.target.closest ? e.target.closest('circle') : null;
    if(c && tapcb) tapcb(+c.getAttribute('data-i'));
  });
  return {
    setColors: function(cols){
      for(var i=0;i<dots.length;i++)
        dots[i].setAttribute('fill', cols[+dots[i].getAttribute('data-i')]);
    },
    setAlphas: function(alphas){
      for(var i=0;i<dots.length;i++){
        var a = alphas[+dots[i].getAttribute('data-i')];
        dots[i].setAttribute('fill-opacity', a);
        dots[i].style.display = (a === 0) ? 'none' : '';   // toggle = truly gone
      }
    },
    showImage: function(uri, w, h){
      var img = document.getElementById('pageimg');
      if(!img) return;
      img.style.display = uri ? '' : 'none';
      if(uri) img.src = uri;
      if(window.moleResetZoom) window.moleResetZoom(img.closest('.zoombox'));
    },
    onTap: function(cb){ tapcb = cb; },
    select: function(i){
      for(var k=0;k<dots.length;k++)
        dots[k].classList.toggle('sel', +dots[k].getAttribute('data-i') === i);
    },
    // the lightweight SVG map has no in-circle labels / size / theme controls;
    // keep the interface identical so the shared page JS runs unchanged
    setLabels: function(){},
    showLabels: function(){},
    setSize: function(px){
      for(var i=0;i<dots.length;i++) dots[i].setAttribute('r', Math.max(1.5, px/2.4));
    },
    setTheme: function(dark){
      if(svg) svg.style.background = dark ? '#12131a' : '#ffffff';
    },
    markNeighbors: function(){},
    clearNeighbors: function(){},
    // the SVG backend draws no category hull; keep the interface identical so the
    // shared page script's legend-tap handler runs without erroring
    showHull: function(){},
    clearHull: function(){},
    // no pan/zoom or size-flash on the static SVG map; showDoc still selects the dot
    centerOn: function(){},
    flash: function(){},
    // pop a clicked hand's dots: a brief CSS-eased grow, then back to their radius
    pulse: function(idxs){
      if(!idxs) return;
      idxs.forEach(function(i){
        var el = svg.querySelector('.dot[data-i="'+i+'"]');
        if(!el) return;
        var r0 = el.getAttribute('r');
        el.style.transition = 'r .18s ease';
        el.setAttribute('r', (parseFloat(r0) || 3.6) * 2.4);
        setTimeout(function(){ el.setAttribute('r', r0); }, 260);
      });
    }
  };
})();
"""


_ZOOM_VIEWER = (
    '<div class="zoombox" id="pagezoom" '
    'title="Scroll to zoom, drag to pan, double-click to reset">'
    '<div class="zoomstage"><img id="pageimg" alt=""></div></div>'
)

# Shared charter-viewer zoom: wheel to zoom on the cursor, drag to pan,
# double-click to fit. Used by the review panes and the viz inspector.
_ZOOM_CSS = r"""
.zoombox{overflow:hidden;position:relative;touch-action:none;cursor:zoom-in;
  background:#0b0b0d}
.zoombox .zoomstage{transform-origin:0 0;width:100%;height:100%;
  display:flex;align-items:center;justify-content:center;will-change:transform}
.zoombox img{max-width:100%;max-height:100%;width:auto;height:auto;
  object-fit:contain;display:block;user-select:none;-webkit-user-drag:none}
.zoombox.zoomed{cursor:grab}
.zoombox.dragging{cursor:grabbing}
"""

_ZOOM_JS = r"""
window.moleResetZoom = function(box){
  if(!box || !box._mz) return;
  box._mz.s = 1; box._mz.x = 0; box._mz.y = 0;
  box._mz.apply();
};
function bindZoom(box){
  if(!box || box._mz) return;
  var stage = box.querySelector('.zoomstage') || box;
  var st = {s:1, x:0, y:0, drag:false, lx:0, ly:0};
  st.apply = function(){
    stage.style.transform = 'translate('+st.x+'px,'+st.y+'px) scale('+st.s+')';
    box.classList.toggle('zoomed', st.s > 1.001);
    if(!st.drag) box.classList.remove('dragging');
  };
  box._mz = st;
  box.addEventListener('wheel', function(e){
    e.preventDefault();
    var r = box.getBoundingClientRect();
    var mx = e.clientX - r.left, my = e.clientY - r.top;
    var old = st.s;
    var next = old * (e.deltaY < 0 ? 1.12 : 1/1.12);
    st.s = Math.min(16, Math.max(1, next));
    st.x = mx - (mx - st.x) * (st.s / old);
    st.y = my - (my - st.y) * (st.s / old);
    if(st.s === 1){ st.x = 0; st.y = 0; }
    st.apply();
  }, {passive:false});
  box.addEventListener('pointerdown', function(e){
    if(e.button !== 0 || st.s <= 1) return;
    st.drag = true; st.lx = e.clientX; st.ly = e.clientY;
    box.classList.add('dragging');
    try{ box.setPointerCapture(e.pointerId); }catch(err){}
  });
  box.addEventListener('pointermove', function(e){
    if(!st.drag) return;
    st.x += e.clientX - st.lx; st.y += e.clientY - st.ly;
    st.lx = e.clientX; st.ly = e.clientY;
    st.apply();
  });
  function endDrag(){ st.drag = false; box.classList.remove('dragging'); }
  box.addEventListener('pointerup', endDrag);
  box.addEventListener('pointercancel', endDrag);
  box.addEventListener('dblclick', function(e){
    e.preventDefault();
    window.moleResetZoom(box);
  });
  var img = box.querySelector('img');
  if(img) img.addEventListener('load', function(){ window.moleResetZoom(box); });
}
document.querySelectorAll('.zoombox').forEach(bindZoom);
"""

_CASE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hand review — __TITLE__</title>
<style>
:root{--bg:#121214;--panel:#1a1a1e;--elev:#24242a;--line:#32323a;--fg:#f0f0f4;
  --dim:#9a9aa4;--accent:#7eb0ff;--accent-weak:rgba(126,176,255,.16);
  --keep:#2f9e6a;--keep-d:#247a52;--reject:#e0554b;--reject-d:#b83d35}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
  background:var(--bg);color:var(--fg);display:flex;flex-direction:column;overflow:hidden}
header{flex:0 0 auto;padding:12px 20px 10px;border-bottom:1px solid var(--line);background:var(--panel)}
.top{display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap}
h1{font-size:15px;font-weight:650;margin:0;letter-spacing:-.02em;color:var(--dim)}
.progress{margin:0;font-variant-numeric:tabular-nums;color:var(--fg);font-weight:650}
.work{display:flex;flex-direction:column;gap:6px;min-width:min(420px,100%)}
.workbar{height:10px;border-radius:99px;background:#2a2a32;overflow:hidden;border:1px solid var(--line)}
.workfill{display:block;height:100%;width:0;background:var(--accent);border-radius:99px;
  transition:width .18s ease}
.caseslider{width:100%;accent-color:var(--accent);cursor:pointer;margin:0;height:18px}
.ask{font-size:22px;font-weight:700;margin:8px 0 4px;letter-spacing:-.03em}
.how{color:var(--dim);margin:0 0 8px;max-width:78ch}
.nums{margin:0 0 10px;font-variant-numeric:tabular-nums;font-size:13px;color:var(--fg);
  display:flex;flex-wrap:wrap;gap:6px 14px}
.nums b{font-weight:700}
.nums .dim{color:var(--dim)}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
.tabs button{background:var(--elev);color:var(--fg);border:1px solid var(--line);
  border-radius:8px;padding:5px 11px;font:inherit;cursor:pointer}
.tabs button.on{background:var(--accent-weak);border-color:var(--accent)}
.tabs button:disabled{opacity:.4;cursor:not-allowed}
main{flex:1 1 auto;min-height:0;display:flex}
.pane{flex:1 1 50%;min-width:0;display:flex;flex-direction:column;padding:12px 16px 8px}
.pane + .pane{border-left:1px solid var(--line)}
.kicker{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  color:var(--dim);margin:0 0 4px}
.pane h2{font-size:16px;font-weight:700;margin:0 0 6px;letter-spacing:-.02em}
.pane .meta{font-size:13px;color:var(--dim);margin:0 0 8px;word-break:break-all}
.tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 8px}
.sort{display:inline-flex;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--elev)}
.sort button{background:transparent;color:var(--fg);border:0;padding:8px 14px;font:inherit;
  cursor:pointer;margin:0}
.sort button + button{border-left:1px solid var(--line)}
.sort button.on{background:var(--accent-weak);font-weight:650}
.flip{display:inline-flex;align-items:center;gap:6px}
.flip button{background:var(--elev);color:var(--fg);border:1px solid var(--line);border-radius:9px;
  padding:8px 14px;font:inherit;font-size:16px;cursor:pointer;min-width:44px}
.flip button:hover,.flip button:focus-visible{border-color:var(--accent)}
.flip .count{color:var(--dim);font-variant-numeric:tabular-nums;min-width:4.8em;text-align:center;font-weight:650}
.dots{display:flex;flex-wrap:wrap;gap:5px;align-items:center}
.dots button{width:10px;height:10px;padding:0;border-radius:50%;border:0;background:#4a4a54;
  cursor:pointer}
.dots button.on{background:var(--accent);transform:scale(1.25)}
.gridpane{flex:1 1 100%;min-width:0;display:flex;flex-direction:column;padding:12px 16px 8px}
.pane[hidden],.gridpane[hidden]{display:none}
.gridpane .tools{justify-content:space-between}
.gridpane h2{font-size:16px;font-weight:700;margin:0;letter-spacing:-.02em}
.gridwrap{flex:1;min-height:0;display:flex;gap:14px}
.grid{flex:1 1 55%;min-width:0;min-height:0;overflow:auto;display:grid;gap:12px;align-content:start;
  grid-template-columns:repeat(auto-fill,minmax(300px,1fr));padding:2px}
.inspect{flex:1 1 45%;min-width:0;min-height:0;display:flex;flex-direction:column;
  border-left:1px solid var(--line);padding-left:14px}
.inspect .page{flex:1;min-height:0}
.card{background:var(--panel);border:2px solid var(--line);border-radius:12px;padding:8px;
  display:flex;flex-direction:column;gap:6px;cursor:pointer}
.card.in{border-color:var(--keep)} .card.out{border-color:var(--reject);opacity:.62}
.card.sel{box-shadow:0 0 0 3px var(--accent);opacity:1}
.card .thumb{height:280px;background:#0b0b0d;border-radius:8px;overflow:hidden;
  display:flex;align-items:center;justify-content:center}
.card .thumb img{max-width:100%;max-height:100%;object-fit:contain;display:block}
.card label{display:flex;align-items:center;gap:8px;font-weight:650;cursor:pointer}
.card label input{width:18px;height:18px;accent-color:var(--keep);cursor:pointer}
.card .cmeta{font-size:12px;color:var(--dim);word-break:break-all;margin:0}
.card .badge{font-size:11px;color:var(--accent);font-weight:700;letter-spacing:.04em;text-transform:uppercase}
.bar .selall{display:inline-flex;gap:6px} .bar .selall[hidden]{display:none}
@media(max-width:1100px){.gridwrap{flex-direction:column}
  .inspect{border-left:none;border-top:1px solid var(--line);padding:12px 0 0;min-height:45vh}}
.page{flex:1;min-height:0;background:#0b0b0d;border:1px solid var(--line);border-radius:12px}
__ZOOM_CSS__
.ph{color:var(--dim);padding:24px;text-align:center}
footer{flex:0 0 auto;background:var(--panel);border-top:1px solid var(--line);
  padding:12px 20px 14px;box-shadow:0 -10px 28px rgba(0,0,0,.28)}
.dec{display:flex;gap:12px;flex-wrap:wrap;align-items:stretch;margin-bottom:10px}
.dec button[data-v]{flex:1 1 160px;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:2px;font:inherit;font-size:20px;font-weight:750;padding:14px 18px;min-height:72px;
  border-radius:14px;border:3px solid transparent;cursor:pointer;letter-spacing:-.02em}
.dec button small{font-size:12px;font-weight:550;opacity:.9;letter-spacing:0}
.dec .keep{background:var(--keep);color:#fff;border-color:var(--keep-d)}
.dec .reject{background:var(--reject);color:#fff;border-color:var(--reject-d)}
.dec .unsure{background:var(--elev);color:var(--fg);border-color:var(--line)}
.dec button.on{box-shadow:0 0 0 3px var(--accent)}
.dec .keep:hover,.dec .reject:hover{filter:brightness(1.07)}
.dec .unsure:hover,.dec .unsure.on{border-color:var(--accent);background:var(--accent-weak)}
.dec input{flex:1 1 180px;background:var(--elev);color:var(--fg);border:1px solid var(--line);
  border-radius:12px;padding:12px 14px;font:inherit}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;color:var(--dim);font-size:13px}
.bar button{background:var(--elev);color:var(--fg);border:1px solid var(--line);border-radius:9px;
  padding:7px 12px;font:inherit;cursor:pointer}
.bar button:hover{border-color:var(--accent)}
#tally b{color:var(--fg)}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
@media(max-width:860px){
  body{overflow:auto} main{flex-direction:column;min-height:70vh}
  .pane + .pane{border-left:none;border-top:1px solid var(--line)}
  .pane{min-height:40vh}
  .dec button[data-v]{flex:1 1 30%;min-width:0;min-height:64px;font-size:17px;padding:10px 8px}
}
</style></head><body>
<header>
  <div class="top">
    <h1>Hand review — __TITLE__</h1>
    <div class="work">
      <div class="workbar" id="workbar" role="progressbar" aria-valuemin="0" aria-valuemax="100"
           aria-valuenow="0" aria-label="Decisions in this tab">
        <span class="workfill" id="workfill"></span>
      </div>
      <p class="progress" id="progress"></p>
      <input type="range" class="caseslider" id="caseslider" min="1" max="1" value="1"
             aria-label="Jump to case">
    </div>
  </div>
  <p class="ask" id="ask">Look at one charter at a time.</p>
  <p class="how" id="how">Left: the charter in question. Right: other pages of the relevant
    hand. Scroll to zoom, drag to pan, double-click to reset. Closest = typical;
    Furthest = the range.</p>
  <p class="nums" id="nums"></p>
  <nav class="tabs" id="tabs"></nav>
</header>
<main>
  <section class="pane" id="querypane">
    <p class="kicker">Charter in question</p>
    <h2 id="qtitle">This page</h2>
    <p class="meta" id="qmeta"></p>
    <div class="page zoombox" title="Scroll to zoom, drag to pan, double-click to reset">
      <div class="zoomstage"><img id="qimg" alt="charter in question"></div>
    </div>
  </section>
  <section class="pane" id="exempane">
    <p class="kicker" id="exkicker">Same recorded hand</p>
    <h2><span id="exrole">Recorded as</span> <span id="exhand">—</span></h2>
    <div class="tools">
      <div class="sort" role="group" aria-label="Which neighbours to show">
        <button type="button" id="sort-closest" class="on">Closest</button>
        <button type="button" id="sort-furthest">Furthest</button>
      </div>
      <div class="flip">
        <button type="button" id="prevEx" title="Previous neighbour (←)">←</button>
        <span class="count" id="excount">—</span>
        <button type="button" id="nextEx" title="Next neighbour (→)">→</button>
      </div>
      <div class="dots" id="exdots"></div>
    </div>
    <p class="meta" id="exmeta"></p>
    <div class="page zoombox" id="expage" title="Scroll to zoom, drag to pan, double-click to reset">
      <div class="zoomstage"><img id="eximg" alt="comparison charter"></div>
    </div>
  </section>
  <section class="gridpane" id="gridpane" hidden>
    <p class="kicker" id="gkicker">Possible unnamed hand</p>
    <div class="tools">
      <h2 id="gtitle">—</h2>
      <span class="meta" id="gmeta"></span>
    </div>
    <div class="gridwrap">
      <div class="grid" id="grid"></div>
      <div class="inspect" id="ginsp">
        <p class="kicker">Selected page — scroll to zoom, drag to pan, double-click to reset</p>
        <h2 id="gititle">—</h2>
        <p class="meta" id="gimeta"></p>
        <div class="page zoombox" title="Scroll to zoom, drag to pan, double-click to reset">
          <div class="zoomstage"><img id="giimg" alt="selected charter"></div>
        </div>
      </div>
    </div>
  </section>
</main>
<footer>
  <div class="dec">
    <button type="button" class="keep" data-v="keep" title="Keep (1)">Keep<small>Same scribe · 1</small></button>
    <button type="button" class="reject" data-v="reject" title="Reject (2)">Reject<small>Unattribute · 2</small></button>
    <button type="button" class="unsure" data-v="unsure" title="Not sure (3)">Not sure<small>Skip for now · 3</small></button>
    <input type="text" id="note" placeholder="note (optional)">
  </div>
  <div class="bar">
    <button type="button" id="prevCase">Previous case</button>
    <button type="button" id="nextCase">Next case</button>
    <span class="selall" id="selall" hidden>
      <button type="button" id="selAll" title="Tick every page (a)">Select all</button>
      <button type="button" id="selNone" title="Untick every page — confirming then rejects the hand (x)">Deselect all</button>
    </span>
    <span id="tally"></span>
    <button type="button" id="dl">Download my decisions (CSV)</button>
    <span>Never writes labels.csv · ←/→ flip pages · j/k cases · 1/2/3 decide · space tick · a/x all/none</span>
  </div>
</footer>
<script>
__ZOOM_JS__
var D = __PAYLOAD__, decisions = {}, sel = {}, checks = {}, insp = {},
    TAB = ((D.sections || [])[0] || {}).kind || 'attributions', selected = 0,
    sort = 'closest', exi = 0;
function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function section(){
  var ss = D.sections || [];
  for(var i=0;i<ss.length;i++) if(ss[i].kind === TAB) return ss[i];
  return {rows:[], heading:''};
}
function rows(){ return section().rows || []; }
function listFor(r){ return (sort === 'furthest' ? r.furthest : r.closest) || []; }
function distFor(r){ return (sort === 'furthest' ? r.furthest_dist : r.closest_dist) || []; }
function img(i){ return (D.images && D.images[i]) || ''; }
function isGrid(r){ return !!(r && r.kind === 'new_hands' && r.members); }
function nhKey(r, row){ return r.id + '#' + row; }
function nhMembers(r){
  // every page of the group, by distance to the most central page
  var ri = r.members.indexOf(r.reference), out = [];
  for(var k=0;k<r.members.length;k++){
    var sim = (ri >= 0 && r.member_sim && r.member_sim[ri]) ? r.member_sim[ri][k] : 0;
    out.push({row:r.members[k], dist:1 - sim, central:r.members[k] === r.reference});
  }
  out.sort(function(a,b){ return a.dist - b.dist; });
  return out;
}
function nhChecks(r){
  // ticked = same scribe. Defaults: the cluster's proposal (all ticked), or a
  // recorded decision when the reviewer comes back to this case.
  if(!checks[r.id]){
    var c = {};
    r.members.forEach(function(row){
      var d = decisions[nhKey(r, row)];
      c[row] = d ? (d.decision === 'keep') : true;
    });
    checks[r.id] = c;
  }
  return checks[r.id];
}
function nhDecided(r){
  var n = 0;
  for(var k=0;k<r.members.length;k++) if(decisions[nhKey(r, r.members[k])]) n++;
  return {done:n, total:r.members.length};
}
function nhCount(r){
  var c = nhChecks(r), n = 0;
  r.members.forEach(function(row){ if(c[row]) n++; });
  return n;
}
function nhInspected(r){ return (insp[r.id] != null) ? insp[r.id] : r.reference; }
function inspect(r, row){
  insp[r.id] = row;
  var c = nhChecks(r), m = null;
  nhMembers(r).forEach(function(x){ if(x.row === row) m = x; });
  document.getElementById('gititle').textContent = D.names[row] || '—';
  var url = (D.urls && D.urls[row]) || '';
  document.getElementById('gimeta').innerHTML =
    (c[row] ? 'Ticked: same scribe' : 'Unticked: not this scribe') +
    (m && !m.central ? ' · cosine distance to the central page '+fmt(m.dist) : ' · the most central page') +
    (url ? ' · <a href="'+esc(url)+'" target="_blank">open original</a>' : '') +
    (!img(row) ? ' · <i>no image in this file</i>' : '');
  var im = document.getElementById('giimg');
  if(im.getAttribute('data-row') !== String(row)){
    im.setAttribute('data-row', String(row));
    showPage(im, row, null, null);
    if(window.moleResetZoom) window.moleResetZoom(im.closest('.zoombox'));
  }
  document.querySelectorAll('#grid .card').forEach(function(card){
    card.classList.toggle('sel', card.getAttribute('data-row') === String(row));
  });
}
function inspectStep(delta){
  var r = rows()[selected]; if(!isGrid(r)) return;
  var M = nhMembers(r), cur = nhInspected(r), k = 0;
  for(var i=0;i<M.length;i++) if(M[i].row === cur) k = i;
  k = (k + delta + M.length) % M.length;
  inspect(r, M[k].row);
  var card = document.querySelector('#grid .card[data-row="'+M[k].row+'"]');
  if(card && card.scrollIntoView) card.scrollIntoView({block:'nearest'});
}
function toggleInspected(){
  var r = rows()[selected]; if(!isGrid(r)) return;
  var c = nhChecks(r), row = nhInspected(r);
  c[row] = !c[row];
  render();
}
function renderGrid(r){
  var box = document.getElementById('grid'), c = nhChecks(r);
  box.innerHTML = '';
  nhMembers(r).forEach(function(m){
    var card = document.createElement('div');
    card.className = 'card ' + (c[m.row] ? 'in' : 'out');
    card.setAttribute('data-row', String(m.row));
    var thumb = document.createElement('div');
    thumb.className = 'thumb';
    var im = document.createElement('img');
    var uri = img(m.row);
    if(uri) im.src = uri;
    im.alt = D.names[m.row] || '';
    thumb.appendChild(im);
    thumb.addEventListener('click', function(){ inspect(r, m.row); });
    card.appendChild(thumb);
    var lab = document.createElement('label');
    var cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = !!c[m.row];
    cb.addEventListener('change', function(){ c[m.row] = cb.checked; inspect(r, m.row); render(); });
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode('Same scribe'));
    if(m.central){
      var b = document.createElement('span'); b.className = 'badge';
      b.textContent = '· most central'; lab.appendChild(b);
    }
    card.appendChild(lab);
    var meta = document.createElement('p'); meta.className = 'cmeta';
    var url = (D.urls && D.urls[m.row]) || '';
    meta.innerHTML = esc(D.names[m.row]||'') +
      (m.central ? '' : ' · distance to central '+fmt(m.dist)) +
      (url ? ' · <a href="'+esc(url)+'" target="_blank">original</a>' : '') +
      (!uri ? ' · <i>no image in this file</i>' : '');
    card.appendChild(meta);
    box.appendChild(card);
  });
  inspect(r, nhInspected(r));
}
function fmt(x){
  if(x == null || x === '' || (typeof x === 'number' && !isFinite(x))) return '—';
  return Number(x).toFixed(3);
}
function showPage(el, i, metaEl, dist){
  if(i == null || i < 0){
    el.removeAttribute('src'); el.alt = '';
    if(window.moleResetZoom) window.moleResetZoom(el.closest('.zoombox'));
    if(metaEl) metaEl.innerHTML = '';
    return;
  }
  var uri = img(i);
  el.src = uri || '';
  el.alt = D.names[i] || '';
  var url = (D.urls && D.urls[i]) || '';
  if(metaEl) metaEl.innerHTML = esc(D.names[i]||'') +
    (D.hands[i] ? ' · recorded as '+esc(D.hands[i]) : ' · unattributed') +
    (dist != null ? ' · cosine distance '+fmt(dist) : '') +
    (url ? ' · <a href="'+esc(url)+'" target="_blank">open original</a>' : '') +
    (!uri ? ' · <i>no image in this file</i>' : '');
}
function renderDots(n, states){
  var box = document.getElementById('exdots');
  box.innerHTML = '';
  for(var k=0;k<n;k++){
    var b = document.createElement('button');
    b.type = 'button';
    b.className = ((k===exi) ? 'on ' : '') + ((states && states[k]) || '');
    b.title = 'Neighbour '+(k+1);
    b.setAttribute('data-k', k);
    b.addEventListener('click', function(){ exi = +this.getAttribute('data-k'); render(); });
    box.appendChild(b);
  }
}
function renderNums(r){
  var box = document.getElementById('nums');
  if(!r){ box.innerHTML = ''; return; }
  var dists = distFor(r), L = listFor(r);
  var bits = [];
  if(L.length && dists && dists[exi] != null)
    bits.push('<b>cosine distance to this page</b> '+fmt(dists[exi]));
  bits.push('<span class="dim">to hand '+esc(r.hand)+' (top-2 mean)</span> '+fmt(r.hand_dist));
  if(r.kind === 'outliers' && r.closer_hand)
    bits.push('<span class="dim">closer to '+esc(r.closer_hand)+'</span> '+fmt(r.closer_dist));
  if(r.kind === 'attributions' && r.runner_up)
    bits.push('<span class="dim">next best '+esc(r.runner_up)+'</span> '+fmt(r.runner_dist));
  if(isGrid(r)){
    bits = [];
    bits.push('<b>ticked</b> '+nhCount(r)+' of '+r.members.length);
    bits.push('<span class="dim">group cohesion</span> '+fmt(r.hand_cos));
    var s = nhDecided(r);
    if(s.done) bits.push('<span class="dim">recorded</span> '+s.done+' / '+s.total);
  }
  else if(r.kind === 'new_hands')
    bits.push('<span class="dim">cluster cohesion</span> '+fmt(r.hand_cos));
  if(r.calibrated_p != null)
    bits.push('<span class="dim">calibrated</span> '+Math.round(r.calibrated_p*100)+'%');
  box.innerHTML = bits.join('<span class="dim"> · </span>');
}
function caseDecided(r){
  if(isGrid(r)){ var s = nhDecided(r); return s.total === 0 || s.done >= s.total; }
  return !!decisions[r.id];
}
function tabDecided(){
  var n = 0, R = rows();
  for(var i=0;i<R.length;i++) if(caseDecided(R[i])) n++;
  return n;
}
function render(){
  var R = rows(), r = R[selected];
  var heading = (section().heading || 'this tab').toLowerCase();
  var done = tabDecided(), left = Math.max(0, R.length - done);
  var inner = '';
  if(r && isGrid(r)){ inner = ' · <b>'+nhCount(r)+'</b> of <b>'+r.members.length+'</b> pages ticked'; }
  document.getElementById('progress').innerHTML = R.length
    ? ('Case <b>'+(selected+1)+'</b> of <b>'+R.length+'</b> · <b>'+done+'</b> decided · <b>'+left+'</b> left'+inner)
    : ('No '+heading+' for this archive.');
  var bar = document.getElementById('workbar');
  var fill = document.getElementById('workfill');
  var pct = R.length ? Math.round(100 * done / R.length) : 0;
  if(bar){ bar.setAttribute('aria-valuenow', pct); bar.setAttribute('aria-valuemax', '100'); }
  if(fill) fill.style.width = pct + '%';
  var sl = document.getElementById('caseslider');
  if(sl){
    sl.disabled = !R.length;
    sl.max = String(Math.max(1, R.length));
    sl.value = String(R.length ? selected + 1 : 1);
  }
  if(!r){
    document.getElementById('ask').textContent = 'Nothing to review in this tab.';
    document.getElementById('how').textContent = '';
    renderNums(null);
    return;
  }
  document.getElementById('ask').textContent = r.ask || ('Does this charter belong with hand '+r.hand+'?');
  document.getElementById('how').textContent = r.text;
  document.getElementById('qtitle').textContent = r.document || 'This page';
  document.getElementById('exhand').textContent = r.hand || '—';
  document.getElementById('exrole').textContent = (r.hand_role === 'proposed')
    ? 'Proposed as' : (r.hand_role === 'cluster' ? 'Possible unnamed hand' : 'Recorded as');
  document.querySelector('#querypane .kicker').textContent = r.query_kicker || 'Charter in question';
  var keepH = document.querySelector('.dec .keep small');
  var rejH = document.querySelector('.dec .reject small');
  if(keepH) keepH.textContent = r.keep_hint || 'Same scribe · 1';
  if(rejH) rejH.textContent = r.reject_hint || 'Unattribute · 2';
  document.getElementById('exkicker').textContent = (function(){
    var who = (r.hand_role === 'cluster') ? 'this possible unnamed hand'
                                         : ('pages of '+r.hand);
    return (sort === 'furthest' ? 'Least' : 'Most')+' like this charter among '+who;
  })();
  var grid = isGrid(r);
  document.getElementById('selall').hidden = !grid;
  document.getElementById('querypane').hidden = grid;
  document.getElementById('exempane').hidden = grid;
  document.getElementById('gridpane').hidden = !grid;
  if(grid){
    document.getElementById('gtitle').textContent = r.hand || 'unnamed hand';
    document.getElementById('gmeta').textContent = r.members.length+' unattributed pages · ordered by distance to the most central page';
    renderGrid(r);
    renderNums(r);
    var first = decisions[nhKey(r, r.members[0])];
    var state = first ? first.decision : '';
    if(state === 'keep' || state === 'reject'){
      // whole-group verdict: all rejected reads as "reject", otherwise "keep"
      state = (nhDecided(r).done && r.members.every(function(row){
        var d = decisions[nhKey(r, row)]; return d && d.decision === 'reject'; })) ? 'reject' : 'keep';
    }
    document.querySelectorAll('.dec button[data-v]').forEach(function(b){
      b.classList.toggle('on', state === b.getAttribute('data-v'));
    });
    document.getElementById('note').value = (first && first.note) || '';
    var nAllG = Object.keys(decisions).length;
    document.getElementById('tally').innerHTML = nAllG ? ('<b>'+nAllG+'</b> recorded') : '';
    return;
  }
  var qi = (r.focus||[])[0];
  showPage(document.getElementById('qimg'), qi, null, null);
  var qurl = (qi != null && D.urls && D.urls[qi]) || '';
  var qlab = (r.hand_role === 'proposed') ? 'Unattributed · proposed as '
           : (r.hand_role === 'cluster') ? 'Unattributed · reference page of '
           : 'Recorded as ';
  document.getElementById('qmeta').innerHTML = qlab+esc(r.hand||'—') +
    (r.hand_dist != null ? ' · cosine distance to hand '+fmt(r.hand_dist) : '') +
    (qurl ? ' · <a href="'+esc(qurl)+'" target="_blank">open original</a>' : '') +
    ((qi == null || !img(qi)) ? ' · <i>no image in this file</i>' : '');
  var L = listFor(r), dists = distFor(r), states = null;
  var prevEx = document.getElementById('prevEx');
  var nextEx = document.getElementById('nextEx');
  if(!L.length){
    exi = 0;
    document.getElementById('excount').textContent = 'none';
    document.getElementById('eximg').removeAttribute('src');
    document.getElementById('exmeta').innerHTML = 'This hand has no other charter to compare against.';
    renderDots(0);
    prevEx.disabled = nextEx.disabled = true;
  } else {
    if(exi >= L.length) exi = 0;
    if(exi < 0) exi = L.length - 1;
    document.getElementById('excount').textContent = (exi+1)+' / '+L.length;
    showPage(document.getElementById('eximg'), L[exi], document.getElementById('exmeta'), dists[exi]);
    renderDots(L.length, states);
    prevEx.disabled = nextEx.disabled = false;
  }
  renderNums(r);
  var prev = decisions[r.id];
  document.querySelectorAll('.dec button[data-v]').forEach(function(b){
    b.classList.toggle('on', !!(prev && prev.decision === b.getAttribute('data-v')));
  });
  document.getElementById('note').value = (prev && prev.note) || '';
  var nAll = Object.keys(decisions).length;
  document.getElementById('tally').innerHTML = nAll ? ('<b>'+nAll+'</b> recorded') : '';
}
function decide(v){
  var r = rows()[selected]; if(!r) return;
  var note = document.getElementById('note').value || '';
  if(isGrid(r)){
    // keep = record the ticks (ticked pages keep, the rest reject);
    // reject = none of them is a hand; unsure = every page unsure
    var c = nhChecks(r);
    if(v === 'reject') r.members.forEach(function(row){ c[row] = false; });
    r.members.forEach(function(row){
      var d = (v === 'unsure') ? 'unsure' : (c[row] ? 'keep' : 'reject');
      decisions[nhKey(r, row)] = {kind:'new_hand', document:D.names[row]||'',
                                  hand:r.hand||'', decision:d, note:note};
    });
    render();
    if(v !== 'unsure' && selected < rows().length - 1){ select(selected + 1); }
    return;
  }
  decisions[r.id] = {kind:r.csv_kind || r.kind, document:r.document||'',
                     hand:r.hand||'', decision:v, note:note};
  render();
  if(v !== 'unsure' && selected < rows().length - 1){ select(selected + 1); }
}
function selectAll(on){
  var r = rows()[selected]; if(!isGrid(r)) return;
  var c = nhChecks(r);
  r.members.forEach(function(row){ c[row] = !!on; });
  render();
}
function select(i){
  var R = rows();
  if(!R.length){ selected = 0; render(); return; }
  selected = Math.max(0, Math.min(R.length-1, i));
  sel[TAB] = selected;
  sort = 'closest'; exi = 0;
  document.getElementById('sort-closest').classList.add('on');
  document.getElementById('sort-furthest').classList.remove('on');
  render();
}
function setTab(kind){
  TAB = kind;
  document.querySelectorAll('.tabs button[data-tab]').forEach(function(b){
    b.classList.toggle('on', b.getAttribute('data-tab') === kind);
  });
  selected = sel[kind] || 0;
  sort = 'closest'; exi = 0;
  document.getElementById('sort-closest').classList.add('on');
  document.getElementById('sort-furthest').classList.remove('on');
  render();
}
(function(){
  var nav = document.getElementById('tabs');
  (D.sections || []).forEach(function(s){
    var b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('data-tab', s.kind);
    b.className = (s.kind === TAB) ? 'on' : '';
    b.textContent = (s.heading || s.kind) + ' (' + (s.rows||[]).length + ')';
    b.addEventListener('click', function(){ setTab(s.kind); });
    nav.appendChild(b);
  });
})();
document.getElementById('selAll').addEventListener('click', function(){ selectAll(true); });
document.getElementById('selNone').addEventListener('click', function(){ selectAll(false); });

document.getElementById('sort-closest').addEventListener('click', function(){
  sort='closest'; exi=0;
  document.getElementById('sort-closest').classList.add('on');
  document.getElementById('sort-furthest').classList.remove('on');
  render();
});
document.getElementById('sort-furthest').addEventListener('click', function(){
  sort='furthest'; exi=0;
  document.getElementById('sort-furthest').classList.add('on');
  document.getElementById('sort-closest').classList.remove('on');
  render();
});
document.getElementById('prevEx').addEventListener('click', function(){ exi -= 1; render(); });
document.getElementById('nextEx').addEventListener('click', function(){ exi += 1; render(); });
document.getElementById('prevCase').addEventListener('click', function(){ select(selected-1); });
document.getElementById('nextCase').addEventListener('click', function(){ select(selected+1); });
document.getElementById('caseslider').addEventListener('input', function(){
  select(parseInt(this.value, 10) - 1);
});
document.querySelectorAll('.dec button[data-v]').forEach(function(b){
  b.addEventListener('click', function(){ decide(b.getAttribute('data-v')); });
});
document.getElementById('note').addEventListener('input', function(){
  var r = rows()[selected]; if(!r) return;
  var v = this.value;
  if(isGrid(r)){
    r.members.forEach(function(row){ if(decisions[nhKey(r, row)]) decisions[nhKey(r, row)].note = v; });
    return;
  }
  if(decisions[r.id]) decisions[r.id].note = v;
});
document.addEventListener('keydown', function(e){
  var tag = (e.target && e.target.tagName) || '';
  if(tag === 'INPUT' || tag === 'TEXTAREA') return;
  var g = isGrid(rows()[selected]);
  if(e.key === 'ArrowLeft'){ e.preventDefault(); if(g) inspectStep(-1); else { exi -= 1; render(); } }
  if(e.key === 'ArrowRight'){ e.preventDefault(); if(g) inspectStep(1); else { exi += 1; render(); } }
  if(e.key === ' ' && g){ e.preventDefault(); toggleInspected(); }
  if(e.key === 'j' || e.key === 'n'){ e.preventDefault(); select(selected+1); }
  if(e.key === 'k' || e.key === 'p'){ e.preventDefault(); select(selected-1); }
  if(e.key === '1') decide('keep');
  if(e.key === '2') decide('reject');
  if(e.key === '3') decide('unsure');
  if(e.key === 'a') selectAll(true);
  if(e.key === 'x') selectAll(false);
});
document.getElementById('dl').addEventListener('click', function(){
  var out = ['kind,document,hand,decision,note'];
  Object.keys(decisions).forEach(function(k){
    var d = decisions[k];
    out.push([d.kind, d.document, d.hand, d.decision, d.note||''].map(function(v){
      return '"'+String(v).replace(/"/g,'""')+'"';}).join(','));
  });
  var blob = new Blob([out.join('\n')], {type:'text/csv'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'decisions.csv'; a.click();
});
render();
</script></body></html>
"""

_REVIEW_CHROME = r"""
<div class="bar">
  <button id="dl">Download my decisions (CSV)</button>
  <label><input type="checkbox" id="nums"> show the numbers</label>
  <span class="sub" id="tally"></span>
  <span class="sub">Keep / Reject / Unsure — never writes labels.csv</span>
</div>
<nav class="tabs" id="tabs">
  <button type="button" class="on" data-tab="outliers">False positives</button>
  <button type="button" disabled title="Stage 2">False negatives</button>
  <button type="button" disabled title="Stage 3">New hands</button>
</nav>
<div class="hsplit" id="hsplit" title="Drag to resize the panels"></div>
<div class="panel dock" id="panel">
  <div class="queue" id="queue"></div>
  <div class="case" id="casepane"><div class="ph">Highest-confidence first. Select a case.</div></div>
</div>
"""

_REVIEW_JS = r"""
var TAB = 'outliers', selected = 0;
var rowsByTab = {};
(D.sections || []).forEach(function(sec){ rowsByTab[sec.kind] = sec.rows; });
function currentRows(){ return rowsByTab[TAB] || []; }
function renderQueue(){
  var rows = currentRows();
  var q = document.getElementById('queue');
  if(!q) return;
  if(!rows.length){
    q.innerHTML = '<div class="ph">No false-positive suggestions for this archive.</div>';
    return;
  }
  q.innerHTML = rows.map(function(r,i){
    return '<div class="qrow'+(i===selected?' on':'')+'" data-i="'+i+'">'+
           '<div class="t">'+r.title+'</div><div class="x">'+esc(r.text)+'</div>'+
           '<div class="num">'+esc(r.numbers||'')+'</div></div>';
  }).join('');
  q.querySelectorAll('.qrow').forEach(function(el){
    el.addEventListener('click', function(){ select(+el.getAttribute('data-i')); });
  });
}
function renderCase(r){
  var box = document.getElementById('casepane'); if(!box || !r) return;
  var out = '<div class="x">'+r.text+'</div><div class="num">'+esc(r.numbers||'')+'</div>';
  out += '<div class="imgs">';
  var list = (r.focus||[]).concat(r.docs||[]), seen = {}, shown = 0;
  for(var n=0;n<list.length && shown<4;n++){
    var i = list[n]; if(seen[i]) continue; seen[i] = 1;
    if(!D.images[i]) continue;
    out += '<figure><img loading="lazy" src="'+D.images[i]+'">'+
           '<figcaption>'+esc(D.names[i])+
           (D.hands[i] ? ' — '+esc(D.hands[i]) : ' — not attributed')+
           (shown===0 ? ' — subject' : ' — recorded hand')+
           '</figcaption></figure>';
    shown++;
  }
  out += '</div>';
  out += '<div class="dec">'+
    '<button data-v="keep">Keep</button>'+
    '<button data-v="reject">Reject</button>'+
    '<button data-v="unsure">Unsure</button>'+
    '<input type="text" placeholder="note (optional)">'+
    '<span class="sub">j/k next · 1 keep · 2 reject · 3 unsure</span></div>';
  box.innerHTML = out;
  var prev = decisions[r.id];
  box.querySelectorAll('button[data-v]').forEach(function(b){
    if(prev && prev.decision === b.getAttribute('data-v')) b.classList.add('on');
    b.addEventListener('click', function(){
      decide(r, b.getAttribute('data-v'), box.querySelector('input').value);
      renderCase(r);
    });
  });
  var note = box.querySelector('input');
  if(prev) note.value = prev.note || '';
  note.addEventListener('input', function(){
    if(decisions[r.id]) decisions[r.id].note = note.value;
  });
}
function decide(r, v, note){
  decisions[r.id] = {kind:'false_positive', document:r.document||'',
                     hand:r.hand||'', decision:v, note:note||''};
  tally();
}
function select(i){
  var rows = currentRows();
  if(!rows.length) return;
  selected = Math.max(0, Math.min(rows.length-1, i));
  var r = rows[selected];
  light(r);
  if((r.focus||[]).length) showDoc(r.focus[0]);
  renderQueue();
  renderCase(r);
}
function tally(){
  var el = document.getElementById('tally');
  if(!el) return;
  var n = Object.keys(decisions).length;
  el.textContent = n ? n + ' recorded' : '';
}
renderQueue();
if(currentRows().length) select(0);
document.addEventListener('keydown', function(e){
  var tag = (e.target && e.target.tagName) || '';
  if(tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if(e.key === 'j' || e.key === 'ArrowDown'){ e.preventDefault(); select(selected+1); }
  if(e.key === 'k' || e.key === 'ArrowUp'){ e.preventDefault(); select(selected-1); }
  var rows = currentRows();
  if(!rows.length) return;
  if(e.key === '1') decide(rows[selected], 'keep', '');
  if(e.key === '2') decide(rows[selected], 'reject', '');
  if(e.key === '3') decide(rows[selected], 'unsure', '');
  if(e.key === '1' || e.key === '2' || e.key === '3') renderCase(rows[selected]);
});
var nums = document.getElementById('nums');
if(nums) nums.addEventListener('change', function(e){
  document.body.classList.toggle('nums', e.target.checked);
});
var dl = document.getElementById('dl');
if(dl) dl.addEventListener('click', function(){
  var out = ['kind,document,hand,decision,note'];
  Object.keys(decisions).forEach(function(k){
    var d = decisions[k];
    out.push([d.kind, d.document, d.hand, d.decision, d.note||''].map(function(v){
      return '"'+String(v).replace(/"/g,'""')+'"';}).join(','));
  });
  var blob = new Blob([out.join('\n')], {type:'text/csv'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'decisions.csv'; a.click();
});
"""

_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__KIND__ — __TITLE__</title>
<style>__BOKEH_CSS__</style>
<style>
 /* Cursor-app feel: near-black canvas, quiet elevated panels, one cool accent,
    hairline borders and soft shadows. Everything keys off a small token set so the
    light (publication) theme is the same layout with an inverted palette. */
 :root{--bg:#141416;--panel:#1b1b1e;--elev:#232327;--line:#2b2b31;--fg:#e6e6ea;
   --dim:#8a8a94;--accent:#6ea8fe;--accent-weak:rgba(110,168,254,.14);
   --shadow:0 1px 2px rgba(0,0,0,.4),0 6px 20px rgba(0,0,0,.22);--fig-h:74vh}
 *{box-sizing:border-box}
 ::selection{background:var(--accent-weak)}
 ::-webkit-scrollbar{width:10px;height:10px}
 ::-webkit-scrollbar-thumb{background:var(--line);border-radius:8px}
 ::-webkit-scrollbar-thumb:hover{background:#3a3a42}
 ::-webkit-scrollbar-track{background:transparent}
 body{font:13.5px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
   "Helvetica Neue",Arial,system-ui,sans-serif;margin:0;padding:16px 20px 26px;
   background:var(--bg);color:var(--fg);width:100%;
   -webkit-font-smoothing:antialiased;letter-spacing:.005em}
 h1{font-size:17px;font-weight:600;margin:0;letter-spacing:-.01em}
 .sub{color:var(--dim);font-size:12.5px;margin-bottom:12px}
 .sub code,.sub b{color:var(--fg);font-weight:600}
 code{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:11.5px}
 a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
 .bar,.ctl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:12.5px}
 .bar{margin-bottom:10px} .ctl{margin-bottom:12px}
 /* controls read as a segmented toolbar: each label is a pill that lights up when
    its checkbox is on (:has), the Cursor toolbar idiom. */
 .ctl label,.bar label{display:inline-flex;align-items:center;gap:7px;cursor:pointer;
   padding:5px 11px;border:1px solid var(--line);border-radius:9px;
   background:var(--panel);color:var(--dim);transition:.13s ease;user-select:none}
 .ctl label:hover,.bar label:hover{border-color:#3a3a44;color:var(--fg)}
 .ctl label:has(input:checked),.bar label:has(input:checked){
   border-color:var(--accent);background:var(--accent-weak);color:var(--fg)}
 input[type=checkbox],input[type=range]{accent-color:var(--accent);cursor:pointer;margin:0}
 button,select,input[type=text]{background:var(--elev);color:var(--fg);
   border:1px solid var(--line);border-radius:9px;padding:5px 11px;font:inherit;
   font-size:12.5px;cursor:pointer;transition:.13s ease}
 button:hover,select:hover{border-color:var(--accent);color:var(--fg)}
 select{color:var(--fg)}
 .ctl input[type=range]{width:120px}
 .ctl #search{width:160px}
 .ctl #search::placeholder{color:var(--dim)}
 .ctl #search.notfound{border-color:#e5534b;color:#e5534b}
 /* top row: map | draggable divider | charter viewer, across the FULL width. */
 .wrap{display:flex;gap:0;align-items:flex-start;width:100%}
 .mapcol{flex:1 1 55%;min-width:260px}
 .viewcol{flex:1 1 45%;min-width:260px;display:flex;flex-direction:column}
 .split{flex:0 0 16px;height:var(--fig-h);cursor:col-resize;position:relative;
   align-self:flex-start;touch-action:none}
 .split::after{content:"";position:absolute;left:7px;top:0;bottom:0;width:2px;
   background:var(--line);border-radius:2px;transition:.13s ease}
 .split:hover::after,.split.drag::after{background:var(--accent);width:3px;left:6.5px}
 body.dragging{user-select:none;cursor:col-resize}
 body.vdragging{user-select:none;cursor:row-resize}
 .hsplit{height:16px;margin:2px 0;cursor:row-resize;position:relative;touch-action:none}
 .hsplit::after{content:"";position:absolute;top:7px;left:0;right:0;height:2px;
   background:var(--line);border-radius:2px;transition:.13s ease}
 .hsplit:hover::after,.hsplit.drag::after{background:var(--accent);height:3px;top:6.5px}
 @media(max-width:900px){.hsplit{display:none}}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:13px;
   box-shadow:var(--shadow)}
 /* Bokeh's stretch_both needs a parent with a definite height; vh units make the
    map and the charter fill the window instead of a hard-coded pixel box. */
 .card.figbox{height:var(--fig-h);min-height:180px;overflow:hidden}
 .card.figbox>div{width:100%;height:100%}
 body.viz{--fig-h:84vh}
 /* publication light theme (toggle): same layout, inverted palette */
 body.light{--bg:#f6f7f9;--panel:#ffffff;--elev:#ffffff;--line:#e2e5ea;
   --fg:#1a1c22;--dim:#5b616e;--accent:#2563eb;--accent-weak:rgba(37,99,235,.10);
   --shadow:0 1px 2px rgba(20,30,60,.06),0 8px 24px rgba(20,30,60,.06)}
 body.light .lg i.xm::after{color:#c3c8d2}
 /* nearest-neighbour strip under the charter viewer */
 #nn{padding:10px 12px 12px}
 #nn .nnh{font-size:11px;font-weight:600;letter-spacing:.03em;text-transform:uppercase;
   color:var(--dim);margin:2px 0 7px}
 .nnrow{display:flex;gap:9px;overflow-x:auto;padding-bottom:4px}
 .nnrow figure{margin:0;flex:0 0 auto;width:92px;cursor:pointer;text-align:center;
   transition:transform .12s ease}
 .nnrow figure:hover{transform:translateY(-2px)}
 .nnrow img,.nnrow .ph2{width:92px;height:92px;object-fit:cover;
   border:1px solid var(--line);border-radius:9px;background:#0c0c0e;display:block;
   transition:border-color .12s ease}
 .nnrow .ph2{display:flex;align-items:center;justify-content:center;color:var(--dim);
   font-size:11px;background:var(--elev)}
 .nnrow figure:hover img,.nnrow figure:hover .ph2{border-color:var(--accent)}
 .nnrow figcaption{font-size:10.5px;margin-top:4px;white-space:nowrap;overflow:hidden;
   text-overflow:ellipsis;max-width:92px;color:var(--fg)}
 .nnrow .rank{font-size:9.5px;color:var(--dim)}
 .legend{display:flex;flex-wrap:wrap;gap:5px;margin-top:10px;font-size:11.5px;
   max-height:120px;overflow:auto;padding:2px}
 .lg{white-space:nowrap;color:var(--dim);background:var(--panel);border:1px solid var(--line);
   padding:2px 8px 2px 6px;border-radius:20px;display:inline-flex;align-items:center}
 .lg i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;
   vertical-align:baseline;position:relative}
 .lg i.xm::after{content:"×";position:absolute;inset:-2px 0 0 0;color:#000;font-size:10px;
   line-height:9px;text-align:center;font-weight:700}
 .lg b{opacity:.55;font-weight:500;margin-left:3px} .more{opacity:.6;font-style:italic}
 .lg.unl.off{opacity:.35;text-decoration:line-through}
 /* legend chips are tappable to outline a category's region (convex hull) */
 .lg[data-cat]{cursor:pointer}
 .lg.hullon{border-color:var(--accent);color:var(--fg);
   background:var(--accent-weak);box-shadow:0 0 0 1px var(--accent) inset}
 .inspect{padding:12px 14px}
 .inspect h2{font-size:14px;font-weight:600;margin:0 0 2px;word-break:break-all}
 .inspect .meta{font-size:12px;color:var(--dim);margin-bottom:8px}
 .inspect .ph{color:var(--dim);font-size:13px;padding:8px 2px}
 .inspect .figwrap{height:calc(var(--fig-h) - 54px);min-height:126px;
   border-radius:10px;overflow:hidden;background:var(--bg)}
 .inspect .figwrap>div{width:100%;height:100%}
 .inspect .figwrap .zoombox{width:100%;height:100%;border-radius:10px}
__ZOOM_CSS__
 .tabs{display:flex;gap:6px;margin:14px 0 8px;flex-wrap:wrap}
 .tabs button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
 .tabs button:disabled{opacity:.45;cursor:not-allowed}
 .panel.dock{display:flex;gap:14px;width:100%;margin-top:8px;align-items:flex-start}
 .queue{flex:0 0 34%;min-width:220px;max-height:52vh;overflow:auto;border:1px solid var(--line);
   border-radius:13px;background:var(--panel);box-shadow:var(--shadow)}
 .case{flex:1 1 66%;min-width:240px;border:1px solid var(--line);border-radius:13px;
   background:var(--panel);box-shadow:var(--shadow);padding:12px 14px}
 .qrow{padding:9px 12px;border-top:1px solid var(--line);cursor:pointer;transition:background .1s}
 .qrow:first-child{border-top:none}
 .qrow:hover,.qrow.on{background:var(--accent-weak)}
 .qrow .t{font-size:13.5px} .qrow .x,.case .x{font-size:12.5px;color:var(--dim)}
 .qrow .num,.case .num{font-size:11.5px;color:var(--dim);font-family:ui-monospace,monospace;display:none}
 body.nums .qrow .num,body.nums .case .num{display:block}
 .imgs{display:flex;flex-direction:column;gap:8px;margin:8px 0}
 .imgs figure{margin:0} .imgs img{width:100%;border:1px solid var(--line);border-radius:9px}
 .imgs figcaption{font-size:12px;color:var(--dim)}
 .dec{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:6px}
 .dec button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
 .dec input{flex:1;min-width:150px}
 @media(max-width:900px){.panel.dock{flex-direction:column}.queue{flex:1 1 auto;max-height:28vh;width:100%}}
 .dot{stroke:#0006;stroke-width:.5} .dot.sel{stroke:var(--accent);stroke-width:2}
 svg#map{background:var(--bg);border:1px solid var(--line);border-radius:13px;
   width:100%;height:auto}
 @media(max-width:900px){.wrap{flex-direction:column;gap:12px}
   .mapcol,.viewcol{flex:1 1 auto !important;width:100%}.split{display:none}}
</style></head><body class="__BODYCLASS__">
<h1>__KIND__ — __TITLE__</h1>
<div class="sub">__SUBTITLE__</div>
<div class="ctl">
  <label title="Publication light background"><input type="checkbox" id="theme"__THEMECHK__> light theme</label>
  <label title="Print the active category id inside each point"><input type="checkbox" id="labels"__LABELCHK__> class IDs</label>
  <label title="When a charter is selected, draw lines to its nearest neighbours in the underlying space (not the map)"><input type="checkbox" id="nnlines"__NNCHK__> neighbour lines</label>
  <label>point size <input type="range" id="psize" min="3" max="26" step="0.5" value="__PSIZE__"></label>
  __PICKER__
  <input type="text" id="search" placeholder="find charter…" title="Type a filename, then Enter, to jump to that charter">
</div>
<div class="wrap">
  <div class="mapcol">
    <div class="card figbox">__MAP__</div>
    <div class="legend" id="legend">__LEGEND__</div>
  </div>
  <div class="split" id="split" title="Drag to resize"></div>
  <div class="viewcol">
    <div class="card inspect" id="inspect">
      <div id="ihead" class="ph">Click any point on the map to open that charter.</div>
      <div class="figwrap">__VIEWER__</div>
      <div id="nn"></div>
    </div>
  </div>
</div>
__REVIEW_CHROME__
<script>__BOKEH_JS__</script>
__BOKEH_SCRIPT__
<script>__MOLE_JS__</script>
<script>__ZOOM_JS__</script>
<script>
var D = __PAYLOAD__, decisions = {}, active = D.first, N = D.names.length;
var isolatedCat = null;                // category click-isolated (hull + dimmed others)
function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML;}
var hidden = {};                       // row-index -> hidden by the unattributed toggle

function baseAlphas(){
  var a = new Array(N);
  for(var i=0;i<N;i++) a[i] = hidden[i] ? 0 : 0.85;
  return a;
}
// spotlight: keep one category bright, dim the rest — for legend hover preview and
// click isolation. Always respects the unattributed toggle (hidden[] stays hidden).
function spotAlphas(cat){
  var cats = (D.schemes[active] || {}).cats || [];
  var a = new Array(N);
  for(var i=0;i<N;i++) a[i] = hidden[i] ? 0 : (cats[i] === cat ? 0.85 : 0.08);
  return a;
}
// return to whatever the "resting" view is: a click-isolated category if one is
// active, otherwise the plain base alphas.
function restoreAlphas(){
  MOLE.setAlphas(isolatedCat !== null ? spotAlphas(isolatedCat) : baseAlphas());
}
function paint(name){
  var sc = D.schemes[name]; if(!sc) return;
  active = name;
  MOLE.setColors(sc.colors);
  MOLE.setLabels(sc.cats);
  document.getElementById('legend').innerHTML = sc.legend;
  // categories differ across schemes, so a hull/isolation for the old one is
  // meaningless — drop the focus and return to a plain view.
  isolatedCat = null;
  if(MOLE.clearHull) MOLE.clearHull();
  MOLE.setAlphas(baseAlphas());
}
function light(row){
  var hot={}, warm={};
  (row.focus||[]).forEach(function(i){hot[i]=1});
  (row.docs||[]).forEach(function(i){if(!hot[i])warm[i]=1});
  var a = new Array(N);
  for(var i=0;i<N;i++) a[i] = hidden[i] ? 0 : (hot[i] ? 1 : (warm[i] ? 0.7 : 0.05));
  MOLE.setAlphas(a);
}
function unlight(){ restoreAlphas(); }

function showDoc(i){
  var uri = D.images[i], dim = D.dims[i] || [1,1], url = D.urls[i] || '';
  var cat = D.schemes[active].cats[i];
  document.getElementById('ihead').className = '';
  document.getElementById('ihead').innerHTML =
    '<h2>' + esc(D.names[i]) + '</h2><div class="meta">' +
    (D.hands[i] ? esc(D.hands[i]) : 'not attributed') +
    (cat && cat !== D.hands[i] ? ' · ' + esc(active) + ': ' + esc(cat) : '') +
    (url ? ' · <a href="' + esc(url) + '" target="_blank">open original</a>' : '') +
    (uri ? '' : ' · <i>no image embedded — rebuild with --image-scope all</i>') +
    '</div>';
  MOLE.showImage(uri || '', dim[0], dim[1]);
  if(MOLE.select) MOLE.select(i);
  shownDoc = i;
  drawNNLines();
  renderNN(i);
}
var shownDoc = null;
var nnBox = document.getElementById('nnlines');
// connector lines from the selected charter to its true nearest neighbours are
// off by default: the thumbnails under the viewer carry the same information
// and the lines confuse first-time readers of the map
function drawNNLines(){
  if(!MOLE.markNeighbors) return;
  if(nnBox && nnBox.checked && shownDoc != null)
    MOLE.markNeighbors(shownDoc, (D.nn && D.nn[shownDoc]) || []);
  else if(MOLE.clearNeighbors) MOLE.clearNeighbors();
}
if(nnBox) nnBox.addEventListener('change', drawNNLines);
function renderNN(i){
  var box = document.getElementById('nn'); if(!box) return;
  var list = (D.nn && D.nn[i]) || [];
  if(!list.length){ box.innerHTML = ''; return; }
  var out = '<div class="nnh">nearest neighbours (cosine)</div><div class="nnrow">';
  list.forEach(function(j, rank){
    var thumb = D.images[j]
      ? '<img loading="lazy" src="' + D.images[j] + '">'
      : '<div class="ph2">no image</div>';
    out += '<figure data-i="' + j + '" title="' + esc(D.names[j]) + '">' + thumb +
           '<figcaption>' + esc(D.names[j]) + '</figcaption>' +
           '<div class="rank">#' + (rank + 1) +
           (D.hands[j] ? ' · ' + esc(D.hands[j]) : '') + '</div></figure>';
  });
  box.innerHTML = out + '</div>';
  box.querySelectorAll('figure').forEach(function(f){
    f.addEventListener('click', function(){ showDoc(+f.getAttribute('data-i')); });
  });
}
MOLE.onTap(showDoc);
paint(active);

// --- viz/publication controls (shared by both commands)
var themeBox = document.getElementById('theme');
if(themeBox) themeBox.addEventListener('change', function(){
  var light = themeBox.checked;
  document.body.classList.toggle('light', light);
  document.body.classList.toggle('dark', !light);
  MOLE.setTheme(!light);
  window.dispatchEvent(new Event('resize'));
});
var labelBox = document.getElementById('labels');
if(labelBox){
  MOLE.showLabels(labelBox.checked);
  labelBox.addEventListener('change', function(){ MOLE.showLabels(labelBox.checked); });
}
var psize = document.getElementById('psize');
if(psize){
  MOLE.setSize(parseFloat(psize.value));
  psize.addEventListener('input', function(){ MOLE.setSize(parseFloat(psize.value)); });
}

// --- draggable divider between the map and the charter viewer
(function(){
  var split = document.getElementById('split'),
      wrap = document.querySelector('.wrap'),
      mapcol = document.querySelector('.mapcol'),
      right = document.querySelector('.viewcol');
  if(!split) return;
  var dragging = false;
  function move(e){
    if(!dragging) return;
    var r = wrap.getBoundingClientRect();
    var x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    var pct = Math.max(15, Math.min(85, x / r.width * 100));
    mapcol.style.flex = '0 0 ' + pct.toFixed(1) + '%';
    right.style.flex = '1 1 auto';
  }
  function stop(){
    if(!dragging) return;
    dragging = false;
    split.classList.remove('drag');
    document.body.classList.remove('dragging');
    // Bokeh sizes itself from a ResizeObserver; nudge it in case the figure was
    // laid out before the container settled.
    window.dispatchEvent(new Event('resize'));
  }
  split.addEventListener('mousedown', function(e){
    dragging = true; split.classList.add('drag');
    document.body.classList.add('dragging'); e.preventDefault();
  });
  split.addEventListener('touchstart', function(e){
    dragging = true; split.classList.add('drag'); e.preventDefault();
  }, {passive:false});
  window.addEventListener('mousemove', move);
  window.addEventListener('touchmove', move, {passive:false});
  window.addEventListener('mouseup', stop);
  window.addEventListener('touchend', stop);
  // vertical drag: how much height the two panes get, the lists take the rest
  var hsplit = document.getElementById('hsplit'), vdrag = false, startY = 0, startH = 0;
  function figPx(){
    var el = document.querySelector('.figbox');
    return el ? el.getBoundingClientRect().height : 0;
  }
  function vmove(e){
    if(!vdrag) return;
    var y = (e.touches ? e.touches[0].clientY : e.clientY);
    var h = Math.max(180, Math.min(window.innerHeight * 0.92, startH + (y - startY)));
    document.documentElement.style.setProperty('--fig-h', h.toFixed(0) + 'px');
  }
  function vstop(){
    if(!vdrag) return;
    vdrag = false;
    hsplit.classList.remove('drag');
    document.body.classList.remove('vdragging');
    window.dispatchEvent(new Event('resize'));
  }
  if(hsplit){
    hsplit.addEventListener('mousedown', function(e){
      vdrag = true; startY = e.clientY; startH = figPx();
      hsplit.classList.add('drag'); document.body.classList.add('vdragging');
      e.preventDefault();
    });
    hsplit.addEventListener('touchstart', function(e){
      vdrag = true; startY = e.touches[0].clientY; startH = figPx();
      hsplit.classList.add('drag'); e.preventDefault();
    }, {passive:false});
    window.addEventListener('mousemove', vmove);
    window.addEventListener('touchmove', vmove, {passive:false});
    window.addEventListener('mouseup', vstop);
    window.addEventListener('touchend', vstop);
    hsplit.addEventListener('dblclick', function(){
      document.documentElement.style.removeProperty('--fig-h');
      window.dispatchEvent(new Event('resize'));
    });
  }

  split.addEventListener('dblclick', function(){        // double-click = back to 58/42
    mapcol.style.flex = ''; right.style.flex = '';
    window.dispatchEvent(new Event('resize'));
  });
})();

var picker = document.getElementById('scheme');
if(picker) picker.addEventListener('change', function(){ paint(picker.value); });

// Legend chips are interactive (event delegation, because paint() rebuilds
// #legend's innerHTML on every scheme change; the "+N more…" chip carries no
// data-cat and is ignored):
//   HOVER  → spotlight preview: dim every other hand while the mouse is on the chip.
//   CLICK  → focus this hand: draw its convex hull AND persistently isolate it.
//            Clicking the focused chip again clears both; a different chip switches.
var legendEl = document.getElementById('legend');
function memberIdx(cat){
  var cats = (D.schemes[active] || {}).cats || [], idx = [];
  for(var i=0;i<cats.length;i++) if(cats[i] === cat) idx.push(i);
  return idx;
}
legendEl.addEventListener('mouseover', function(e){
  var chip = e.target.closest ? e.target.closest('.lg[data-cat]') : null;
  if(!chip) return;
  var cat = chip.getAttribute('data-cat');
  if(cat === isolatedCat) return;         // already showing this one; nothing to preview
  MOLE.setAlphas(spotAlphas(cat));        // temporary — restored on mouseout
});
legendEl.addEventListener('mouseout', function(e){
  var chip = e.target.closest ? e.target.closest('.lg[data-cat]') : null;
  if(!chip) return;
  // returning to a real element inside the same chip is not a leave
  var to = e.relatedTarget;
  if(to && chip.contains(to)) return;
  restoreAlphas();                        // back to click-isolation, or plain base
});
legendEl.addEventListener('click', function(e){
  var chip = e.target.closest ? e.target.closest('.lg[data-cat]') : null;
  if(!chip) return;
  var cat = chip.getAttribute('data-cat');
  var chips = legendEl.querySelectorAll('.lg.hullon');
  for(var k=0;k<chips.length;k++) chips[k].classList.remove('hullon');
  if(isolatedCat === cat){                // clicking the focused chip clears everything
    isolatedCat = null;
    if(MOLE.clearHull) MOLE.clearHull();
    MOLE.setAlphas(baseAlphas());
  } else {                                // focus a (new) hand: hull + isolation together
    isolatedCat = cat;
    chip.classList.add('hullon');
    var members = memberIdx(cat);
    if(MOLE.showHull) MOLE.showHull(members);
    MOLE.setAlphas(spotAlphas(cat));
    if(MOLE.pulse) MOLE.pulse(members);   // pop the group so it really stands out
  }
});

// Filename search: jump the viewer + map to a charter by (partial) name.
var search = document.getElementById('search');
function doSearch(){
  if(!search) return;
  var q = search.value.trim().toLowerCase();
  if(!q) return;
  var stem = function(n){ return String(n).replace(/\.[^.]+$/, ''); };
  var exact = -1, sub = -1;
  for(var i=0;i<N;i++){
    var nm = String(D.names[i]).toLowerCase(), st = stem(nm);
    if(st === q){ exact = i; break; }                       // prefer an exact stem
    if(sub < 0 && (nm.indexOf(q) >= 0 || st.indexOf(q) >= 0)) sub = i;
  }
  var idx = exact >= 0 ? exact : sub;
  if(idx < 0){                                              // no match: brief red cue
    search.classList.add('notfound');
    setTimeout(function(){ search.classList.remove('notfound'); }, 1000);
    return;
  }
  showDoc(idx);                                             // select + load + NN strip
  if(MOLE.centerOn) MOLE.centerOn(idx);                     // recenter/zoom the map
  if(MOLE.flash) MOLE.flash(idx);                           // brief size pulse
}
if(search) search.addEventListener('keydown', function(e){
  if(e.key === 'Enter'){ e.preventDefault(); doSearch(); }
});
var unl = document.getElementById('unl');
function syncUnl(){
  if(!unl) return;
  var vis = unl.checked, cats = D.schemes['hand'].cats;
  for(var i=0;i<N;i++) hidden[i] = (!vis && !D.hands[i]);
  restoreAlphas();
  var keys = document.querySelectorAll('.lg.unl');
  for(var j=0;j<keys.length;j++) keys[j].classList.toggle('off', !vis);
}
if(unl) unl.addEventListener('change', syncUnl);
__REVIEW_JS__
</script></body></html>"""
