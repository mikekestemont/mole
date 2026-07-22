"""The review sheet: one self-contained HTML a non-technical colleague can use.

Left, the familiar 2D map of the archive. Right, the six suggestion lists. Hover
a suggestion and the map answers it — everything dims except the charters that
row is about. Click to open the actual handwriting, stacked at one scale, and
record a decision. Decisions leave as a CSV; ``labels.csv`` is never touched.

Design rules, all of which have a reason:

* **Plain language by default.** No cosine appears unless "show the numbers" is
  ticked. Attributions state a calibrated fact ("of 40 suggestions this
  confident, 36 were correct"); everything else asks a QUESTION, because nothing
  else here has ground truth behind it.
* **Whole pages, losslessly.** See :mod:`mole.review.images` — thumbnails are too
  fuzzy to judge letterforms and lossy coding is *bigger* on bilevel scans.
* **One file.** Images are inlined, so there is no folder to keep alongside it and
  nothing to break when it is emailed.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np

# how many suggestions per list end up in the sheet (D1: ~180 documents, ~10 MB)
DEFAULT_LIMIT = 25
DEFAULT_MAX_MB = 10.0

_SECTIONS = [
    ("attributions", "Unattributed charters that match a known hand",
     "Each of these has no scribe recorded, but its handwriting matches one that does."),
    ("doubts", "Recorded attributions worth re-checking",
     "These charters are recorded under one scribe but sit closer to another."),
    ("merges", "Two names that may be one scribe",
     "The charters under these two names are as alike as each name is to itself."),
    ("splits", "One name that may cover two scribes",
     "The charters under this name fall into two groups that do not resemble each other."),
    ("new_hands", "Groups of unattributed charters that hang together",
     "None of these match a recorded scribe, but they closely match each other — "
     "possibly one hand nobody has named yet."),
    ("duplicates", "The same charter twice?",
     "These pairs are nearly identical images filed under different names."),
    ("isolated", "Unlike anything else",
     "These resemble nothing in the collection — often blank pages, covers or "
     "photographs of something other than a charter."),
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


def _rows_for(kind: str, items: list[dict], members: dict[str, list[int]],
              cal: dict, name_of: list[str]) -> list[dict]:
    """One UI row per suggestion: what to say, what to light up, what to show."""
    rows = []
    for n, it in enumerate(items):
        r = {"kind": kind, "id": f"{kind}-{n}", "numbers": ""}
        if kind == "attributions":
            hand = it["hand"]
            support = members.get(hand, [])[:3]
            r.update(title=f"{it['document']} → <b>{escape(_short(hand))}</b>",
                     text=_confidence_sentence(it.get("calibrated_p"), cal),
                     focus=[it["row"]], docs=[it["row"], *support],
                     numbers=f"score {it['score']:.3f}, margin "
                             f"{(it['margin'] or 0):.3f}, {it['n_support']} charters "
                             f"under this hand")
        elif kind == "doubts":
            hand, other = it["hand"], it["closer_hand"]
            support = members.get(other, [])[:2] + members.get(hand, [])[:2]
            r.update(title=f"{it['document']} — recorded as <b>{escape(_short(hand))}</b>, "
                           f"resembles <b>{escape(_short(other))}</b>",
                     text="Worth a second look: this charter sits closer to the other "
                          "scribe's work than to the one it is filed under.",
                     focus=[it["row"]], docs=[it["row"], *support],
                     numbers=f"own {it['own_score']:.3f} vs {it['closer_score']:.3f} "
                             f"(gap {it['gap']:.3f})")
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
            r.update(title=f"{it['n_docs']} charters that match each other",
                     text="None of these is attributed, and none matches a recorded "
                          "scribe — they may be one hand that has not been named.",
                     focus=it["rows"][:3], docs=it["rows"],
                     numbers=f"cohesion {it['cohesion']:.3f} vs typical "
                             f"{it['reference_cohesion']:.3f}")
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


def _svg(coords: np.ndarray, first_colors: list[str], base_cats: list[str],
         names: list[str], size: int = 620) -> str:
    """The map, with unlabeled documents crossed through as in ``mole viz``.

    The cross is a property of the DOCUMENT, not of the active colouring, so it is
    fixed to the ground-truth (hand) scheme and stays put while fills change —
    under a cluster scheme an unlabeled point is still coloured by its cluster,
    which is exactly the attribution question ("which cluster did it join?").
    """
    from mole.viz.scatter import _UNLABELED_CROSS, _is_unlabeled

    xs, ys = coords[:, 0].astype(float), coords[:, 1].astype(float)

    def norm(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo or 1.0)

    pad = 18
    nx = norm(xs) * (size - 2 * pad) + pad
    ny = (1.0 - norm(ys)) * (size - 2 * pad) + pad
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


def _picker(schemes, scheme_data, hands, expert: bool = False) -> str:
    """Scheme dropdown + the show/hide-unlabeled toggle (both from `mole viz`)."""
    from mole.viz.scatter import _is_unlabeled

    n_unl = sum(1 for h in hands if _is_unlabeled(h or "unlabeled"))
    bits = []
    if len(schemes) > 1:
        opts = "".join(
            f'<option value="{escape(n, quote=True)}">{escape(n)} '
            f'({scheme_data[n]["n_cats"]})</option>' for n, _ in schemes)
        bits.append(f'<label>colour by <select id="scheme">{opts}</select></label>')
    if n_unl:
        bits.append(f'<label><input type="checkbox" id="unl" checked> '
                    f'show unattributed <b>{n_unl}</b></label>')
    checked = " checked" if expert else ""
    bits.append('<label title="Hide the suggestion lists and just browse the map">'
                f'<input type="checkbox" id="expert"{checked}> expert view</label>')
    return "".join(bits)


def render_review(embeddings: str | Path, *, out: str | Path | None = None,
                  clusters: str | Path | None = None, limit: int = DEFAULT_LIMIT,
                  max_mb: float = DEFAULT_MAX_MB, image_cache: str | Path | None = None,
                  image_url: str | None = None, images: bool = True,
                  image_scope: str = "listed", map_backend: str = "auto",
                  expert: bool = False, cluster_method: str = "both",
                  method: str = "auto", seed: int = 0) -> tuple[Path, str]:
    """Build the review sheet. Returns ``(path, summary_line)``."""
    from mole.review.images import ImageBudget
    from mole.review.suggest import build_review, document_table
    from mole.viz.scatter import reduce_2d

    embeddings = Path(embeddings)
    report = build_review(embeddings, clusters=clusters, limit=limit, seed=seed,
                          cluster_method=cluster_method)
    X, meta, rows_meta, names, paths, hands, _docs = document_table(embeddings)
    coords, used_method = reduce_2d(X, method, seed)
    schemes = _schemes(report, hands, paths, clusters)

    members: dict[str, list[int]] = {}
    for i, h in enumerate(hands):
        if h:
            members.setdefault(h, []).append(i)

    sections = []
    for kind, heading, blurb in _SECTIONS:
        items = getattr(report, kind, [])[:limit]
        if items:
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
        for _kind, _h, _b, rws in sections:
            for r in rws:
                wanted.extend(r.get("focus", [])[:4])
        for _kind, _h, _b, rws in sections:
            for r in rws:
                wanted.extend(r.get("docs", [])[:4])
        if image_scope == "all":
            # expert mode clicks arbitrary dots, so every document needs a page —
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
        "dims": {k: list(v) for k, v in budget.dims.items()},
        "schemes": {n: {"colors": p["colors"], "cats": p["cats"],
                        "legend": p["legend"]} for n, p in scheme_data.items()},
        "first": schemes[0][0],
        "sections": [{"kind": k, "heading": h, "blurb": b, "rows": r}
                     for k, h, b, r in sections],
        "images": budget.uris,
        "names": names,
        "hands": [_short(h) for h in hands],
        "urls": ([image_url.replace("{filename}", n) for n in names] if image_url
                 else [p.resolve().as_uri() if p.is_file() else "" for p in paths]),
    }
    subtitle = (f"{report.n_documents} charters · {report.n_labeled} with a recorded "
                f"scribe · {report.n_hands} scribes · map: {used_method}")

    if use_bokeh:
        bk_script, map_div, view_div, bk_css, bk_js = bokeh_map.build(
            coords, names, [_short(h) for h in hands], first["colors"])
        glue = bokeh_map.glue_js()
        viewer_html = view_div
    else:
        bk_script = map_div = bk_css = bk_js = ""
        map_div = _svg(coords, first["colors"], schemes[0][1], names)
        glue = _svg_glue_js()
        viewer_html = '<img id="pageimg" style="display:none">' 
    html = _HTML.replace("__BODYCLASS__", "expert" if expert else "") \
                .replace("__TITLE__", escape(", ".join(report.datasets) or "archive")) \
                .replace("__SUBTITLE__", subtitle) \
                .replace("__BOKEH_CSS__", bk_css) \
                .replace("__MAP__", map_div) \
                .replace("__VIEWER__", viewer_html) \
                .replace("__PICKER__", _picker(schemes, scheme_data, hands, expert)) \
                .replace("__LEGEND__", first["legend"]) \
                .replace("__PAYLOAD__", json.dumps(payload)) \
                .replace("__BOKEH_JS__", bk_js) \
                .replace("__BOKEH_SCRIPT__", bk_script) \
                .replace("__MOLE_JS__", glue)

    out_path = Path(out) if out else embeddings.with_suffix(".review.html")
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
      for(var i=0;i<dots.length;i++)
        dots[i].setAttribute('fill-opacity', alphas[+dots[i].getAttribute('data-i')]);
    },
    showImage: function(uri, w, h){
      var img = document.getElementById('pageimg');
      if(!img) return;
      img.style.display = uri ? '' : 'none';
      if(uri) img.src = uri;
    },
    onTap: function(cb){ tapcb = cb; },
    select: function(i){
      for(var k=0;k<dots.length;k++)
        dots[k].classList.toggle('sel', +dots[k].getAttribute('data-i') === i);
    }
  };
})();
"""

_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scribe review — __TITLE__</title>
<style>__BOKEH_CSS__</style>
<style>
 :root{--bg:#0f1016;--panel:#171922;--line:#2a2c39;--fg:#e8e8ec;--dim:#9aa0b0}
 *{box-sizing:border-box}
 /* Roboto if the reader has it (common on Linux/Android and any machine with
    Google Fonts installed), otherwise the nearest system equivalent. Not fetched:
    a webfont link would break the offline guarantee, and embedding one costs
    ~40 KB per weight — say the word and it becomes --embed-font. */
 body{font:15px/1.55 Roboto,"Helvetica Neue",Arial,system-ui,sans-serif;
   margin:0;padding:14px 18px;background:var(--bg);
   color:var(--fg);width:100%}
 h1{font-size:19px;margin:0} .sub{opacity:.65;font-size:13px;margin-bottom:10px}
 .bar,.ctl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:13px}
 .bar{margin-bottom:10px} .ctl{margin-bottom:6px}
 .ctl label,.bar label{display:inline-flex;align-items:center;gap:6px;cursor:pointer}
 button,select,input[type=text]{background:#1e2130;color:var(--fg);border:1px solid var(--line);
   border-radius:7px;padding:5px 12px;font:inherit;font-size:13px;cursor:pointer}
 .wrap{display:flex;gap:0;align-items:flex-start;width:100%}
 .mapcol{flex:1 1 58%;min-width:260px}
 .right{flex:1 1 42%;min-width:260px;display:flex;flex-direction:column;gap:12px}
 /* draggable divider: grab anywhere in the 14px gutter */
 .split{flex:0 0 14px;height:74vh;cursor:col-resize;position:relative;
   align-self:flex-start;touch-action:none}
 .split::after{content:"";position:absolute;left:6px;top:0;bottom:0;width:2px;
   background:var(--line);border-radius:2px}
 .split:hover::after,.split.drag::after{background:#5a7fd6}
 body.expert .split{height:84vh}
 body.dragging{user-select:none;cursor:col-resize}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:10px}
 /* Bokeh's stretch_both needs a parent with a definite height; vh units make the
    map and the charter fill the window instead of a hard-coded pixel box. */
 .card.figbox{height:74vh;min-height:380px}
 .card.figbox>div{width:100%;height:100%}
 body.expert .figbox{height:84vh}
 .legend{display:flex;flex-wrap:wrap;gap:3px 12px;margin-top:8px;font-size:12px;
   max-height:120px;overflow:auto}
 .lg{white-space:nowrap;opacity:.9}
 .lg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;
   vertical-align:baseline;position:relative}
 .lg i.xm::after{content:"×";position:absolute;inset:-1px 0 0 0;color:#000;font-size:11px;
   line-height:10px;text-align:center;font-weight:700}
 .lg b{opacity:.5;font-weight:500} .more{opacity:.6;font-style:italic}
 .lg.unl.off{opacity:.3;text-decoration:line-through}
 .inspect{padding:10px 12px}
 .inspect h2{font-size:14px;margin:0 0 2px;word-break:break-all}
 .inspect .meta{font-size:12px;color:var(--dim);margin-bottom:6px}
 .inspect .ph{color:var(--dim);font-size:13px;padding:6px 0}
 .inspect .figwrap{height:66vh;min-height:320px}
 .inspect .figwrap>div{width:100%;height:100%}
 body.expert .inspect .figwrap{height:76vh}
 #pageimg{width:100%;max-height:70vh;object-fit:contain;border-radius:6px;background:#000}
 .panel{display:flex;flex-direction:column;gap:9px}
 body.expert .panel,body.expert .bar{display:none}
 details.sec{border:1px solid var(--line);border-radius:10px;background:var(--panel)}
 details.sec>summary{cursor:pointer;padding:10px 13px;font-weight:600;list-style:none}
 details.sec>summary::-webkit-details-marker{display:none}
 .count{background:#ffffff14;border-radius:20px;padding:1px 9px;font-size:12px;margin-left:6px}
 .blurb{padding:0 13px 8px;font-size:12.5px;color:var(--dim)}
 .row{padding:8px 13px;border-top:1px solid #ffffff0f;cursor:pointer}
 .row:hover{background:#ffffff0a}
 .row .t{font-size:13.5px} .row .x{font-size:12.5px;color:var(--dim)}
 .row .num{font-size:11.5px;color:var(--dim);font-family:ui-monospace,monospace;display:none}
 body.nums .row .num{display:block}
 .detail{display:none;padding:8px 0 4px} .row.open .detail{display:block}
 .imgs{display:flex;flex-direction:column;gap:8px;margin:8px 0}
 .imgs figure{margin:0} .imgs img{width:100%;border:1px solid var(--line);border-radius:6px}
 .imgs figcaption{font-size:12px;color:var(--dim)}
 .dec{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:6px}
 .dec button.on{background:#e8e8ec;color:#111} .dec input{flex:1;min-width:150px}
 a{color:#8ab4f8} .dot{stroke:#0006;stroke-width:.5} .dot.sel{stroke:#fff;stroke-width:2}
 svg#map{background:#12131a;border:1px solid var(--line);border-radius:10px;width:100%;height:auto}
 @media(max-width:900px){.wrap{flex-direction:column;gap:12px}
   .mapcol,.right{flex:1 1 auto !important;width:100%}.split{display:none}}
</style></head><body class="__BODYCLASS__">
<h1>Scribe review — __TITLE__</h1>
<div class="sub">__SUBTITLE__</div>
<div class="bar">
  <button id="dl">Download my decisions (CSV)</button>
  <label><input type="checkbox" id="nums"> show the numbers</label>
  <span class="sub" id="tally"></span>
</div>
<div class="wrap">
  <div class="mapcol">
    <div class="ctl">__PICKER__</div>
    <div class="card figbox">__MAP__</div>
    <div class="legend" id="legend">__LEGEND__</div>
  </div>
  <div class="split" id="split" title="Drag to resize"></div>
  <div class="right">
    <div class="card inspect" id="inspect">
      <div id="ihead" class="ph">Click any point on the map to open that charter.</div>
      <div class="figwrap">__VIEWER__</div>
    </div>
    <div class="panel" id="panel"></div>
  </div>
</div>
<script>__BOKEH_JS__</script>
__BOKEH_SCRIPT__
<script>__MOLE_JS__</script>
<script>
var D = __PAYLOAD__, decisions = {}, active = D.first, N = D.names.length;
function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML;}
var hidden = {};                       // row-index -> hidden by the unattributed toggle

function baseAlphas(){
  var a = new Array(N);
  for(var i=0;i<N;i++) a[i] = hidden[i] ? 0 : 0.85;
  return a;
}
function paint(name){
  var sc = D.schemes[name]; if(!sc) return;
  active = name;
  MOLE.setColors(sc.colors);
  document.getElementById('legend').innerHTML = sc.legend;
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
function unlight(){ MOLE.setAlphas(baseAlphas()); }

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
}
MOLE.onTap(showDoc);
paint(active);

// --- draggable divider between the map and the charter viewer
(function(){
  var split = document.getElementById('split'),
      wrap = document.querySelector('.wrap'),
      mapcol = document.querySelector('.mapcol'),
      right = document.querySelector('.right');
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
  split.addEventListener('dblclick', function(){        // double-click = back to 58/42
    mapcol.style.flex = ''; right.style.flex = '';
    window.dispatchEvent(new Event('resize'));
  });
})();

var picker = document.getElementById('scheme');
if(picker) picker.addEventListener('change', function(){ paint(picker.value); });
var unl = document.getElementById('unl');
function syncUnl(){
  if(!unl) return;
  var vis = unl.checked, cats = D.schemes['hand'].cats;
  for(var i=0;i<N;i++) hidden[i] = (!vis && !D.hands[i]);
  MOLE.setAlphas(baseAlphas());
  var keys = document.querySelectorAll('.lg.unl');
  for(var j=0;j<keys.length;j++) keys[j].classList.toggle('off', !vis);
}
if(unl) unl.addEventListener('change', syncUnl);
var expert = document.getElementById('expert');
if(expert) expert.addEventListener('change', function(){
  document.body.classList.toggle('expert', expert.checked);
});

var panel = document.getElementById('panel');
D.sections.forEach(function(sec){
  var d = document.createElement('details'); d.className = 'sec';
  var rowsHtml = sec.rows.map(function(r){
    return '<div class="row" data-id="'+r.id+'"><div class="t">'+r.title+'</div>'+
           '<div class="x">'+esc(r.text)+'</div><div class="num">'+esc(r.numbers||'')+'</div>'+
           '<div class="detail"></div></div>';
  }).join('');
  d.innerHTML = '<summary>'+esc(sec.heading)+'<span class="count">'+sec.rows.length+
                '</span></summary><div class="blurb">'+esc(sec.blurb)+'</div>'+rowsHtml;
  panel.appendChild(d);
  var map = {}; sec.rows.forEach(function(r){ map[r.id] = r; });
  d.querySelectorAll('.row').forEach(function(el){
    var r = map[el.getAttribute('data-id')];
    el.addEventListener('mouseenter', function(){ light(r); });
    el.addEventListener('mouseleave', unlight);
    el.addEventListener('click', function(ev){
      if(ev.target.closest('.dec') || ev.target.tagName === 'A') return;
      if((r.focus||[]).length) showDoc(r.focus[0]);
      var det = el.querySelector('.detail'), open = el.classList.toggle('open');
      if(open && !det.innerHTML){
        var out = '<div class="imgs">', shown = 0;
        var list = (r.focus||[]).concat(r.docs||[]), seen = {};
        for(var n=0;n<list.length && shown<4;n++){
          var i = list[n]; if(seen[i]) continue; seen[i] = 1;
          if(!D.images[i]) continue;
          out += '<figure><img loading="lazy" src="'+D.images[i]+'">'+
                 '<figcaption>'+esc(D.names[i])+
                 (D.hands[i] ? ' — '+esc(D.hands[i]) : ' — not attributed')+
                 '</figcaption></figure>';
          shown++;
        }
        out += '</div>' + (shown ? '' : '<div class="x">(no images for this row)</div>');
        det.innerHTML = out + '<div class="dec">' +
          ['yes','no','unsure'].map(function(v){
            return '<button data-v="'+v+'">'+(v==='yes'?'Looks right':
                   v==='no'?'Not right':'Not sure')+'</button>';}).join('') +
          '<input type="text" placeholder="note (optional)"></div>';
        det.querySelectorAll('button').forEach(function(b){
          b.addEventListener('click', function(){
            det.querySelectorAll('button').forEach(function(o){o.classList.remove('on')});
            b.classList.add('on');
            decisions[r.id] = {kind:r.kind, title:r.title.replace(/<[^>]+>/g,''),
                               decision:b.getAttribute('data-v'),
                               note:det.querySelector('input').value};
            tally();
          });
        });
        det.querySelector('input').addEventListener('input', function(e){
          if(decisions[r.id]) decisions[r.id].note = e.target.value;
        });
      }
    });
  });
});
if(D.sections.length) panel.querySelector('details').open = true;
function tally(){
  var n = Object.keys(decisions).length;
  document.getElementById('tally').textContent = n ? n + ' recorded' : '';
}
document.getElementById('nums').addEventListener('change', function(e){
  document.body.classList.toggle('nums', e.target.checked);
});
document.getElementById('dl').addEventListener('click', function(){
  var out = [['kind','suggestion','decision','note'].join(',')];
  Object.keys(decisions).forEach(function(k){
    var d = decisions[k];
    out.push([d.kind,d.title,d.decision,d.note||''].map(function(v){
      return '"'+String(v).replace(/"/g,'""')+'"';}).join(','));
  });
  var blob = new Blob([out.join('\n')], {type:'text/csv'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'decisions.csv'; a.click();
});
</script></body></html>"""
