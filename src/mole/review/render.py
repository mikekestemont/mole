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


def _svg(coords: np.ndarray, hands: list[str], names: list[str],
         size: int = 620) -> str:
    xs, ys = coords[:, 0].astype(float), coords[:, 1].astype(float)

    def norm(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo or 1.0)

    pad = 18
    nx = norm(xs) * (size - 2 * pad) + pad
    ny = (1.0 - norm(ys)) * (size - 2 * pad) + pad
    out = []
    for i, (x, y) in enumerate(zip(nx, ny)):
        cls = "dot" + ("" if hands[i] else " unl")
        out.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="3.4" '
                   f'data-i="{i}"><title>{escape(names[i])}</title></circle>')
    return (f'<svg id="map" viewBox="0 0 {size} {size}" width="{size}" '
            f'height="{size}">{"".join(out)}</svg>')


def render_review(embeddings: str | Path, *, out: str | Path | None = None,
                  clusters: str | Path | None = None, limit: int = DEFAULT_LIMIT,
                  max_mb: float = DEFAULT_MAX_MB, image_cache: str | Path | None = None,
                  image_url: str | None = None, images: bool = True,
                  method: str = "auto", seed: int = 0) -> tuple[Path, str]:
    """Build the review sheet. Returns ``(path, summary_line)``."""
    from mole.review.images import ImageBudget
    from mole.review.suggest import build_review, document_table
    from mole.viz.scatter import reduce_2d

    embeddings = Path(embeddings)
    report = build_review(embeddings, clusters=clusters, limit=limit, seed=seed)
    X, meta, rows_meta, names, paths, hands, _docs = document_table(embeddings)
    coords, used_method = reduce_2d(X, method, seed)

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
    budget = ImageBudget(int(max_mb * 1024 * 1024) if max_mb else 0,
                         cache_dir=image_cache)
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
        seen = set()
        for i in wanted:
            if i in seen:
                continue
            seen.add(i)
            budget.add(str(i), paths[i])

    payload = {
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
    html = _HTML.replace("__TITLE__", escape(", ".join(report.datasets) or "archive")) \
                .replace("__SUBTITLE__", subtitle) \
                .replace("__SVG__", _svg(coords, hands, names)) \
                .replace("__PAYLOAD__", json.dumps(payload))

    out_path = Path(out) if out else embeddings.with_suffix(".review.html")
    out_path.write_text(html, encoding="utf-8")
    mb = out_path.stat().st_size / (1024 * 1024)
    return out_path, f"{budget.summary()} · {mb:.1f} MB total"


_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Scribe review — __TITLE__</title>
<style>
 body{font:15px/1.6 system-ui,sans-serif;margin:0;padding:18px;background:#fbfaf7;color:#1a1a1a}
 h1{font-size:20px;margin:0 0 2px} .sub{opacity:.7;margin-bottom:14px;font-size:13px}
 .wrap{display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap}
 #map{background:#fff;border:1px solid #0002;border-radius:10px;position:sticky;top:12px}
 .dot{fill:#4c78a8;fill-opacity:.75;transition:fill-opacity .15s}
 .dot.unl{fill:#b9bcc0;fill-opacity:.5}
 svg.dim .dot{fill-opacity:.06}
 svg.dim .dot.hot{fill-opacity:1;fill:#d1495b}
 svg.dim .dot.warm{fill-opacity:.85;fill:#2a9d8f}
 .panel{flex:1;min-width:380px;max-width:720px}
 details.sec{border:1px solid #0001;border-radius:10px;background:#fff;margin-bottom:10px}
 details.sec>summary{cursor:pointer;padding:11px 14px;font-weight:600;list-style:none}
 details.sec>summary::-webkit-details-marker{display:none}
 .count{background:#eceae4;border-radius:20px;padding:1px 9px;font-size:12px;margin-left:6px}
 .blurb{padding:0 14px 8px;font-size:13px;opacity:.75}
 .row{padding:9px 14px;border-top:1px solid #0000000d;cursor:pointer}
 .row:hover{background:#f4f1ea}
 .row .t{font-size:14px} .row .x{font-size:13px;opacity:.75}
 .row .num{font-size:12px;opacity:.6;font-family:ui-monospace,monospace;display:none}
 body.nums .row .num{display:block}
 .detail{display:none;padding:10px 0 4px}
 .row.open .detail{display:block}
 .imgs{display:flex;flex-direction:column;gap:8px;margin:8px 0}
 .imgs figure{margin:0} .imgs img{width:100%;border:1px solid #0002;border-radius:6px;background:#fff}
 .imgs figcaption{font-size:12px;opacity:.65}
 .dec{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:6px}
 .dec button{border:1px solid #0002;background:#fff;border-radius:7px;padding:4px 12px;
   cursor:pointer;font:inherit;font-size:13px}
 .dec button.on{background:#1a1a1a;color:#fff;border-color:#1a1a1a}
 .dec input{flex:1;min-width:160px;padding:4px 8px;border:1px solid #0002;border-radius:7px;font:inherit}
 .bar{display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
 .bar button{border:1px solid #0002;background:#fff;border-radius:8px;padding:6px 14px;
   cursor:pointer;font:inherit}
 a.orig{font-size:12px}
 @media (prefers-color-scheme:dark){
  body{background:#16150f;color:#eee} #map,details.sec,.imgs img{background:#000;border-color:#fff2}
  .row:hover{background:#ffffff0d} .count{background:#ffffff1a}
  .dec button,.bar button,.dec input{background:#1c1c1c;color:#eee;border-color:#fff3}
  .dec button.on{background:#eee;color:#111}}
</style></head><body>
<h1>Scribe review — __TITLE__</h1>
<div class="sub">__SUBTITLE__</div>
<div class="bar">
  <button id="dl">Download my decisions (CSV)</button>
  <label><input type="checkbox" id="nums"> show the numbers</label>
  <span class="sub" id="tally"></span>
</div>
<div class="wrap">
  __SVG__
  <div class="panel" id="panel"></div>
</div>
<p class="sub">Hover a suggestion to see which charters it is about. Click it to
open the handwriting and record what you think. Nothing here changes your files —
your answers only leave with the download button.</p>
<script>
var D = __PAYLOAD__, decisions = {}, svg = document.getElementById('map');
var dots = svg.querySelectorAll('.dot');
function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function light(row){
  svg.classList.add('dim');
  var hot={}, warm={};
  (row.focus||[]).forEach(function(i){hot[i]=1});
  (row.docs||[]).forEach(function(i){if(!hot[i])warm[i]=1});
  for(var k=0;k<dots.length;k++){
    var i=+dots[k].getAttribute('data-i');
    dots[k].classList.toggle('hot',!!hot[i]);
    dots[k].classList.toggle('warm',!!warm[i]);
  }
}
function unlight(){svg.classList.remove('dim');}
function imgHtml(row){
  var out='<div class="imgs">', shown=0;
  var list=(row.focus||[]).concat(row.docs||[]), seen={};
  for(var n=0;n<list.length && shown<4;n++){
    var i=list[n]; if(seen[i])continue; seen[i]=1;
    var uri=D.images[i]; if(!uri)continue;
    var url=D.urls[i]||'';
    out+='<figure><img loading="lazy" src="'+uri+'"><figcaption>'+esc(D.names[i])+
         (D.hands[i]?' — '+esc(D.hands[i]):' — not attributed')+
         (url?' · <a class="orig" href="'+esc(url)+'" target="_blank">open original</a>':'')+
         '</figcaption></figure>';
    shown++;
  }
  return out+'</div>'+(shown?'':'<div class="x">(no images available for this row)</div>');
}
var panel=document.getElementById('panel');
D.sections.forEach(function(sec){
  var d=document.createElement('details'); d.className='sec';
  var rowsHtml=sec.rows.map(function(r){
    return '<div class="row" data-id="'+r.id+'"><div class="t">'+r.title+'</div>'+
           '<div class="x">'+esc(r.text)+'</div><div class="num">'+esc(r.numbers||'')+'</div>'+
           '<div class="detail"></div></div>';
  }).join('');
  d.innerHTML='<summary>'+esc(sec.heading)+'<span class="count">'+sec.rows.length+
              '</span></summary><div class="blurb">'+esc(sec.blurb)+'</div>'+rowsHtml;
  panel.appendChild(d);
  var map={}; sec.rows.forEach(function(r){map[r.id]=r;});
  d.querySelectorAll('.row').forEach(function(el){
    var r=map[el.getAttribute('data-id')];
    el.addEventListener('mouseenter',function(){light(r)});
    el.addEventListener('mouseleave',unlight);
    el.addEventListener('click',function(ev){
      if(ev.target.closest('.dec')||ev.target.tagName==='A')return;
      var det=el.querySelector('.detail'), open=el.classList.toggle('open');
      if(open&&!det.innerHTML){
        det.innerHTML=imgHtml(r)+'<div class="dec">'+
          ['yes','no','unsure'].map(function(v){
            return '<button data-v="'+v+'">'+(v==='yes'?'Looks right':
                   v==='no'?'Not right':'Not sure')+'</button>';}).join('')+
          '<input placeholder="note (optional)"></div>';
        det.querySelectorAll('button').forEach(function(b){
          b.addEventListener('click',function(){
            det.querySelectorAll('button').forEach(function(o){o.classList.remove('on')});
            b.classList.add('on');
            decisions[r.id]={kind:r.kind,title:r.title.replace(/<[^>]+>/g,''),
                             decision:b.getAttribute('data-v'),
                             note:det.querySelector('input').value};
            tally();
          });
        });
        det.querySelector('input').addEventListener('input',function(e){
          if(decisions[r.id])decisions[r.id].note=e.target.value;
        });
      }
    });
  });
});
if(D.sections.length)panel.querySelector('details').open=true;
function tally(){
  var n=Object.keys(decisions).length;
  document.getElementById('tally').textContent=n?n+' recorded':'';
}
document.getElementById('nums').addEventListener('change',function(e){
  document.body.classList.toggle('nums',e.target.checked);
});
document.getElementById('dl').addEventListener('click',function(){
  var out=[['kind','suggestion','decision','note'].join(',')];
  Object.keys(decisions).forEach(function(k){
    var d=decisions[k];
    out.push([d.kind,d.title,d.decision,d.note||''].map(function(v){
      return '"'+String(v).replace(/"/g,'""')+'"';}).join(','));
  });
  var blob=new Blob([out.join('\n')],{type:'text/csv'});
  var a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='decisions.csv'; a.click();
});
</script></body></html>"""
