"""2D scatter of an embeddings file for visual inspection of cluster structure.

Reads a ``mole embed`` output (``.npy`` + its ``.mapping.json`` sidecar), projects
the page vectors to 2D, and writes a **self-contained interactive HTML** scatter —
one point per document, coloured by a metadata field, filename on hover. Designed to
be produced on the server and downloaded, or run locally on a downloaded ``.npy``.

Projection backends (``--method``):

* ``pca``  -- always available (NumPy SVD); instant, linear.
* ``tsne`` -- scikit-learn (a core dep); good local cluster structure.
* ``umap`` -- ``umap-learn`` (optional ``mole[viz]`` extra); best global+local.
* ``auto`` -- umap if installed, else pca.

High-dimensional inputs (e.g. VLAD's ``K*dim``) are PCA-reduced before t-SNE/UMAP.
The default UMAP path follows the Sluis charter-viz recipe: **sklearn PCA(150,
whiten=True) → precomputed Euclidean distances → UMAP(n=15, min_dist=0.1)**.
Everything is seeded, so a run is reproducible.
"""

from __future__ import annotations

import colorsys
import json
import re
import unicodedata
from html import escape
from pathlib import Path

import numpy as np

_HIGHLIGHT_STROKE = "#CC0000"


# ------------------------------------------------------------------------ load
def _load_embeddings(path: Path):
    path = Path(path)
    npy = path if path.suffix == ".npy" else path.with_suffix(".npy")
    X = np.load(npy)
    sidecar = npy.with_suffix(".mapping.json")
    meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    rows = meta.get("rows") or [{"row": i, "image": str(i)} for i in range(len(X))]
    if len(rows) != len(X):  # defensive: keep vectors and labels aligned
        rows = [{"row": i, "image": str(i)} for i in range(len(X))]
    return X, meta, rows


# -------------------------------------------------------------------- projection
def _pca(X: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` principal-component scores via SVD (no sklearn needed).

    When there are far more dimensions than rows — VLAD is 38,400-d and an archive
    holds a few hundred pages — a full SVD wastes most of its work computing a
    38,400-wide basis nobody asked for. The Gram route (n x n eigendecomposition)
    is mathematically identical for the scores and orders of magnitude cheaper.
    """
    Xc = (X - X.mean(0, keepdims=True)).astype(np.float32)
    n, d = Xc.shape
    if d > 2 * n:
        w, v = np.linalg.eigh(Xc @ Xc.T)                # [n, n], ascending
        idx = np.argsort(-w)[:min(k, n)]
        return (v[:, idx] * np.sqrt(np.maximum(w[idx], 0.0))).astype(np.float32)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, vt.shape[0])
    return (Xc @ vt[:k].T).astype(np.float32)


def _pca_sklearn(X: np.ndarray, k: int, seed: int, *, whiten: bool) -> np.ndarray:
    """PCA pre-reduction via sklearn (supports whitening — the Sluis viz recipe)."""
    from sklearn.decomposition import PCA

    k = min(int(k), X.shape[0], X.shape[1])
    if k < 2:
        return np.asarray(X, dtype=np.float32)
    return PCA(n_components=k, whiten=whiten, random_state=seed).fit_transform(
        np.asarray(X, dtype=np.float64)).astype(np.float32)


def reduce_2d(X: np.ndarray, method: str = "auto", seed: int = 0,
              pca_dim: int = 150, *, pca_whiten: bool = True,
              umap_neighbors: int = 15, umap_min_dist: float = 0.1) -> tuple[np.ndarray, str]:
    """Project ``[N, D]`` to ``[N, 2]``; returns (coords, method_used).

    For the nonlinear backends the input is first PCA-reduced to ``pca_dim`` dims.
    **UMAP** (and ``auto`` when umap-learn is installed) uses the Sluis charter-viz
    recipe by default: whitened PCA, then UMAP on a **precomputed Euclidean distance
    matrix** with ``n_neighbors=15`` and ``min_dist=0.1``. Plain ``pca`` goes straight
    to 2 components (linear, fast, but rarely as pretty as UMAP).
    """
    X = np.asarray(X, dtype=np.float32)
    method = method.lower()
    if method == "auto":
        try:
            import umap  # noqa: F401
            method = "umap"
        except ImportError:
            method = "pca"
    if method == "pca":
        return _pca(X, 2), "pca"

    k = min(pca_dim, X.shape[1], max(2, X.shape[0] - 1))
    reduced = X.shape[1] > k
    if method == "umap":
        pre = _pca_sklearn(X, k, seed, whiten=pca_whiten) if reduced else X
    else:
        pre = _pca(X, k) if reduced else X
    whiten_tag = " whiten" if (method == "umap" and pca_whiten and reduced) else ""
    tag = f" (pca-{k}{whiten_tag})" if reduced else ""
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(5, (len(pre) - 1) // 3))
        coords = TSNE(n_components=2, random_state=seed, init="pca",
                      perplexity=perplexity).fit_transform(pre)
        return coords, f"tsne{tag}"
    if method == "umap":
        import warnings

        try:
            import umap
        except ImportError as e:
            raise ImportError("method='umap' needs umap-learn: pip install 'mole[viz]' "
                              "(or use --method tsne / pca)") from e
        from scipy.spatial.distance import pdist, squareform

        n_neighbors = min(int(umap_neighbors), len(pre) - 1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*n_jobs value.*overridden.*")
            warnings.filterwarnings("ignore", message=r".*precomputed metric.*")
            dist = squareform(pdist(pre, metric="euclidean"))
            coords = umap.UMAP(
                n_components=2,
                n_neighbors=max(2, n_neighbors),
                min_dist=float(umap_min_dist),
                metric="precomputed",
                random_state=seed,
            ).fit_transform(dist)
        umap_tag = (f"n={n_neighbors} d={umap_min_dist:g} precomputed")
        return coords, f"umap ({umap_tag}{tag})"
    raise ValueError(f"unknown method {method!r} (pca|tsne|umap|auto)")


# ------------------------------------------------------------------- categories
def _categories(rows: list[dict], color: str, color_regex: str | None) -> list[str]:
    """A category label per point, for colouring (dataset / hand / regex / none)."""
    from mole.data.datasets import load_labels

    label_cache: dict[Path, object] = {}

    def hand_of(img: Path) -> str:
        if img.parent not in label_cache:
            label_cache[img.parent] = load_labels(img.parent)
        return label_cache[img.parent].hand_by_filename.get(img.name, "unlabeled")

    pattern = re.compile(color_regex) if color_regex else None
    cats = []
    for r in rows:
        img = Path(r["image"])
        if pattern is not None:
            m = pattern.search(img.name)
            cats.append((m.group(1) if m.groups() else m.group(0)) if m else "—")
        elif color == "hand":
            cats.append(hand_of(img))
        elif color == "none":
            cats.append("all")
        else:  # dataset = parent folder name
            cats.append(img.parent.name or "root")
    return cats


# Category values that mean "this document has no ground-truth hand". Covers mole's
# own placeholders ("unlabeled" from a missing labels.csv row, "—" from a regex miss)
# and the conventions that show up in supplied label files ("-1" is the noise/unassigned
# marker in the Antwerp clusterings spreadsheet).
_UNLABELED = {"unlabeled", "-1", "—", "-", "", "none", "nan", "unknown", "na", "n/a", "?"}

# Unlabeled points are drawn as a neutral grey disc with a cross through it: they read
# as "no ground truth here" rather than as just another hand, and the categorical
# palette is then spent entirely on real hands.
_UNLABELED_GREY = "#9aa0a6"
_UNLABELED_CROSS = "#2f3336"


def _is_unlabeled(cat: str) -> bool:
    return str(cat).strip().lower() in _UNLABELED


def _palette(n: int) -> list[str]:
    if n <= 1:
        return ["#4c78a8"]
    return ["#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 0.62, 0.9)))
            for i in range(n)]


def _nfc(s: str) -> str:
    """Unicode NFC — macOS filenames are often NFD while highlight lists are NFC."""
    return unicodedata.normalize("NFC", s)


def _parse_highlights(items: list[str] | None,
                      highlight_file: str | Path | None) -> set[str]:
    """Normalised filename stems to ring-highlight (Sluis-style target overlay)."""
    out: set[str] = set()
    for raw in items or []:
        raw = raw.strip()
        if raw:
            out.add(_nfc(Path(raw).stem))
    if highlight_file:
        for line in Path(highlight_file).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                out.add(_nfc(Path(line).stem))
    return out


def _is_highlighted(image: str, highlights: set[str]) -> bool:
    return _nfc(Path(image).stem) in highlights if highlights else False


def _text_on(hex_color: str) -> str:
    """Black or white label text for legibility on a filled circle."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "#111"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#111" if lum > 140 else "#fff"


def _label_text(cat: str, max_len: int = 8) -> str:
    """Short class id for in-circle labels (hand-group numbers, cluster ids, …)."""
    s = str(cat).strip()
    if _is_unlabeled(s):
        return ""
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


# -------------------------------------------------------------------------- html
def _scheme_payload(cats: list[str]) -> dict:
    """Colours + legend for one colouring of the points.

    Unlabeled always takes grey so the categorical palette is spent on real
    categories; the legend is capped because a FINCH level can have hundreds of
    clusters and an uncapped legend would dwarf the plot.
    """
    uniq = sorted(set(cats), key=lambda c: (-cats.count(c), c))
    labeled_uniq = [c for c in uniq if not _is_unlabeled(c)]
    cmap = dict(zip(labeled_uniq, _palette(len(labeled_uniq))))
    cmap.update({c: _UNLABELED_GREY for c in uniq if _is_unlabeled(c)})
    show = uniq[:60]
    # data-cat carries the machine-readable category (== the values in ``cats``) so
    # the page can find a chip's member points and draw its convex hull on tap. The
    # "…and N more" chip is deliberately NOT tagged: it names no single category.
    legend = "".join(
        f'<span class="lg{" unl" if _is_unlabeled(c) else ""}" '
        f'data-cat="{escape(str(c), quote=True)}">'
        f'<i class="{"xm" if _is_unlabeled(c) else ""}" style="background:{cmap[c]}"></i>'
        f'{escape(str(c))} <b>{cats.count(c)}</b></span>' for c in show)
    if len(uniq) > len(show):
        legend += f'<span class="lg more">+{len(uniq) - len(show)} more…</span>'
    return {"colors": [cmap[c] for c in cats], "cats": [str(c) for c in cats],
            "legend": legend, "n_cats": len(uniq)}


def _build_html(coords, schemes, rows, meta, method, *,
                highlights: set[str] | None = None,
                theme: str = "light",
                show_labels: bool = False,
                point_size: float = 4.5) -> str:
    """``schemes`` is ``[(name, cats), ...]``; the first is shown initially.

    Extra schemes (e.g. one per FINCH level) are switchable in the browser: the
    circles' fill is repainted from a per-scheme colour array, so ground truth and
    discovered clusters can be flipped in place on the SAME projection — which is the
    only honest way to ask "do the clusters recover the known hands?".

    ``highlights`` ring specific documents (hollow red overlay + stem label), abstracting
    the Sluis ``HIGHLIGHT_FILES`` pattern for any archive.
    """
    highlights = highlights or set()
    theme = "dark" if theme.lower() == "dark" else "light"
    point_size = float(max(2.0, min(30.0, point_size)))
    xs, ys = coords[:, 0].astype(float), coords[:, 1].astype(float)

    def norm(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo or 1.0)

    W = H = 900
    pad = 44
    nx = norm(xs) * (W - 2 * pad) + pad
    ny = (1.0 - norm(ys)) * (H - 2 * pad) + pad          # SVG y grows downward

    names = [n for n, _ in schemes]
    payloads = {n: _scheme_payload(c) for n, c in schemes}
    first = payloads[names[0]]
    base_cats = schemes[0][1]
    hl_ring = point_size + 6

    pts = []
    n_highlighted = 0
    for i, (x, y, r) in enumerate(zip(nx, ny, rows)):
        stem = Path(r["image"]).stem
        name = escape(Path(r["image"]).name, quote=True)
        fill = first["colors"][i]
        cat = base_cats[i]
        lbl = escape(_label_text(cat), quote=True)
        txt_fill = _text_on(fill)
        hl = _is_highlighted(r["image"], highlights)
        if hl:
            n_highlighted += 1
        unl = _is_unlabeled(cat)
        parts = [
            f'<g class="pt" data-i="{i}" data-hl="{1 if hl else 0}"'
            f' data-unl="{1 if unl else 0}">',
            (f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="{point_size:.1f}" '
             f'fill="{fill}" fill-opacity="0.82" stroke="#0003" stroke-width="0.5" '
             f'data-i="{i}" data-name="{name}"/>'),
        ]
        if hl:
            parts.append(
                f'<circle class="hl-ring" cx="{x:.1f}" cy="{y:.1f}" r="{hl_ring:.1f}" '
                f'fill="none" stroke="{_HIGHLIGHT_STROKE}" stroke-width="2.5" '
                f'pointer-events="none"/>')
            parts.append(
                f'<text class="hl-lbl" x="{x + hl_ring + 4:.1f}" y="{y:.1f}" '
                f'dominant-baseline="central" fill="{_HIGHLIGHT_STROKE}" font-size="11" '
                f'font-weight="700" pointer-events="none">{escape(stem)}</text>')
        if lbl:
            vis = "visible" if show_labels else "hidden"
            fs = max(6.0, point_size * 0.85)
            parts.append(
                f'<text class="lbl" x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
                f'dominant-baseline="central" font-size="{fs:.1f}" font-weight="600" '
                f'fill="{txt_fill}" visibility="{vis}" pointer-events="none">{lbl}</text>')
        if unl:
            a = point_size * 0.48
            parts.append(
                f'<path class="unl-x" d="M{x - a:.1f} {y - a:.1f}L{x + a:.1f} {y + a:.1f}'
                f'M{x - a:.1f} {y + a:.1f}L{x + a:.1f} {y - a:.1f}" stroke="{_UNLABELED_CROSS}" '
                f'stroke-width="1.1" stroke-linecap="round" pointer-events="none"/>')
        parts.append("</g>")
        pts.append("".join(parts))

    n_unlabeled = sum(1 for c in base_cats if _is_unlabeled(c))
    legend = first["legend"]

    toggle = (f'<label class="tgl"><input type="checkbox" id="unl" checked> '
              f'show unlabeled <b>{n_unlabeled}</b></label>') if n_unlabeled else ""
    picker = ""
    if len(schemes) > 1:
        opts = "".join(f'<option value="{escape(n, quote=True)}">{escape(n)} '
                       f'({payloads[n]["n_cats"]})</option>' for n in names)
        picker = (f'<label class="tgl">colour by <select id="scheme">{opts}</select></label>')

    hl_note = (f' · <b>{n_highlighted}</b> highlighted'
               if highlights else "")
    if highlights and n_highlighted < len(highlights):
        hl_note += f' ({len(highlights) - n_highlighted} not found)'

    mid = meta.get("model_id", "?")
    pooling = meta.get("pooling", "?")
    subtitle = (f"{len(rows)} documents · pooling <b>{escape(pooling)}</b> · "
                f"projection <b>{method}</b> · colour by <b id=\"cdesc\">{escape(names[0])}</b>"
                f"{hl_note} · model <code>{escape(str(mid))}</code>")
    schemes_json = json.dumps({n: {"colors": p["colors"], "cats": p["cats"],
                                   "legend": p["legend"]} for n, p in payloads.items()})
    labels_checked = " checked" if show_labels else ""
    theme_checked = " checked" if theme == "dark" else ""
    body_class = theme

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>mole embedding scatter</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 20px; }}
  body.light {{ background: #f8f9fa; color: #111; }}
  body.dark {{ background: #1a1d21; color: #e8eaed; }}
  h1 {{ font-size: 17px; margin: 0 0 2px; }}
  .sub {{ opacity: .75; margin-bottom: 12px; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center;
               margin-bottom: 10px; }}
  .wrap {{ display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }}
  svg {{ border-radius: 8px; max-width: 100%; height: auto; }}
  svg.light {{ background: #fff; border: 1px solid #0002; }}
  svg.dark {{ background: #252830; border: 1px solid #fff2; }}
  .dot:hover {{ fill-opacity: 1; }}
  .legend {{ max-width: 320px; display: flex; flex-wrap: wrap; gap: 4px 12px;
             align-content: flex-start; }}
  .lg {{ white-space: nowrap; opacity: .9; }}
  .lg i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px;
           margin-right: 5px; vertical-align: baseline; position: relative; }}
  .lg i.xm::after {{ content: "×"; position: absolute; inset: -1px 0 0 0; color: #2f3336;
                     font-size: 11px; line-height: 10px; text-align: center; font-weight: 700; }}
  body.dark .lg i.xm::after {{ color: #cbd0d6; }}
  .lg b {{ opacity: .55; font-weight: 500; }}
  .more {{ opacity: .6; font-style: italic; }}
  .tgl {{ display: inline-flex; align-items: center; gap: 6px;
          cursor: pointer; user-select: none; opacity: .85; white-space: nowrap; }}
  .tgl b {{ opacity: .55; font-weight: 500; }}
  .tgl input[type=range] {{ width: 110px; vertical-align: middle; }}
  .lg.unl.off {{ opacity: .3; text-decoration: line-through; }}
  code {{ font-size: 12px; opacity: .8; }}
  .dot {{ cursor: crosshair; }}
  #tt {{ position: fixed; z-index: 10; display: none; pointer-events: none;
         padding: 5px 8px; border-radius: 6px; font-size: 12px; max-width: 340px;
         box-shadow: 0 2px 8px #0006; }}
  body.light #tt {{ background: #111d; color: #fff; }}
  body.dark #tt {{ background: #fffd; color: #111; }}
  #tt b {{ font-weight: 600; }} #tt span {{ opacity: .7; }}
</style></head><body class="{body_class}">
<h1>Document embedding scatter</h1>
<div class="sub">{subtitle}</div>
<div class="controls">
  <label class="tgl"><input type="checkbox" id="theme"{theme_checked}> dark mode</label>
  <label class="tgl"><input type="checkbox" id="labels"{labels_checked}> show class IDs</label>
  <label class="tgl">point size <input type="range" id="psize" min="3" max="24"
         step="0.5" value="{point_size:.1f}"></label>
  {toggle}{picker}
</div>
<div class="wrap">
  <svg class="{body_class}" viewBox="0 0 {W} {H}" width="{W}" height="{H}">{''.join(pts)}</svg>
  <div class="legend">{legend}</div>
</div>
<div id="tt"></div>
<p class="sub">Hover a point for its filename. Ringed points are explicitly highlighted targets.
Toggle <b>show class IDs</b> to print the active colour category inside each circle (hand-group
numbers, cluster ids, …). Same-hand documents should form neighbourhoods as the model learns.
Crosses mark documents with no ground truth.</p>
<script>
(function() {{
  var svg = document.querySelector('svg'), body = document.body, tt = document.getElementById('tt');
  var SCHEMES = {schemes_json};
  var active = {json.dumps(names[0])};
  function textOn(hex) {{
    var h = hex.replace('#','');
    if (h.length !== 6) return '#111';
    var r = parseInt(h.slice(0,2),16), g = parseInt(h.slice(2,4),16), b = parseInt(h.slice(4,6),16);
    var lum = 0.299*r + 0.587*g + 0.114*b;
    return lum > 140 ? '#111' : '#fff';
  }}
  function labelText(cat) {{
    var s = String(cat || '').trim().toLowerCase();
    var bad = {{unlabeled:1,'-1':1,'—':1,'-':1,'none':1,nan:1,unknown:1,na:1,'n/a':1,'?':1}};
    if (!s || bad[s]) return '';
    return String(cat).length > 8 ? String(cat).slice(0,7)+'…' : String(cat);
  }}
  function size() {{ return parseFloat(document.getElementById('psize').value); }}
  function applySize() {{
    var r = size(), ring = r + 6, fs = Math.max(6, r * 0.85);
    document.querySelectorAll('.pt').forEach(function(g) {{
      var dot = g.querySelector('.dot');
      if (!dot) return;
      var x = parseFloat(dot.getAttribute('cx')), y = parseFloat(dot.getAttribute('cy'));
      dot.setAttribute('r', r);
      var ringEl = g.querySelector('.hl-ring');
      if (ringEl) ringEl.setAttribute('r', ring);
      var hl = g.querySelector('.hl-lbl');
      if (hl) hl.setAttribute('x', x + ring + 4);
      var lbl = g.querySelector('.lbl');
      if (lbl) {{ lbl.setAttribute('font-size', fs); lbl.setAttribute('x', x); lbl.setAttribute('y', y); }}
      var xmark = g.querySelector('.unl-x');
      if (xmark) {{
        var a = r * 0.48;
        xmark.setAttribute('d', 'M'+(x-a)+' '+(y-a)+'L'+(x+a)+' '+(y+a)+
                          'M'+(x-a)+' '+(y+a)+'L'+(x+a)+' '+(y-a));
      }}
    }});
  }}
  function paint(name) {{
    var s = SCHEMES[name];
    if (!s) return;
    active = name;
    document.querySelectorAll('.pt').forEach(function(g) {{
      var i = +g.getAttribute('data-i');
      var dot = g.querySelector('.dot');
      if (!dot) return;
      var col = s.colors[i];
      dot.setAttribute('fill', col);
      var lbl = g.querySelector('.lbl');
      if (lbl) {{
        var t = labelText(s.cats[i]);
        lbl.textContent = t;
        lbl.setAttribute('fill', textOn(col));
        lbl.setAttribute('visibility', t && document.getElementById('labels').checked
                         ? 'visible' : 'hidden');
      }}
    }});
    document.querySelector('.legend').innerHTML = s.legend;
    document.getElementById('cdesc').textContent = name;
    tt.style.display = 'none';
  }}
  var picker = document.getElementById('scheme');
  if (picker) picker.addEventListener('change', function() {{ paint(picker.value); }});
  svg.addEventListener('mouseover', function(e) {{
    var t = e.target;
    if (t.classList && t.classList.contains('dot')) {{
      tt.innerHTML = '<b></b><br><span></span>';
      tt.querySelector('b').textContent = t.getAttribute('data-name');
      tt.querySelector('span').textContent =
        '[' + SCHEMES[active].cats[+t.getAttribute('data-i')] + ']';
      tt.style.display = 'block';
    }}
  }});
  svg.addEventListener('mousemove', function(e) {{
    tt.style.left = (e.clientX + 14) + 'px';
    tt.style.top = (e.clientY + 14) + 'px';
  }});
  svg.addEventListener('mouseout', function(e) {{
    if (e.target.classList && e.target.classList.contains('dot')) tt.style.display = 'none';
  }});
  var unl = document.getElementById('unl');
  if (unl) {{
    var keys = document.querySelectorAll('.lg.unl');
    unl.addEventListener('change', function() {{
      var vis = unl.checked;
      document.querySelectorAll('.pt[data-unl="1"]').forEach(function(g) {{
        g.style.display = vis ? '' : 'none';
      }});
      for (var j = 0; j < keys.length; j++) keys[j].classList.toggle('off', !vis);
      tt.style.display = 'none';
    }});
  }}
  document.getElementById('theme').addEventListener('change', function(e) {{
    var dark = e.target.checked;
    body.className = dark ? 'dark' : 'light';
    svg.className = body.className;
  }});
  document.getElementById('labels').addEventListener('change', function(e) {{
    var vis = e.target.checked ? 'visible' : 'hidden';
    document.querySelectorAll('.lbl').forEach(function(t) {{
      if (t.textContent) t.setAttribute('visibility', vis);
    }});
  }});
  document.getElementById('psize').addEventListener('input', applySize);
  applySize();
  paint(active);
}})();
</script>
</body></html>"""


# -------------------------------------------------------------------------- api
def _cluster_schemes(clusters: str | Path, rows: list[dict]) -> list[tuple[str, list[str]]]:
    """Colour schemes from a ``mole cluster`` report, one per FINCH level.

    Alignment is checked against the embedding rows: a clusters file produced from a
    different embedding would silently mis-colour every point, so a mismatch raises
    rather than rendering something plausible-looking and wrong.
    """
    report = json.loads(Path(clusters).read_text())
    imgs = report.get("images", [])
    if len(imgs) != len(rows) or any(a != str(b["image"]) for a, b in zip(imgs, rows)):
        raise ValueError(
            f"{clusters} was computed from a different embedding "
            f"({len(imgs)} rows vs {len(rows)}) — regenerate it with `mole cluster`")
    schemes = []
    for lv in report["levels"]:
        name = f"FINCH L{lv['level']} ({lv['n_clusters']} clusters)"
        schemes.append((name, [f"c{v}" for v in lv["labels"]]))
    return schemes


def plot_embeddings(embeddings: str | Path, out: str | Path | None = None,
                    method: str = "auto", color: str = "dataset",
                    color_regex: str | None = None, seed: int = 0,
                    pca_dim: int = 150, pca_whiten: bool = True,
                    umap_neighbors: int = 15, umap_min_dist: float = 0.1,
                    clusters: str | Path | None = None,
                    highlight: list[str] | None = None,
                    highlight_file: str | Path | None = None,
                    theme: str = "light",
                    show_labels: bool = False,
                    point_size: float = 4.5) -> tuple[Path, str]:
    """Write a lightweight self-contained SVG scatter (no images, no Bokeh).

    This is the dependency-free helper used programmatically and as the ``svg``
    fallback. The user-facing ``mole viz`` command renders the full map + charter
    viewer bipanel through :func:`mole.review.render.render_review` (same renderer
    as ``mole review``), so the two commands share one interactive interface.

    Returns ``(output_path, method_used)``. Default projection is PCA(``pca_dim``)
    → UMAP. ``color`` is ``dataset`` | ``hand`` | ``none``; ``color_regex`` overrides
    it. ``clusters`` adds one switchable colour scheme per FINCH level. ``highlight``
    / ``highlight_file`` ring specific documents (Sluis target pattern).
    """
    embeddings = Path(embeddings)
    X, meta, rows = _load_embeddings(embeddings)
    coords, used = reduce_2d(X, method, seed, pca_dim=pca_dim, pca_whiten=pca_whiten,
                            umap_neighbors=umap_neighbors, umap_min_dist=umap_min_dist)
    cats = _categories(rows, color, color_regex)
    color_desc = f"regex {color_regex}" if color_regex else color
    schemes = [(color_desc, cats)]
    if clusters:
        schemes.extend(_cluster_schemes(clusters, rows))
    highlights = _parse_highlights(highlight, highlight_file)
    html = _build_html(coords, schemes, rows, meta, used,
                       highlights=highlights, theme=theme,
                       show_labels=show_labels, point_size=point_size)
    out = Path(out) if out else embeddings.with_suffix(".viz.html")
    out.write_text(html, encoding="utf-8")
    return out, used
