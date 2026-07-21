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

High-dimensional inputs (e.g. VLAD's ``K*dim``) are PCA-reduced to 50 dims before
t-SNE/UMAP, the standard denoise-then-embed recipe. Everything is seeded, so a run
is reproducible.
"""

from __future__ import annotations

import colorsys
import json
import re
from html import escape
from pathlib import Path

import numpy as np


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
    """Top-``k`` principal-component scores via SVD (no sklearn needed)."""
    Xc = X - X.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, vt.shape[0])
    return (Xc @ vt[:k].T).astype(np.float32)


def reduce_2d(X: np.ndarray, method: str = "auto", seed: int = 0,
              pca_dim: int = 150) -> tuple[np.ndarray, str]:
    """Project ``[N, D]`` to ``[N, 2]``; returns (coords, method_used).

    For the nonlinear backends the input is first PCA-reduced to ``pca_dim`` dims
    (the standard denoise-then-embed recipe) — the default pipeline is
    **PCA(150) → UMAP**. ``pca`` alone goes straight to 2 components.
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

    # linear denoise to pca_dim before the nonlinear embedding
    k = min(pca_dim, X.shape[1], max(2, X.shape[0] - 1))
    reduced = X.shape[1] > k
    pre = _pca(X, k) if reduced else X
    tag = f" (pca-{k})" if reduced else ""
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
        with warnings.catch_warnings():
            # a fixed random_state makes UMAP single-threaded (for reproducibility);
            # its "n_jobs overridden" notice is expected and harmless — silence it.
            warnings.filterwarnings("ignore", message=r".*n_jobs value.*overridden.*")
            coords = umap.UMAP(n_components=2, random_state=seed).fit_transform(pre)
        return coords, f"umap{tag}"
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
    legend = "".join(
        f'<span class="lg{" unl" if _is_unlabeled(c) else ""}">'
        f'<i class="{"xm" if _is_unlabeled(c) else ""}" style="background:{cmap[c]}"></i>'
        f'{escape(str(c))} <b>{cats.count(c)}</b></span>' for c in show)
    if len(uniq) > len(show):
        legend += f'<span class="lg more">+{len(uniq) - len(show)} more…</span>'
    return {"colors": [cmap[c] for c in cats], "cats": [str(c) for c in cats],
            "legend": legend, "n_cats": len(uniq)}


def _build_html(coords, schemes, rows, meta, method) -> str:
    """``schemes`` is ``[(name, cats), ...]``; the first is shown initially.

    Extra schemes (e.g. one per FINCH level) are switchable in the browser: the
    circles' fill is repainted from a per-scheme colour array, so ground truth and
    discovered clusters can be flipped in place on the SAME projection — which is the
    only honest way to ask "do the clusters recover the known hands?".
    """
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
    # "No ground truth" is a property of the DOCUMENT, not of the active colouring, so
    # the cross marker is fixed to the hand labels (scheme 0) and stays put while the
    # fill changes — under a cluster scheme an unlabeled point is still coloured by its
    # cluster, which is exactly the attribution question ("which cluster did it join?").
    base_cats = schemes[0][1]

    dots = []
    for i, (x, y, r) in enumerate(zip(nx, ny, rows)):
        name = escape(Path(r["image"]).name, quote=True)
        dot = (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{first["colors"][i]}" '
               f'fill-opacity="0.82" stroke="#0003" stroke-width="0.5" data-i="{i}" '
               f'data-name="{name}"/>')
        if _is_unlabeled(base_cats[i]):
            a = 2.2
            dot = (f'<g data-unl="1">{dot}<path d="M{x - a:.1f} {y - a:.1f}L{x + a:.1f} {y + a:.1f}'
                   f'M{x - a:.1f} {y + a:.1f}L{x + a:.1f} {y - a:.1f}" stroke="{_UNLABELED_CROSS}" '
                   f'stroke-width="1.1" stroke-linecap="round" pointer-events="none"/></g>')
        dots.append(dot)

    n_unlabeled = sum(1 for c in base_cats if _is_unlabeled(c))
    legend = first["legend"]

    toggle = (f'<label class="tgl"><input type="checkbox" id="unl" checked> '
              f'show unlabeled <b>{n_unlabeled}</b></label>') if n_unlabeled else ""
    picker = ""
    if len(schemes) > 1:
        opts = "".join(f'<option value="{escape(n, quote=True)}">{escape(n)} '
                       f'({payloads[n]["n_cats"]})</option>' for n in names)
        picker = (f'<label class="tgl">colour by <select id="scheme">{opts}</select></label>')

    mid = meta.get("model_id", "?")
    pooling = meta.get("pooling", "?")
    subtitle = (f"{len(rows)} documents · pooling <b>{escape(pooling)}</b> · "
                f"projection <b>{method}</b> · colour by <b id=\"cdesc\">{escape(names[0])}</b> · "
                f"model <code>{escape(str(mid))}</code>")
    schemes_json = json.dumps({n: {"colors": p["colors"], "cats": p["cats"],
                                   "legend": p["legend"]} for n, p in payloads.items()})

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>mole embedding scatter</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 20px;
          background: Canvas; color: CanvasText; }}
  h1 {{ font-size: 17px; margin: 0 0 2px; }}
  .sub {{ opacity: .75; margin-bottom: 12px; }}
  .wrap {{ display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }}
  svg {{ background: #fff; border: 1px solid #0002; border-radius: 8px;
         max-width: 100%; height: auto; }}
  circle:hover {{ r: 7; fill-opacity: 1; }}
  .legend {{ max-width: 320px; display: flex; flex-wrap: wrap; gap: 4px 12px;
             align-content: flex-start; }}
  .lg {{ white-space: nowrap; opacity: .9; }}
  .lg i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px;
           margin-right: 5px; vertical-align: baseline; position: relative; }}
  .lg i.xm::after {{ content: "×"; position: absolute; inset: -1px 0 0 0; color: #2f3336;
                     font-size: 11px; line-height: 10px; text-align: center; font-weight: 700; }}
  .lg b {{ opacity: .55; font-weight: 500; }}
  .more {{ opacity: .6; font-style: italic; }}
  .tgl {{ display: inline-flex; align-items: center; gap: 6px; margin-bottom: 10px;
          cursor: pointer; user-select: none; opacity: .85; }}
  .tgl b {{ opacity: .55; font-weight: 500; }}
  .lg.unl.off {{ opacity: .3; text-decoration: line-through; }}
  code {{ font-size: 12px; opacity: .8; }}
  circle {{ cursor: crosshair; }}
  #tt {{ position: fixed; z-index: 10; display: none; pointer-events: none;
         background: #111d; color: #fff; padding: 5px 8px; border-radius: 6px;
         font-size: 12px; max-width: 340px; box-shadow: 0 2px 8px #0006; }}
  #tt b {{ font-weight: 600; }} #tt span {{ opacity: .7; }}
</style></head><body>
<h1>Document embedding scatter</h1>
<div class="sub">{subtitle}</div>
{toggle}{picker}
<div class="wrap">
  <svg viewBox="0 0 {W} {H}" width="{W}" height="{H}">{''.join(dots)}</svg>
  <div class="legend">{legend}</div>
</div>
<div id="tt"></div>
<p class="sub">Hover a point for its filename. Same-hand / same-scribe documents
should form neighbourhoods as the model learns. Where a FINCH level is available,
switch the colouring to compare discovered clusters against the known hands —
crosses mark documents with no ground truth, whichever colouring is active.</p>
<script>
(function() {{
  var svg = document.querySelector('svg'), tt = document.getElementById('tt');
  var SCHEMES = {schemes_json};
  var active = {json.dumps(names[0])};
  var dots = svg.querySelectorAll('circle');
  function paint(name) {{
    var s = SCHEMES[name];
    if (!s) return;
    active = name;
    for (var i = 0; i < dots.length; i++) {{
      var k = +dots[i].getAttribute('data-i');
      dots[i].setAttribute('fill', s.colors[k]);
    }}
    document.querySelector('.legend').innerHTML = s.legend;
    document.getElementById('cdesc').textContent = name;
    tt.style.display = 'none';
  }}
  var picker = document.getElementById('scheme');
  if (picker) picker.addEventListener('change', function() {{ paint(picker.value); }});
  svg.addEventListener('mouseover', function(e) {{
    var t = e.target;
    if (t.tagName === 'circle') {{
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
    if (e.target.tagName === 'circle') tt.style.display = 'none';
  }});
  var unl = document.getElementById('unl');
  if (unl) {{
    var pts = svg.querySelectorAll('[data-unl]');   // the <g> wrapping disc + cross
    var keys = document.querySelectorAll('.lg.unl');
    unl.addEventListener('change', function() {{
      var vis = unl.checked;
      for (var i = 0; i < pts.length; i++) pts[i].style.display = vis ? '' : 'none';
      for (var j = 0; j < keys.length; j++) keys[j].classList.toggle('off', !vis);
      tt.style.display = 'none';
    }});
  }}
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
                    pca_dim: int = 150,
                    clusters: str | Path | None = None) -> tuple[Path, str]:
    """Project an embeddings file to 2D and write an interactive HTML scatter.

    Returns ``(output_path, method_used)``. Default projection is PCA(``pca_dim``)
    → UMAP. ``color`` is ``dataset`` | ``hand`` | ``none``; ``color_regex`` overrides
    it, colouring by a capture group extracted from each filename (e.g. ``r'_(\\d{4})-'``
    to colour by year). ``clusters`` adds one switchable colour scheme per FINCH level
    from a ``mole cluster`` report, so discovered clusters can be flipped against the
    ground-truth colouring on the same projection.
    """
    embeddings = Path(embeddings)
    X, meta, rows = _load_embeddings(embeddings)
    coords, used = reduce_2d(X, method, seed, pca_dim=pca_dim)
    cats = _categories(rows, color, color_regex)
    color_desc = f"regex {color_regex}" if color_regex else color
    schemes = [(color_desc, cats)]
    if clusters:
        schemes.extend(_cluster_schemes(clusters, rows))
    html = _build_html(coords, schemes, rows, meta, used)
    out = Path(out) if out else embeddings.with_suffix(".viz.html")
    out.write_text(html, encoding="utf-8")
    return out, used
