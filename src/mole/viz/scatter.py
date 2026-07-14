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


def _palette(n: int) -> list[str]:
    if n <= 1:
        return ["#4c78a8"]
    return ["#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 0.62, 0.9)))
            for i in range(n)]


# -------------------------------------------------------------------------- html
def _build_html(coords, cats, rows, meta, method, color_desc) -> str:
    xs, ys = coords[:, 0].astype(float), coords[:, 1].astype(float)

    def norm(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo or 1.0)

    W = H = 900
    pad = 44
    nx = norm(xs) * (W - 2 * pad) + pad
    ny = (1.0 - norm(ys)) * (H - 2 * pad) + pad          # SVG y grows downward
    uniq = sorted(set(cats), key=lambda c: (-cats.count(c), c))
    cmap = dict(zip(uniq, _palette(len(uniq))))

    dots = []
    for x, y, c, r in zip(nx, ny, cats, rows):
        name = escape(Path(r["image"]).name, quote=True)
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{cmap[c]}" '
                    f'fill-opacity="0.82" stroke="#0003" stroke-width="0.5" '
                    f'data-name="{name}" data-cat="{escape(str(c), quote=True)}"/>')

    show = uniq[:60]
    legend = "".join(
        f'<span class="lg"><i style="background:{cmap[c]}"></i>{escape(str(c))} '
        f'<b>{cats.count(c)}</b></span>' for c in show)
    if len(uniq) > len(show):
        legend += f'<span class="lg more">+{len(uniq) - len(show)} more…</span>'

    mid = meta.get("model_id", "?")
    pooling = meta.get("pooling", "?")
    subtitle = (f"{len(rows)} documents · pooling <b>{escape(pooling)}</b> · "
                f"projection <b>{method}</b> · colour by <b>{escape(color_desc)}</b> · "
                f"model <code>{escape(str(mid))}</code>")

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
           margin-right: 5px; vertical-align: baseline; }}
  .lg b {{ opacity: .55; font-weight: 500; }}
  .more {{ opacity: .6; font-style: italic; }}
  code {{ font-size: 12px; opacity: .8; }}
  circle {{ cursor: crosshair; }}
  #tt {{ position: fixed; z-index: 10; display: none; pointer-events: none;
         background: #111d; color: #fff; padding: 5px 8px; border-radius: 6px;
         font-size: 12px; max-width: 340px; box-shadow: 0 2px 8px #0006; }}
  #tt b {{ font-weight: 600; }} #tt span {{ opacity: .7; }}
</style></head><body>
<h1>Document embedding scatter</h1>
<div class="sub">{subtitle}</div>
<div class="wrap">
  <svg viewBox="0 0 {W} {H}" width="{W}" height="{H}">{''.join(dots)}</svg>
  <div class="legend">{legend}</div>
</div>
<div id="tt"></div>
<p class="sub">Hover a point for its filename. Same-hand / same-scribe documents
should form neighbourhoods as the model learns.</p>
<script>
(function() {{
  var svg = document.querySelector('svg'), tt = document.getElementById('tt');
  svg.addEventListener('mouseover', function(e) {{
    var t = e.target;
    if (t.tagName === 'circle') {{
      tt.innerHTML = '<b></b><br><span></span>';
      tt.querySelector('b').textContent = t.getAttribute('data-name');
      tt.querySelector('span').textContent = '[' + t.getAttribute('data-cat') + ']';
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
}})();
</script>
</body></html>"""


# -------------------------------------------------------------------------- api
def plot_embeddings(embeddings: str | Path, out: str | Path | None = None,
                    method: str = "auto", color: str = "dataset",
                    color_regex: str | None = None, seed: int = 0,
                    pca_dim: int = 150) -> tuple[Path, str]:
    """Project an embeddings file to 2D and write an interactive HTML scatter.

    Returns ``(output_path, method_used)``. Default projection is PCA(``pca_dim``)
    → UMAP. ``color`` is ``dataset`` | ``hand`` | ``none``; ``color_regex`` overrides
    it, colouring by a capture group extracted from each filename (e.g. ``r'_(\\d{4})-'``
    to colour by year).
    """
    embeddings = Path(embeddings)
    X, meta, rows = _load_embeddings(embeddings)
    coords, used = reduce_2d(X, method, seed, pca_dim=pca_dim)
    cats = _categories(rows, color, color_regex)
    color_desc = f"regex {color_regex}" if color_regex else color
    html = _build_html(coords, cats, rows, meta, used, color_desc)
    out = Path(out) if out else embeddings.with_suffix(".viz.html")
    out.write_text(html, encoding="utf-8")
    return out, used
