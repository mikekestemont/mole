"""Visualise what a VLAD codebook learned.

A VLAD codebook (``<embeddings>.codebook.npy``) is ``[K, dim]`` k-means centres over
the model's foreground ViT patch tokens — a *visual vocabulary* of local ink motifs.
A centre is a 384-d vector, not directly viewable; it only means something through the
patches assigned to it. This module reconstructs that assignment and renders the four
standard bag-of-visual-words diagnostics into one offline HTML page:

1. **Nearest-patch montages** — for each word, the real image patches closest to the
   centre. The canonical "what did this word learn" figure (ascenders, loops, ...).
2. **Per-window assignment mosaics** — a window beside a grid coloured by the word each
   patch fell into: which word fires where.
3. **Occupancy histogram** — patches per word; reveals dead / dominant words.
4. **Codebook geometry** — a 2-D projection of the K centres (shared colour identity
   with 1–3) plus the K×K cosine-similarity matrix: vocabulary structure and near-
   duplicate words.

The patch descriptors are *contextualised* ViT tokens, so a montage shows the pixel
region while the grouping reflects attention-influenced appearance — interpretable in
practice, but worth stating.
"""

from __future__ import annotations

import base64
import colorsys
import heapq
import io
import json
from html import escape
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- colour
def _hsl_hex(h: float, s: float, l: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, max(0.0, min(1.0, l)), max(0.0, min(1.0, s)))
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _word_colors(coords: np.ndarray) -> list[str]:
    """One stable colour per word from its 2-D position, so montages, mosaics, the
    histogram and the scatter all speak the same colour language."""
    xy = np.asarray(coords, dtype=np.float64)

    def norm(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo or 1.0)

    nx, ny = norm(xy[:, 0]), norm(xy[:, 1])
    return [_hsl_hex(0.02 + 0.92 * nx[i], 0.62, 0.40 + 0.30 * ny[i]) for i in range(len(xy))]


# ------------------------------------------------------------------- encoding
def _png_data_uri(arr: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), "RGB").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _crop_patch(win_img, p: int, grid: int, patch_size: int, out_px: int) -> np.ndarray:
    """Pixel crop of raster patch ``p`` from a resized (model_size) window, upscaled."""
    from PIL import Image

    row, col = divmod(int(p), grid)
    box = (col * patch_size, row * patch_size, (col + 1) * patch_size, (row + 1) * patch_size)
    crop = win_img.crop(box).resize((out_px, out_px), Image.BICUBIC)
    return np.asarray(crop.convert("RGB"), dtype=np.uint8)


# --------------------------------------------------------------------- core
def _collect(embeddings: Path, checkpoint: str | Path | None, *, per_word: int,
             patch_px: int, max_pages: int, mosaic_windows: int, device: str | None):
    """Run the backbone over (a sample of) the source pages and build, per word, a
    bounded set of nearest patches, plus occupancy counts and a few mosaics.

    Returns ``(codebook, counts, montages, mosaics, meta)``.
    """
    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    from mole.data.patches import load_rgb
    from mole.embed.extract import (
        _build_transform,
        _foreground_mask,
        _page_index,
        _page_tokens,
        _pick_device,
        load_backbone,
    )
    from mole.embed.pooling import patch_descriptors

    sidecar = json.loads(embeddings.with_suffix(".mapping.json").read_text())
    if sidecar.get("pooling") != "vlad":
        raise ValueError(f"{embeddings.name} is pooling={sidecar.get('pooling')}, not vlad — "
                         "there is no codebook to visualise.")
    cb_path = embeddings.with_suffix(".codebook.npy")
    if not cb_path.exists():
        raise FileNotFoundError(f"codebook not found next to embeddings: {cb_path}")
    codebook = np.load(cb_path).astype(np.float32)
    K, dim = codebook.shape

    checkpoint = checkpoint or sidecar.get("checkpoint")
    if not checkpoint or not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"model checkpoint not found ({checkpoint!r}); pass --checkpoint explicitly.")

    dev = torch.device(device) if device else _pick_device()
    model, meta = load_backbone(checkpoint, map_location=str(dev))
    if meta["embed_dim"] != dim:
        raise ValueError(f"checkpoint dim {meta['embed_dim']} != codebook dim {dim}")

    invert = bool(sidecar.get("invert", meta.get("invert", False)))
    fg_method = sidecar.get("foreground_method", "contrast")
    fg_thr = float(sidecar.get("foreground_threshold",
                               0.05 if fg_method == "contrast" else 0.02))
    ws, ov, uz = meta["window_size"], meta["overlap"], meta["use_zones"]
    model_size, patch_size, nct = meta["model_size"], meta["patch_size"], meta["num_class_tokens"]
    grid = model_size // patch_size
    batch = 32

    # source images: prefer the rows recorded in the sidecar (exact set embedded)
    row_imgs = [r["image"] for r in sidecar.get("rows", []) if r.get("image")]
    if row_imgs:
        seen: set[str] = set()
        page_paths = []
        for p in row_imgs:
            if p not in seen and Path(p).exists():
                seen.add(p)
                page_paths.append(Path(p))
        # windows per page, mirroring extraction geometry
        from mole.data.patches import window_coords
        from mole.data.zones import find_zones, load_zones
        pages = []
        zcache: dict = {}
        for p in page_paths:
            folder = p.parent
            if uz and folder not in zcache:
                zp = find_zones(folder)
                zcache[folder] = load_zones(zp) if zp else None
            manifest = zcache.get(folder) if uz else None
            from PIL import Image
            bbox = manifest.bbox_for(p.name) if manifest else None
            size = (manifest.images[p.name].size
                    if (manifest and p.name in manifest.images) else Image.open(p).size)
            pages.append((p, window_coords(size[0], size[1], ws, ov, bbox)))
    else:
        pages = _page_index(Path(sidecar["rows"][0]["image"]).parent, ws, ov, uz)

    rng = np.random.default_rng(0)
    if max_pages and len(pages) > max_pages:
        idx = rng.choice(len(pages), max_pages, replace=False)
        pages = [pages[i] for i in sorted(idx)]

    resize = transforms.Resize((model_size, model_size),
                               interpolation=InterpolationMode.BICUBIC, antialias=True)
    to_tensor = _build_transform(model_size)
    csq = (codebook * codebook).sum(1)

    heaps: list[list] = [[] for _ in range(K)]          # per word: bounded max-heap on dist
    counts = np.zeros(K, dtype=np.int64)
    mosaics: list[dict] = []
    ctr = 0

    from mole.progress import track
    for pi, (img, wins) in enumerate(track(pages, "Scanning pages", unit="page")):
        if not wins:
            continue
        page = load_rgb(img, invert=invert)
        pil_wins = [resize(page.crop((w.x, w.y, w.x + w.size, w.y + w.size))) for w in wins]
        crops = [to_tensor(page.crop((w.x, w.y, w.x + w.size, w.y + w.size))) for w in wins]
        tokens = _page_tokens(model, crops, dev, batch)
        desc = patch_descriptors(tokens, nct).numpy().astype(np.float32)   # [W,P,dim]
        fg = _foreground_mask(crops, patch_size, fg_thr, fg_method).numpy()  # [W,P] bool
        W, P, _ = desc.shape
        for w in range(W):
            dvec = desc[w]                                                 # [P,dim]
            dist = (dvec * dvec).sum(1)[:, None] + csq[None, :] - 2.0 * dvec @ codebook.T
            assign = dist.argmin(1)
            mindist = dist[np.arange(P), assign]
            fgw = fg[w]
            counts += np.bincount(assign[fgw], minlength=K)
            for p in np.where(fgw)[0]:
                wid = int(assign[p]); d = float(mindist[p]); h = heaps[wid]
                if len(h) < per_word:
                    patch = _crop_patch(pil_wins[w], p, grid, patch_size, patch_px)
                    heapq.heappush(h, (-d, ctr, patch)); ctr += 1
                elif -h[0][0] > d:
                    patch = _crop_patch(pil_wins[w], p, grid, patch_size, patch_px)
                    heapq.heapreplace(h, (-d, ctr, patch)); ctr += 1
            if len(mosaics) < mosaic_windows and fgw.mean() > 0.15:
                mosaics.append(dict(
                    img=_png_data_uri(np.asarray(pil_wins[w].convert("RGB"), np.uint8)),
                    assign=[int(a) if fgw[j] else -1 for j, a in enumerate(assign)],
                    grid=grid, name=Path(img).name))

    montages = []
    for k in range(K):
        patches = [p for _, _, p in sorted(heaps[k], key=lambda t: -t[0])]  # closest first
        montages.append(patches)

    prov = dict(model_id=meta["model_id"], checkpoint=str(checkpoint), K=K, dim=dim,
                pages=len(pages), invert=invert, foreground=f"{fg_method}>{fg_thr:g}",
                seed=int(sidecar.get("vlad_seed", 0)), source=embeddings.name,
                total_patches=int(counts.sum()))
    return codebook, counts, montages, mosaics, prov


# ------------------------------------------------------------------ rendering
def _montage_uri(patches: list[np.ndarray], cols: int, cell: int) -> str:
    """Tile up to ``cols*rows`` patches into one PNG data URI (fewer <img> tags)."""
    from PIL import Image

    if not patches:
        return ""
    rows = int(np.ceil(len(patches) / cols))
    canvas = np.full((rows * cell, cols * cell, 3), 255, np.uint8)
    for i, p in enumerate(patches):
        r, c = divmod(i, cols)
        tile = p if p.shape[0] == cell else np.asarray(
            Image.fromarray(p.astype(np.uint8)).resize((cell, cell), Image.BICUBIC))
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = tile
    return _png_data_uri(canvas)


def _svg_histogram(counts: np.ndarray, colors: list[str], width: int = 900,
                   height: int = 220) -> str:
    K = len(counts)
    order = np.argsort(counts)[::-1]
    mx = float(counts.max() or 1)
    bw = width / K
    bars = []
    for rank, k in enumerate(order):
        h = (counts[k] / mx) * (height - 24)
        x = rank * bw
        dead = counts[k] == 0
        bars.append(
            f'<rect x="{x:.2f}" y="{height - 20 - h:.1f}" width="{max(1, bw - 0.6):.2f}" '
            f'height="{h:.1f}" fill="{"#d0d4dc" if dead else colors[k]}">'
            f'<title>word {k} · {int(counts[k]):,} patches</title></rect>')
    n_dead = int((counts == 0).sum())
    note = (f'<text x="0" y="{height - 4}" font-size="11" fill="#5a6070">'
            f'{K} words, largest {int(counts.max()):,} · smallest {int(counts.min()):,}'
            f'{f" · {n_dead} empty" if n_dead else ""} (bars sorted by usage)</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet">{"".join(bars)}{note}</svg>')


def _svg_scatter(coords: np.ndarray, counts: np.ndarray, colors: list[str],
                 size: int = 460) -> str:
    xy = np.asarray(coords, np.float64)

    def norm(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo or 1.0)

    pad = 22
    nx = norm(xy[:, 0]) * (size - 2 * pad) + pad
    ny = (1 - norm(xy[:, 1])) * (size - 2 * pad) + pad
    mx = float(counts.max() or 1)
    out = []
    for k in range(len(xy)):
        r = 4 + 12 * (counts[k] / mx) ** 0.5
        out.append(
            f'<circle cx="{nx[k]:.1f}" cy="{ny[k]:.1f}" r="{r:.1f}" fill="{colors[k]}" '
            f'fill-opacity="0.85" stroke="#0003" stroke-width="0.5">'
            f'<title>word {k} · {int(counts[k]):,} patches</title></circle>'
            f'<text x="{nx[k]:.1f}" y="{ny[k]:.1f}" font-size="8" fill="#111" '
            f'text-anchor="middle" dominant-baseline="central">{k}</text>')
    return (f'<svg viewBox="0 0 {size} {size}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" style="max-width:{size}px">{"".join(out)}</svg>')


def _svg_simmatrix(codebook: np.ndarray, coords: np.ndarray, size: int = 460) -> str:
    c = codebook / np.clip(np.linalg.norm(codebook, axis=1, keepdims=True), 1e-12, None)
    S = c @ c.T
    xy = np.asarray(coords, np.float64)
    ang = np.arctan2(xy[:, 1] - xy[:, 1].mean(), xy[:, 0] - xy[:, 0].mean())
    order = np.argsort(ang)                       # adjacency reveals blocks of similar words
    S = S[np.ix_(order, order)]
    K = len(order)
    cell = size / K
    lo = float(S[~np.eye(K, dtype=bool)].min())
    rects = []
    for i in range(K):
        for j in range(K):
            t = (S[i, j] - lo) / (1.0 - lo or 1.0)
            t = max(0.0, min(1.0, t))
            shade = int(255 * (1 - t))
            rects.append(f'<rect x="{j * cell:.2f}" y="{i * cell:.2f}" '
                         f'width="{cell + 0.5:.2f}" height="{cell + 0.5:.2f}" '
                         f'fill="rgb({shade},{shade},{255 - int(120 * t)})"/>')
    return (f'<svg viewBox="0 0 {size} {size}" width="100%" '
            f'preserveAspectRatio="xMidYMid meet" style="max-width:{size}px">'
            f'{"".join(rects)}</svg>')


def _mosaic_html(m: dict, colors: list[str], px: int = 220) -> str:
    grid = m["grid"]
    cell = px / grid
    rects = []
    for p, a in enumerate(m["assign"]):
        row, col = divmod(p, grid)
        fill = "none" if a < 0 else colors[a]
        op = "0" if a < 0 else "0.72"
        rects.append(f'<rect x="{col * cell:.2f}" y="{row * cell:.2f}" '
                     f'width="{cell + 0.4:.2f}" height="{cell + 0.4:.2f}" '
                     f'fill="{fill}" fill-opacity="{op}"/>')
    return (f'<figure class="mosaic"><div class="mstack">'
            f'<img src="{m["img"]}" width="{px}" height="{px}">'
            f'<svg class="mover" viewBox="0 0 {px} {px}" width="{px}" height="{px}">'
            f'{"".join(rects)}</svg></div>'
            f'<figcaption>{escape(m["name"])}</figcaption></figure>')


def _build_html(codebook, counts, montages, mosaics, prov, colors, coords, *,
                cols: int, cell: int, theme: str) -> str:
    order = np.argsort(counts)[::-1]
    word_cards = []
    for k in order:
        uri = _montage_uri(montages[k], cols, cell)
        if not uri:
            body = '<div class="empty">empty word</div>'
        else:
            body = f'<img src="{uri}" class="mont">'
        word_cards.append(
            f'<div class="word"><div class="whead"><i style="background:{colors[k]}"></i>'
            f'word <b>{k}</b><span class="n">{int(counts[k]):,}</span></div>{body}</div>')

    mosaics_html = "".join(_mosaic_html(m, colors) for m in mosaics) or \
        '<div class="empty">no sufficiently inked window sampled</div>'
    body_class = "dark" if theme == "dark" else "light"
    sub = (f"{prov['K']} words · dim {prov['dim']} · {prov['total_patches']:,} foreground "
           f"patches over {prov['pages']} pages · {escape(prov['foreground'])}"
           f"{' · inverted' if prov['invert'] else ''} · model "
           f"<code>{escape(prov['model_id'])}</code>")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLAD codebook — {escape(prov['source'])}</title>
<style>
 :root{{--bg:#f4f5f7;--panel:#fff;--line:#d7dae1;--fg:#14161a;--dim:#5a6070}}
 body.dark{{--bg:#12131a;--panel:#171922;--line:#2a2c39;--fg:#e8eaed;--dim:#9aa0b0}}
 *{{box-sizing:border-box}}
 body{{font:14px/1.5 system-ui,sans-serif;margin:0;padding:18px 22px;
   background:var(--bg);color:var(--fg)}}
 h1{{font-size:19px;margin:0 0 2px}} h2{{font-size:15px;margin:22px 0 8px}}
 .sub{{color:var(--dim);font-size:12.5px;margin-bottom:6px}} code{{font-size:11.5px}}
 .card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
 .two{{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}}
 .two>div{{flex:1 1 380px}} .cap{{font-size:12px;color:var(--dim);margin-top:4px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}}
 .word{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px}}
 .whead{{font-size:12px;display:flex;align-items:center;gap:5px;margin-bottom:5px}}
 .whead i{{width:11px;height:11px;border-radius:2px;display:inline-block}}
 .whead .n{{margin-left:auto;color:var(--dim)}}
 .mont{{width:100%;image-rendering:pixelated;border-radius:4px;display:block}}
 .empty{{color:var(--dim);font-style:italic;padding:14px;text-align:center}}
 .mosaics{{display:flex;gap:14px;flex-wrap:wrap}}
 .mosaic figcaption{{font-size:11px;color:var(--dim);margin-top:3px;text-align:center}}
 .mstack{{position:relative;width:220px;height:220px;border:1px solid var(--line);border-radius:6px;overflow:hidden}}
 .mstack img{{display:block;image-rendering:pixelated}}
 .mover{{position:absolute;inset:0}}
 .tgl{{float:right;font-size:12px;color:var(--dim);cursor:pointer;user-select:none}}
</style></head><body class="{body_class}">
<label class="tgl"><input type="checkbox" id="dk"{' checked' if theme == 'dark' else ''}> dark</label>
<h1>VLAD codebook — {escape(prov['source'])}</h1>
<div class="sub">{sub}</div>

<h2>1 · Visual words — nearest patches</h2>
<div class="sub">Each tile is the real ink closest to a word's centre (closest first),
sorted by how many patches use the word. This is what the codebook actually groups.</div>
<div class="grid">{''.join(word_cards)}</div>

<h2>2 · Where words fire — per-window assignment</h2>
<div class="sub">A sampled window beside its patch grid, each foreground patch tinted by
its word (background patches left clear). Same colours as the scatter below.</div>
<div class="mosaics">{mosaics_html}</div>

<h2>3 · Word usage &amp; 4 · codebook geometry</h2>
<div class="two">
  <div class="card"><b>Occupancy</b> — foreground patches per word
    {_svg_histogram(counts, colors)}
    <div class="cap">Flat-ish is healthy; a few tall bars or many empty words mean the
    vocabulary is under-used.</div></div>
  <div class="card"><b>Centroid map</b> — 2-D projection, size ∝ usage
    {_svg_scatter(coords, counts, colors)}
    <div class="cap">Nearby dots are similar words; this is the colour source everywhere
    on the page.</div></div>
  <div class="card"><b>Similarity K×K</b> — cosine between centres (ordered)
    {_svg_simmatrix(codebook, coords)}
    <div class="cap">Bright blocks off the diagonal are groups of near-duplicate words —
    candidates for a smaller K.</div></div>
</div>
<script>
 document.getElementById('dk').addEventListener('change',function(e){{
   document.body.className = e.target.checked ? 'dark' : 'light';
 }});
</script>
</body></html>"""


def visualize_codebook(embeddings: str | Path, *, checkpoint: str | Path | None = None,
                       out: str | Path | None = None, per_word: int = 24,
                       cols: int = 6, cell: int = 44, patch_px: int = 44,
                       max_pages: int = 250, mosaic_windows: int = 6,
                       method: str = "umap", seed: int = 0,
                       theme: str = "light", device: str | None = None) -> Path:
    """Render a VLAD codebook diagnostic report next to ``embeddings``.

    Returns the output HTML path. Reads ``<embeddings>.codebook.npy`` and the sidecar
    for geometry / foreground / invert, re-runs the backbone over the recorded pages
    (sampled to ``max_pages``), and writes montages + mosaics + histogram + geometry.
    """
    from mole.viz.scatter import reduce_2d

    embeddings = Path(embeddings)
    codebook, counts, montages, mosaics, prov = _collect(
        embeddings, checkpoint, per_word=per_word, patch_px=patch_px,
        max_pages=max_pages, mosaic_windows=mosaic_windows, device=device)
    coords, _ = reduce_2d(codebook, method, seed=seed,
                          pca_dim=min(50, codebook.shape[1]))
    colors = _word_colors(coords)
    html = _build_html(codebook, counts, montages, mosaics, prov, colors, coords,
                       cols=cols, cell=cell, theme=theme)
    out = Path(out) if out else embeddings.with_suffix(".codebook.viz.html")
    out.write_text(html, encoding="utf-8")
    return out
