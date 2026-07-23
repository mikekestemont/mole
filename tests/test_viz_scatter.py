"""Interactive scatter HTML features (highlights, labels, theme, point size)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mole.viz.scatter import (
    _is_highlighted,
    _label_text,
    _parse_highlights,
    _text_on,
    plot_embeddings,
)


def _fixture(tmp_path: Path, names: list[str]):
    npy = tmp_path / "emb.npy"
    X = np.random.default_rng(0).standard_normal((len(names), 8)).astype(np.float32)
    np.save(npy, X)
    rows = [{"row": i, "image": str(tmp_path / "imgs" / n)} for i, n in enumerate(names)]
    (tmp_path / "imgs").mkdir()
    for n in names:
        (tmp_path / "imgs" / n).write_bytes(b"x")
    sidecar = npy.with_suffix(".mapping.json")
    sidecar.write_text(json.dumps({"rows": rows, "model_id": "test", "pooling": "mean"}))
    labels = tmp_path / "imgs" / "labels.csv"
    labels.write_text("filename,hand_id\na.png,12\nb.png,34\nc.png,\n")
    return npy


def test_parse_highlights_stems_and_file(tmp_path):
    f = tmp_path / "targets.txt"
    f.write_text("# Sluis targets\nBüdingen1r\n332o\n\nGenois-1327a\n")
    got = _parse_highlights(["foo.png", " bar "], f)
    assert "foo" in got and "bar" in got and "332o" in got and "Genois-1327a" in got


def test_highlight_match_uses_stem():
    assert _is_highlighted("/data/leroy-bin/332o.png", {"332o"})
    assert not _is_highlighted("/data/leroy-bin/999o.png", {"332o"})


def test_label_text_and_contrast():
    assert _label_text("unlabeled") == ""
    assert _label_text("68") == "68"
    assert _text_on("#ffffff") == "#111"
    assert _text_on("#111111") == "#fff"


def test_plot_svg_includes_controls_and_highlights(tmp_path):
    npy = _fixture(tmp_path, ["a.png", "b.png", "c.png"])
    out, used = plot_embeddings(
        npy, out=tmp_path / "out.html", method="pca", color="hand",
        highlight=["b"], show_labels=True, theme="light", point_size=8)
    html = out.read_text(encoding="utf-8")
    assert used == "pca"
    assert 'class="hl-ring"' in html
    assert 'class="hl-lbl"' in html and "b" in html
    assert 'id="theme"' in html and 'body class="light"' in html
    assert 'id="labels"' in html and " checked" in html.split('id="labels"')[1][:20]
    assert 'id="psize"' in html
    assert 'class="lbl"' in html


def test_plot_defaults_light_no_labels(tmp_path):
    npy = _fixture(tmp_path, ["a.png", "b.png", "c.png"])
    out, _ = plot_embeddings(npy, out=tmp_path / "out.html", method="pca")
    html = out.read_text(encoding="utf-8")
    assert 'body class="light"' in html
    assert 'id="labels"' in html
    assert 'class="hl-ring"' not in html


def test_nearest_neighbors_cosine_excludes_self():
    from mole.review.render import _nearest_neighbors

    # three tight points near +x, one far near -x: the trio are each other's NN
    X = np.array([[1.0, 0.02], [1.0, 0.0], [0.98, 0.01], [-1.0, 0.0]], dtype=np.float32)
    nn = _nearest_neighbors(X, k=2)
    assert len(nn) == 4
    for i, row in enumerate(nn):
        assert i not in row
        assert len(row) == 2
    assert 3 not in nn[0]           # the far point is never a top-2 of the cluster
    assert nn[3][0] in (0, 1, 2)    # its closest is the least-far of the cluster


def test_review_svg_map_rings_highlights():
    """The shared review map (used by ``mole viz``) rings highlighted indices."""
    from mole.review.render import _svg

    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]], dtype=np.float32)
    svg = _svg(coords, ["#123456", "#abcdef", "#0f0f0f"],
               ["12", "34", "unlabeled"], ["a.png", "b.png", "c.png"],
               highlight_idx=[1])
    from mole.viz.scatter import _HIGHLIGHT_STROKE
    assert svg.count(_HIGHLIGHT_STROKE) >= 1
    assert "b.png" in svg


def test_umap_uses_sluis_style_precomputed():
    pytest.importorskip("umap")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 200)).astype(np.float32)
    coords, tag = __import__("mole.viz.scatter", fromlist=["reduce_2d"]).reduce_2d(
        X, "umap", seed=42, pca_dim=20)
    assert coords.shape == (40, 2)
    assert "precomputed" in tag
    assert "whiten" in tag
    assert "n=15" in tag
