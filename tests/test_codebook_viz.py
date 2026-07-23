"""Rendering of the VLAD codebook report (pure helpers — no model / torch)."""

from __future__ import annotations

import numpy as np

from mole.viz.codebook import (
    _build_html,
    _hsl_hex,
    _montage_uri,
    _svg_histogram,
    _svg_scatter,
    _svg_simmatrix,
    _word_colors,
)


def _fake(K=8, dim=6):
    rng = np.random.default_rng(0)
    codebook = rng.standard_normal((K, dim)).astype(np.float32)
    counts = np.arange(K, dtype=np.int64) * 3      # includes a zero (empty word)
    coords = rng.standard_normal((K, 2)).astype(np.float32)
    colors = _word_colors(coords)
    montages = [[np.full((44, 44, 3), (i * 20) % 255, np.uint8)
                 for _ in range(3)] for i in range(K)]
    montages[0] = []                                # empty word renders a placeholder
    mosaics = [dict(img="data:image/png;base64,AA", grid=4,
                    assign=[0, -1, 3, 2] * 4, name="page1.png")]
    prov = dict(model_id="vit_small@abc+step0", checkpoint="ck.pth", K=K, dim=dim,
                pages=5, invert=True, foreground="contrast>0.05", seed=0,
                source="leroy.ssl.npy", total_patches=123)
    return codebook, counts, montages, mosaics, prov, colors, coords


def test_hsl_hex_format():
    h = _hsl_hex(0.5, 0.6, 0.4)
    assert h.startswith("#") and len(h) == 7


def test_word_colors_one_per_word():
    coords = np.random.default_rng(1).standard_normal((10, 2))
    cols = _word_colors(coords)
    assert len(cols) == 10
    assert all(c.startswith("#") and len(c) == 7 for c in cols)


def test_svg_pieces_render():
    cb, counts, _, _, _, colors, coords = _fake()
    assert "<svg" in _svg_histogram(counts, colors)
    assert "word 0" in _svg_scatter(coords, counts, colors)   # <title> per point
    assert "<rect" in _svg_simmatrix(cb, coords)


def test_montage_uri_tiles_and_empty():
    assert _montage_uri([], 6, 44) == ""
    uri = _montage_uri([np.zeros((44, 44, 3), np.uint8)] * 5, 6, 44)
    assert uri.startswith("data:image/png;base64,")


def test_build_html_has_all_four_sections():
    cb, counts, montages, mosaics, prov, colors, coords = _fake()
    html = _build_html(cb, counts, montages, mosaics, prov, colors, coords,
                       cols=6, cell=44, theme="light")
    assert "nearest patches" in html
    assert "assignment" in html
    assert "Occupancy" in html
    assert "Similarity" in html
    assert 'body class="light"' in html
    assert "empty word" in html                     # the zero-usage word
    assert "leroy.ssl.npy" in html
