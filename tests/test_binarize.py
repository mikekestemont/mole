"""Sauvola binarization for mole prep."""

from __future__ import annotations

import numpy as np
from PIL import Image

from mole.prep.binarize import binarize_folder, binarize_image, downscale_max_side


def _page(tmp_path, name="doc.png"):
    a = np.full((120, 160), 205, np.uint8)      # light "parchment"
    a[40:60, 20:140] = 45                        # a dark "ink" stroke
    p = tmp_path / name
    Image.fromarray(a).save(p)
    return p


def test_sauvola_black_on_white(tmp_path):
    out = np.asarray(binarize_image(Image.open(_page(tmp_path))))
    assert set(np.unique(out).tolist()) <= {0, 255}   # bitonal
    assert out[50, 80] == 0                            # ink -> black
    assert out[5, 5] == 255                            # background -> white


def test_binarize_folder_writes_and_qc(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    for i in range(3):
        _page(src, f"p{i}.png")
    out = tmp_path / "out"; qc = tmp_path / "qc.html"
    recs = binarize_folder(src, out, sample=None, qc_html=qc)
    assert len(recs) == 3
    assert sorted(p.name for p in out.glob("*.png")) == ["p0.png", "p1.png", "p2.png"]
    assert qc.is_file() and "Binarization QC" in qc.read_text()


def test_max_side_downscales_but_never_upsamples():
    big = Image.new("RGB", (800, 600))
    assert downscale_max_side(big, 400).size == (400, 300)   # longest side capped
    small = Image.new("RGB", (300, 200))
    assert downscale_max_side(small, 400).size == (300, 200)  # already small -> untouched
    assert downscale_max_side(big, 0).size == (800, 600)      # 0/None = off
    assert downscale_max_side(big, None).size == (800, 600)


def test_binarize_image_respects_max_side():
    src = Image.new("RGB", (800, 600))
    assert binarize_image(src, max_side=400).size == (400, 300)
    assert binarize_image(src).size == (800, 600)             # default = no cap


def test_binarize_preview_writes_nothing(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    for i in range(4):
        _page(src, f"p{i}.png")
    out = tmp_path / "out"; qc = tmp_path / "qc.html"
    recs = binarize_folder(src, out, sample=2, qc_html=qc)   # preview
    assert len(recs) == 2
    assert not out.exists() or not list(out.glob("*.png"))   # nothing written
    assert qc.is_file()
