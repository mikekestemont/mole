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


def test_carry_labels_rewrites_extension_to_png(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    _page(src, "a.jpg"); _page(src, "b.tif")
    (src / "labels.csv").write_text("filename,hand_id\na.jpg,H1\nb.tif,H2\n")
    out = tmp_path / "out"; qc = tmp_path / "qc.html"
    binarize_folder(src, out, sample=None, qc_html=qc)
    got = (out / "labels.csv").read_text()
    assert "a.png,H1" in got and "b.png,H2" in got   # extensions rewritten to match binarized files
    assert ".jpg" not in got and ".tif" not in got
    # and the labels now actually match the binarized images on the folder
    from mole.data.datasets import load_labels
    tbl = load_labels(out)
    assert tbl.hand_by_filename == {"a.png": "H1", "b.png": "H2"} and not tbl.orphan_rows


def test_carry_labels_noop_without_labels(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    _page(src, "a.jpg")
    out = tmp_path / "out"
    binarize_folder(src, out, sample=None, qc_html=tmp_path / "qc.html")
    assert not (out / "labels.csv").exists()          # nothing to carry, no empty file


def test_binarize_preview_writes_nothing(tmp_path):
    src = tmp_path / "in"; src.mkdir()
    for i in range(4):
        _page(src, f"p{i}.png")
    out = tmp_path / "out"; qc = tmp_path / "qc.html"
    recs = binarize_folder(src, out, sample=2, qc_html=qc)   # preview
    assert len(recs) == 2
    assert not out.exists() or not list(out.glob("*.png"))   # nothing written
    assert qc.is_file()
