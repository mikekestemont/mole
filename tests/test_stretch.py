"""Percentile stretch that runs before Sauvola in ``mole prep``."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from mole.prep.binarize import binarize_folder, binarize_image
from mole.prep.stretch import bbox_mask, is_bitonal, stretch_gray


def test_washed_page_maps_onto_ink_paper_range():
    # faint camera plate: parchment ~200, ink ~160 (span 40, well above MIN_SPAN)
    page = np.full((80, 120), 200, np.uint8)
    page[20:50, 15:105] = 160
    out, stats = stretch_gray(page)
    assert stats["applied"] == 1
    assert stats["span"] >= 8
    assert out[5, 5] >= 240          # parchment → near 255
    assert out[30, 60] <= 40         # ink → near 20


def test_bitonal_is_left_alone():
    page = np.full((60, 80), 255, np.uint8)
    page[10:40, 10:70] = 0
    out, stats = stretch_gray(page)
    assert is_bitonal(page)
    assert stats["applied"] == 0
    assert np.array_equal(out, page)


def test_near_bitonal_jpeg_is_skipped():
    page = np.full((50, 60), 255, np.uint8)
    page[5:45, 5:55] = 0
    page[0, 0] = 8
    page[1, 1] = 250
    assert is_bitonal(page)
    out, stats = stretch_gray(page)
    assert stats["applied"] == 0
    assert np.array_equal(out, page)


def test_mask_ignores_black_mount_for_percentiles():
    page = np.zeros((100, 120), np.uint8)          # black mount
    page[20:80, 20:100] = 200                      # parchment
    page[40:55, 30:90] = 160                       # ink
    mask = bbox_mask(page.shape, (20, 20, 100, 80))
    _, with_mask = stretch_gray(page, mask)
    _, whole = stretch_gray(page)
    assert with_mask["applied"] == 1
    assert with_mask["p2"] > 100                   # ink, not the mount
    assert whole["p2"] < 20                        # mount zeros set p2


def test_span_below_min_is_skipped():
    page = np.full((40, 50), 180, np.uint8)
    page[10:20, 10:40] = 176                       # span 4 < MIN_SPAN 8
    out, stats = stretch_gray(page)
    assert stats["applied"] == 0
    assert np.array_equal(out, page)


def test_binarize_image_still_black_on_white():
    a = np.full((80, 100), 200, np.uint8)
    a[30:50, 10:90] = 160
    out = np.asarray(binarize_image(Image.fromarray(a)))
    assert set(np.unique(out).tolist()) <= {0, 255}
    assert out[40, 50] == 0
    assert out[5, 5] == 255


def test_binarize_image_no_stretch_still_black_on_white():
    a = np.full((80, 100), 205, np.uint8)
    a[30:50, 10:90] = 45
    out = np.asarray(binarize_image(Image.fromarray(a), stretch=False))
    assert out[40, 50] == 0
    assert out[5, 5] == 255


def test_folder_qc_notes_stretch_on_and_off(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    a = np.full((80, 100), 205, np.uint8)
    a[30:50, 10:90] = 45
    Image.fromarray(a).save(src / "p.png")
    on, off = tmp_path / "on.html", tmp_path / "off.html"
    binarize_folder(src, tmp_path / "out-on", sample=None, qc_html=on)
    binarize_folder(src, tmp_path / "out-off", sample=None, qc_html=off, stretch=False)
    assert "stretch p2→20/p98→255" in on.read_text()
    assert "stretch off" in off.read_text()


def test_folder_uses_zone_bbox_not_the_mount(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    page = np.zeros((100, 120), np.uint8)
    page[20:80, 20:100] = 200
    page[40:55, 30:90] = 160
    Image.fromarray(page).save(src / "p.png")
    (src / "zones.json").write_text(json.dumps({
        "meta": {},
        "images": {"p.png": {"bbox": [20, 20, 100, 80], "size": [120, 100],
                             "fell_back": False, "detections": []}},
    }))
    recs = binarize_folder(src, tmp_path / "out", sample=None, qc_html=tmp_path / "qc.html")
    binary = np.asarray(Image.open(tmp_path / "out" / "p.png"))
    assert recs[0]["stretched"] is True
    assert binary[47, 60] == 0                     # ink → black
    assert binary[25, 40] == 255                   # parchment in the zone → white
    assert binary[2, 2] == 0                       # black mount stays dark
