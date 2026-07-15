"""Multi-frame / pyramidal TIFF handling in load_rgb (and thus prep/embed)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from mole.data.patches import load_rgb


def _make_multiframe_tiff(path, first=(60, 40), second=(400, 300)):
    """A TIFF whose frame 0 is a small thumbnail and frame 1 is the full image —
    exactly the shape that crashes OpenCV's imdecodemulti (mismatched frames)."""
    thumb = Image.new("RGB", first, "gray")
    full = Image.new("RGB", second, "white")
    thumb.save(path, format="TIFF", save_all=True, append_images=[full])


def test_load_rgb_picks_largest_frame(tmp_path):
    p = tmp_path / "pyramid.tif"
    _make_multiframe_tiff(p, first=(60, 40), second=(400, 300))
    img = load_rgb(p)                       # must not raise, must pick the full frame
    assert img.mode == "RGB"
    assert img.size == (400, 300)           # largest frame, not the (60,40) thumbnail


def test_load_rgb_single_frame_unchanged(tmp_path):
    p = tmp_path / "plain.tif"
    Image.new("RGB", (123, 45), "white").save(p, format="TIFF")
    assert load_rgb(p).size == (123, 45)


def test_heuristic_detector_survives_multiframe(tmp_path):
    """The heuristic detector now loads via load_rgb, so a multi-frame TIFF that
    would crash cv2 is handled (returns a list, no exception)."""
    from mole.prep.detect import get_detector

    p = tmp_path / "doc.tif"
    # frame 0 tiny thumbnail; frame 1 a page with a dark text block on white
    thumb = Image.new("RGB", (50, 50), "gray")
    full = np.full((300, 400, 3), 255, np.uint8)
    full[80:200, 60:340] = 20               # "ink" block
    Image.fromarray(thumb.__array__() if False else np.asarray(thumb)).save(
        p, format="TIFF", save_all=True, append_images=[Image.fromarray(full)])

    dets = get_detector("heuristic").detect(p)
    assert isinstance(dets, list)           # no crash on the mismatched frames
