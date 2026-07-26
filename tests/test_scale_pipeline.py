"""Scale normalization as wired into `mole prep` and `mole embed`."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from mole.prep import scale as S
from mole.prep.binarize import binarize_folder, resolve_scale_target
from test_prep_scale import page


def corpus(tmp_path, name, xheights, *, grey=True):
    """A folder of photographed-looking pages (light parchment, dark ink)."""
    folder = tmp_path / name
    folder.mkdir()
    for i, xh in enumerate(xheights):
        img = np.asarray(page(xh, 800, 640, seed=i)).astype(np.float32)
        if grey:                                  # parchment tone + ink, not pure bitonal
            img = 40.0 + img * 0.75
        Image.fromarray(img.astype(np.uint8), "L").save(folder / f"p{i}.png")
    return folder


# ------------------------------------------------------------------------- prep
def test_prep_collapses_scale_and_records_it(tmp_path):
    src = corpus(tmp_path, "raw", [10, 14, 20, 26, 14, 20])
    out = tmp_path / "bin"
    records = binarize_folder(src, out, qc_html=tmp_path / "qc.html",
                              normalize_scale="profile")

    assert len(records) == 6
    manifest = S.load_scale(out / S.SCALE_FILENAME)
    assert manifest.target and manifest.meta["method"] == "profile"
    assert manifest.meta["target_source"].startswith("auto")
    # the collapse itself: pages come in spread over an octave and leave together
    assert manifest.meta["spread_before"] > 0.25
    assert manifest.meta["spread_after"] < 0.10
    assert manifest.meta["rescaled"] >= 4

    for name, entry in manifest.images.items():
        assert (out / name).is_file()
        assert Image.open(out / name).size == tuple(entry.size)   # sizes are the written ones
        assert entry.module_out == pytest.approx(manifest.target, rel=0.15)


def test_prep_honours_an_explicit_target(tmp_path):
    src = corpus(tmp_path, "raw", [12, 18, 24])
    out = tmp_path / "bin"
    binarize_folder(src, out, normalize_scale="profile", target_module=30.0)
    manifest = S.load_scale(out / S.SCALE_FILENAME)
    assert manifest.target == 30.0
    assert manifest.meta["target_source"] == "given"
    assert manifest.meta["median_after"] == pytest.approx(30.0, rel=0.15)


def test_prep_without_normalization_is_unchanged(tmp_path):
    src = corpus(tmp_path, "raw", [12, 18])
    plain, scaled = tmp_path / "plain", tmp_path / "scaled"
    binarize_folder(src, plain)
    binarize_folder(src, scaled, normalize_scale="profile", target_module=30.0)
    assert not (plain / S.SCALE_FILENAME).exists()          # opt-in, nothing written
    assert Image.open(plain / "p0.png").size == Image.open(src / "p0.png").size
    assert Image.open(scaled / "p0.png").size != Image.open(plain / "p0.png").size


def test_prep_preview_writes_nothing(tmp_path):
    src = corpus(tmp_path, "raw", [12, 18, 24, 30])
    out = tmp_path / "bin"
    binarize_folder(src, out, sample=2, qc_html=tmp_path / "qc.html",
                    normalize_scale="profile", target_module=28.0)
    assert not out.exists() or not list(out.glob("*"))


def test_prep_qc_reports_the_collapse(tmp_path):
    src = corpus(tmp_path, "raw", [12, 24])
    qc = tmp_path / "qc.html"
    binarize_folder(src, tmp_path / "bin", qc_html=qc, normalize_scale="profile",
                    target_module=28.0)
    html = qc.read_text()
    assert "scale-normalized to 28.0px" in html
    assert "spread" in html and "module" in html


def test_prep_rejects_an_unknown_method(tmp_path):
    src = corpus(tmp_path, "raw", [12])
    with pytest.raises(ValueError, match="normalize_scale"):
        binarize_folder(src, tmp_path / "bin", normalize_scale="magic")


def test_resolve_scale_target_prefers_the_explicit_value(tmp_path):
    src = corpus(tmp_path, "raw", [12, 18])
    target, source = resolve_scale_target(src, 33.0, method="profile", max_side=None,
                                          window=25, k=0.2, sample=10)
    assert (target, source) == (33.0, "given")
    measured, source = resolve_scale_target(src, None, method="profile", max_side=None,
                                            window=25, k=0.2, sample=10)
    assert measured > 0 and source.startswith("auto")


def test_normalized_folder_is_recognised_as_already_scaled(tmp_path):
    src = corpus(tmp_path, "raw", [10, 16, 22, 28])
    out = tmp_path / "bin"
    binarize_folder(src, out, normalize_scale="profile", target_module=26.0)

    scaler = S.PageScaler(26.0, manifest=S.load_scale(out / S.SCALE_FILENAME))
    for path in sorted(out.glob("*.png")):
        result = scaler.rescale(Image.open(path), name=path.name)
        assert result.source == "manifest"           # answered from prep, nothing re-measured
        # a second pass has nothing left to do beyond the estimator's own accuracy
        assert result.factor == pytest.approx(1.0, abs=0.1)
    assert scaler.summary()["median_scale"] == pytest.approx(1.0, abs=0.05)


# ------------------------------------------------------------------------ embed
def test_codebook_target_travels_in_the_provenance(tmp_path):
    from mole.embed.extract import codebook_module_target

    cb = tmp_path / "cb.npy"
    np.save(cb, np.zeros((4, 8), np.float32))
    assert codebook_module_target(cb) is None                      # no sidecar
    assert codebook_module_target(None) is None

    sidecar = tmp_path / "cb.npy.json"
    sidecar.write_text(json.dumps({"clusters": 4}))
    assert codebook_module_target(cb) is None                      # sidecar, no target
    sidecar.write_text(json.dumps({"script_module_target": 24.5}))
    assert codebook_module_target(cb) == 24.5
    sidecar.write_text(json.dumps({"word_height_target": 31.0}))   # the plan's original name
    assert codebook_module_target(cb) == 31.0
    sidecar.write_text("{not json")
    assert codebook_module_target(cb) is None                      # never crash the embed


def test_embed_scaler_resolution_rules(tmp_path):
    from mole.embed.extract import _resolve_scaler

    cb = tmp_path / "cb.npy"
    (tmp_path / "cb.npy.json").write_text(json.dumps({"script_module_target": 24.0}))

    scaler, target = _resolve_scaler([tmp_path], codebook_from=None, scale_normalize=None,
                                     target_module=None)
    assert scaler is None and target is None                       # off without a target

    scaler, target = _resolve_scaler([tmp_path], codebook_from=cb, scale_normalize=None,
                                     target_module=None)
    assert scaler is not None and target == 24.0                   # the codebook carries it

    scaler, _ = _resolve_scaler([tmp_path], codebook_from=cb, scale_normalize=False,
                                target_module=None)
    assert scaler is None                                          # explicit opt-out wins

    scaler, target = _resolve_scaler([tmp_path], codebook_from=cb, scale_normalize=True,
                                     target_module=31.0)
    assert target == 31.0                                          # explicit target wins

    with pytest.raises(ValueError, match="needs a target"):
        _resolve_scaler([tmp_path], codebook_from=None, scale_normalize=True,
                        target_module=None)


def test_embed_scaler_picks_up_the_prep_manifest(tmp_path):
    from mole.embed.extract import _resolve_scaler

    src = corpus(tmp_path, "raw", [12, 20])
    out = tmp_path / "bin"
    binarize_folder(src, out, normalize_scale="profile", target_module=26.0)

    scaler, target = _resolve_scaler([out], codebook_from=None, scale_normalize=None,
                                     target_module=26.0)
    assert target == 26.0
    assert scaler.manifest is not None and len(scaler.manifest.images) == 2


def test_rescaled_pages_are_retiled(tmp_path):
    from mole.data.patches import window_coords
    from mole.embed.extract import PageEntry, _rescale_page

    src = corpus(tmp_path, "raw", [10])
    out = tmp_path / "bin"
    binarize_folder(src, out)                       # no normalization: page stays small
    path = next(out.glob("*.png"))
    img = Image.open(path)
    settings = {"window_size": 224, "overlap": 0.0}
    entry = PageEntry(path, window_coords(img.width, img.height, 224, 0.0), None)

    scaler = S.PageScaler(S.script_module(img) * 2)  # ask for twice the module
    page_out, windows = _rescale_page(scaler, img, entry, settings)
    assert page_out.size[0] == pytest.approx(img.size[0] * 2, rel=0.02)
    assert len(windows) > len(entry.windows)         # a bigger page is tiled into more windows
    assert all(w.x + w.size <= page_out.size[0] for w in windows)


def test_zone_is_carried_through_the_rescale(tmp_path):
    from mole.data.patches import window_coords
    from mole.embed.extract import PageEntry, _rescale_page

    src = corpus(tmp_path, "raw", [10])
    out = tmp_path / "bin"
    binarize_folder(src, out)
    path = next(out.glob("*.png"))
    img = Image.open(path)
    zone = (0, 0, img.width // 2, img.height)
    settings = {"window_size": 128, "overlap": 0.0}
    entry = PageEntry(path, window_coords(img.width, img.height, 128, 0.0, zone), zone)

    scaler = S.PageScaler(S.script_module(img) * 2)
    page_out, windows = _rescale_page(scaler, img, entry, settings)
    # windows stay inside the (also doubled) zone rather than spilling over the page
    assert max(w.x + w.size for w in windows) <= page_out.size[0] // 2 + 128
