"""Script-scale normalization: the estimator, the resampler, and the prep wiring.

The property that matters is **equivariance**: resample a page by a known factor
and the measured module must move by exactly that factor. Everything else
(collapse onto a target, idempotence, provenance) follows from it.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mole.prep import scale as S


def page(xheight: int = 16, width: int = 720, height: int = 560, seed: int = 0) -> Image.Image:
    """A synthetic ruled page: lines of words built from minims and ascenders.

    Ink is spread through the x-height band the way real writing spreads it —
    a page whose ink sat on the baseline alone would have a script module of two
    pixels no matter how large the letters.
    """
    rng = np.random.default_rng(seed)
    a = np.full((height, width), 255, np.uint8)
    pitch = int(round(2.6 * xheight))
    stroke = max(1, round(xheight / 6))
    y = pitch
    while y + xheight < height - pitch // 2:
        x = int(0.06 * width)
        while x < 0.94 * width:
            for _ in range(int(rng.integers(3, 8))):        # letters of one word
                top = y - (xheight // 2 if rng.integers(0, 4) == 0 else 0)   # ascender?
                a[top:y + xheight, x:x + stroke] = 0                          # left minim
                a[y:y + xheight, x + int(0.55 * xheight):x + int(0.55 * xheight) + stroke] = 0
                mid = y + xheight // 2
                a[mid:mid + stroke, x:x + int(0.55 * xheight)] = 0            # cross-stroke
                x += int(0.85 * xheight)
            x += int(1.3 * xheight)                                            # word gap
        y += pitch
    return Image.fromarray(a, "L")


def rescaled(img: Image.Image, factor: float) -> Image.Image:
    return img.resize((round(img.width * factor), round(img.height * factor)), Image.LANCZOS)


# --------------------------------------------------------------------- estimator
def test_module_scales_with_the_page():
    base = page(16)
    m0 = S.script_module(base)
    assert m0 is not None
    for factor in (0.5, 0.75, 1.5, 2.0):
        m = S.script_module(rescaled(base, factor))
        assert m == pytest.approx(m0 * factor, rel=0.08), f"not equivariant at x{factor}"


def test_module_tracks_the_script_not_the_page_size():
    # same page dimensions, script twice as big -> module twice as big
    small = S.script_module(page(10, 720, 560))
    large = S.script_module(page(20, 720, 560))
    assert large == pytest.approx(2 * small, rel=0.15)


def test_pitch_and_confidence_are_reported():
    est = S.estimate_module(page(16))
    assert est and est.method == "profile"
    assert est.pitch == pytest.approx(2.6 * 16, rel=0.1)   # the generator's line pitch
    assert 0.0 < est.confidence <= 1.0
    assert est.module < est.pitch                           # the band is part of the period


def test_unmeasurable_pages_return_none():
    blank = Image.fromarray(np.full((400, 400), 255, np.uint8), "L")
    assert S.script_module(blank) is None                   # nothing to measure
    inked = Image.fromarray(np.zeros((400, 400), np.uint8), "L")
    assert S.script_module(inked) is None                   # a failed binarization, not a page
    noise = Image.fromarray((np.random.default_rng(0).random((400, 400)) > 0.9).astype(np.uint8) * 255, "L")
    assert S.script_module(noise) is None                   # no line rhythm


def test_polarity_is_auto_detected():
    normal = page(16)
    inverted = Image.fromarray(255 - np.asarray(normal), "L")
    assert S.script_module(inverted) == pytest.approx(S.script_module(normal), rel=0.02)


def test_zone_restricts_the_measurement():
    # writing in the top half only; a zone over the blank half cannot be measured
    p = np.asarray(page(16, 720, 560)).copy()
    p[280:] = 255
    img = Image.fromarray(p, "L")
    assert S.script_module(img, (0, 0, 720, 280)) is not None
    assert S.script_module(img, (0, 300, 720, 560)) is None


def test_word_method_is_available_and_validated():
    est = S.estimate_module(page(16), method="word")
    assert est and est.method == "word" and est.n_blobs > 15
    with pytest.raises(ValueError, match="scale method"):
        S.estimate_module(page(16), method="rlsa")


# --------------------------------------------------------------------- resampling
def test_scale_factor_is_clamped_and_guarded():
    assert S.scale_factor(20.0, 40.0) == pytest.approx(2.0)
    assert S.scale_factor(1.0, 100.0) == S.SCALE_CLAMP[1]      # implausible -> clipped
    assert S.scale_factor(100.0, 1.0) == S.SCALE_CLAMP[0]
    assert S.scale_factor(None, 30.0) == 1.0                    # unmeasured -> leave alone
    assert S.scale_factor(0.0, 30.0) == 1.0


def test_resample_skips_negligible_factors_and_can_rebinarize():
    img = page(16)
    assert S.resample(img, 1.0) is img
    assert S.resample(img, 1.0 + S.RESIZE_EPS / 2) is img
    up = S.resample(img, 2.0)
    assert up.size == (img.width * 2, img.height * 2)
    assert set(np.unique(np.asarray(S.resample(img, 0.5, rethreshold=True)))) <= {0, 255}


def test_normalization_collapses_a_mixed_scale_corpus():
    target = 24.0
    achieved = []
    for xheight in (8, 12, 16, 24, 32):
        img = page(xheight, 900, 700)
        module = S.script_module(img)
        out = S.resample(img, S.scale_factor(module, target), rethreshold=True)
        achieved.append(S.script_module(out))
    assert all(m == pytest.approx(target, rel=0.12) for m in achieved), achieved
    spread = (max(achieved) - min(achieved)) / np.median(achieved)
    assert spread < 0.15


def test_scaled_zone_follows_the_page():
    assert S.scaled_zone((10, 20, 110, 220), 2.0) == (20, 40, 220, 440)
    assert S.scaled_zone(None, 2.0) is None


# ---------------------------------------------------------------------- manifest
def test_manifest_round_trip_and_staleness_guard(tmp_path):
    manifest = S.ScaleManifest(meta=S.scale_meta(24.0, "profile", "given"))
    manifest.images["a.png"] = S.ScaleEntry(module=12.0, scale=2.0, size=(100, 80),
                                            pitch=30.0, confidence=0.6, module_out=23.5)
    manifest.images["b.png"] = S.ScaleEntry(module=None, scale=1.0, size=(50, 50))
    path = S.save_scale(tmp_path / S.SCALE_FILENAME, manifest)
    loaded = S.load_scale(path)

    assert loaded.target == 24.0
    assert loaded.module_for("a.png", size=(100, 80)) == 23.5   # measured after normalization
    assert loaded.module_for("a.png", size=(999, 80)) is None   # stale entry -> re-measure
    assert loaded.module_for("b.png", size=(50, 50)) is None    # never measurable
    assert loaded.module_for("missing.png") is None
    assert S.find_scale(tmp_path) == path
    assert S.find_scale(tmp_path / "nope") is None


def test_entry_predicts_the_current_module_when_not_re_measured():
    entry = S.ScaleEntry(module=10.0, scale=2.0, size=(10, 10))
    assert entry.current_module == 20.0


# -------------------------------------------------------------------- PageScaler
def test_page_scaler_normalizes_and_reports():
    scaler = S.PageScaler(24.0)
    out = scaler.rescale(page(12, 900, 700))
    assert out.changed and out.source == "measured"
    assert S.script_module(out.image) == pytest.approx(24.0, rel=0.12)
    summary = scaler.summary()
    assert summary["pages"] == 1 and summary["rescaled"] == 1
    assert summary["script_module_target"] == 24.0
    assert "resampled to module" in scaler.note()


def test_page_scaler_reuses_the_manifest_instead_of_measuring():
    img = page(16, 900, 700)
    manifest = S.ScaleManifest(meta=S.scale_meta(24.0, "profile", "given"))
    manifest.images["p.png"] = S.ScaleEntry(module=12.0, scale=1.0, size=img.size,
                                            module_out=12.0)
    scaler = S.PageScaler(24.0, manifest=manifest)
    out = scaler.rescale(img, name="p.png")
    assert out.source == "manifest" and out.module == 12.0
    assert out.factor == pytest.approx(2.0)          # trusted the manifest, not the pixels
    assert scaler.summary()["from_manifest"] == 1


def test_page_scaler_is_idempotent_on_already_normalized_pages():
    scaler = S.PageScaler(24.0)
    once = scaler.rescale(page(12, 900, 700))
    twice = S.PageScaler(24.0).rescale(once.image)
    assert not twice.changed and twice.image is once.image


def test_page_scaler_leaves_unmeasurable_pages_alone():
    blank = Image.fromarray(np.full((300, 300), 255, np.uint8), "L")
    scaler = S.PageScaler(24.0)
    out = scaler.rescale(blank)
    assert out.factor == 1.0 and out.image is blank and out.source == "unmeasurable"
    assert scaler.summary()["unmeasurable"] == 1


def test_page_scaler_rejects_a_nonsense_target():
    with pytest.raises(ValueError, match="positive"):
        S.PageScaler(0)
    with pytest.raises(ValueError, match="scale method"):
        S.PageScaler(24.0, method="magic")


# ---------------------------------------------------------------- corpus measure
def _corpus(tmp_path, name, xheights):
    folder = tmp_path / name
    folder.mkdir()
    for i, xh in enumerate(xheights):
        page(xh, 700, 560, seed=i).save(folder / f"p{i}.png")
    return folder


def test_measure_corpus_and_pooled_target(tmp_path):
    small = _corpus(tmp_path, "small", [10, 10, 11, 10])
    large = _corpus(tmp_path, "large", [20, 21, 20, 20])
    a = S.measure_corpus(small, progress=False)
    b = S.measure_corpus(large, progress=False)
    assert a.n_measured == 4 and a.n_failed == 0
    assert b.median == pytest.approx(2 * a.median, rel=0.15)
    assert a.spread < 0.15                                  # a uniform corpus is tight

    pooled, scans = S.corpus_target([small, large], progress=False)
    assert len(scans) == 2
    assert min(a.median, b.median) < pooled < max(a.median, b.median)


def test_measure_corpus_samples_deterministically(tmp_path):
    folder = _corpus(tmp_path, "many", [12] * 10)
    scan = S.measure_corpus(folder, sample=4, progress=False)
    assert scan.n_measured + scan.n_failed == 4
    assert list(scan.modules) == list(S.measure_corpus(folder, sample=4, progress=False).modules)


def test_measure_corpus_needs_images(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        S.measure_corpus(tmp_path / "empty", progress=False)
