"""The review sheet must be ONE file that survives being emailed.

These pin the properties a non-technical reviewer depends on: no external
references (it opens offline, with no folder beside it), a size cap that is
actually enforced, and language that does not present uncalibrated guesses as
facts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from mole.review.render import render_review

pytest.importorskip("PIL")


def _corpus(tmp_path: Path, n_hands=5, docs=4):
    """A small archive with REAL images, so the encoder has something to encode."""
    from PIL import Image

    rng = np.random.default_rng(0)
    ds = tmp_path / "arch1"
    ds.mkdir()
    dim = 24
    vecs, names, rows, lab = [], [], [], ["filename,hand_id"]
    for h in range(n_hands):
        center = rng.standard_normal(dim)
        for d in range(docs):
            name = f"h{h}_d{d}_x.png"
            # bilevel page, so the encoder exercises its lossless path
            arr = (rng.random((300, 220)) > 0.82).astype("uint8") * 255
            Image.fromarray(arr, mode="L").convert("1").save(ds / name)
            vecs.append(center + 0.05 * rng.standard_normal(dim))
            names.append(name)
            rows.append({"row": len(rows), "image": str(ds / name)})
            if h < n_hands - 1:                     # last hand stays unattributed
                lab.append(f"{name},H{h}")
    (ds / "labels.csv").write_text("\n".join(lab) + "\n")
    npy = tmp_path / "e.npy"
    np.save(npy, np.asarray(vecs, dtype=np.float32))
    (tmp_path / "e.mapping.json").write_text(
        json.dumps({"model_id": "t@0", "rows": rows}))
    return npy


def test_sheet_is_a_single_self_contained_file(tmp_path):
    npy = _corpus(tmp_path)
    out, summary = render_review(npy, out=tmp_path / "r.html", method="pca", limit=10)
    html = out.read_text()

    # nothing may be fetched: no http(s), no relative src/href, no <link>/<script src>
    external = re.findall(r'(?:src|href)\s*=\s*"(?!data:|#|file://)[^"\']+"', html)
    assert not external, external
    assert "<link" not in html
    assert "data:image/" in html                    # the charters really are inline
    assert "page images" in summary


def test_size_cap_is_enforced_not_merely_advertised(tmp_path):
    npy = _corpus(tmp_path, n_hands=6, docs=4)
    big, _ = render_review(npy, out=tmp_path / "big.html", method="pca", max_mb=0)
    small, summary = render_review(npy, out=tmp_path / "small.html", method="pca",
                                   max_mb=0.02)
    assert small.stat().st_size < big.stat().st_size
    assert small.stat().st_size < 1024 * 1024        # the cap really bit
    assert "omitted" in summary                      # ... and it said so


def test_no_images_mode_is_small_and_still_useful(tmp_path):
    npy = _corpus(tmp_path)
    out, _ = render_review(npy, out=tmp_path / "t.html", method="pca", images=False)
    html = out.read_text()
    assert "data:image/" not in html
    assert out.stat().st_size < 400 * 1024
    assert "Scribe review" in html                   # the lists are still there


def test_uncalibrated_lists_ask_questions_rather_than_assert(tmp_path):
    """Merges/splits/new hands have no ground truth — they must not sound certain."""
    npy = _corpus(tmp_path, n_hands=6, docs=5)
    out, _ = render_review(npy, out=tmp_path / "q.html", method="pca", images=False)
    payload = json.loads(re.search(r"var D = (\{.*?\}), decisions", out.read_text(),
                                   re.S).group(1))
    kinds = {s["kind"]: s for s in payload["sections"]}
    for kind in ("merges", "splits", "new_hands"):
        for row in kinds.get(kind, {}).get("rows", []):
            assert "?" in row["text"] or "may" in row["text"], (kind, row["text"])
    # ... while attributions are allowed to state a calibrated fact
    for row in kinds.get("attributions", {}).get("rows", []):
        assert "correct" in row["text"] or "confident" in row["text"] or \
               "No confidence" in row["text"]


def test_local_file_links_by_default_and_template_when_given(tmp_path):
    npy = _corpus(tmp_path)
    out, _ = render_review(npy, out=tmp_path / "a.html", method="pca", images=False)
    assert "file://" in out.read_text()
    out2, _ = render_review(npy, out=tmp_path / "b.html", method="pca", images=False,
                            image_url="https://arch.example/{filename}")
    assert "https://arch.example/h0_d0_x.png" in out2.read_text()


def test_colour_schemes_and_the_unlabeled_toggle_survive(tmp_path):
    """`mole viz`'s two affordances must exist here too: scheme switching and
    show/hide-unlabeled with the cross marker."""
    npy = _corpus(tmp_path, n_hands=5, docs=4)
    out, _ = render_review(npy, out=tmp_path / "s.html", method="pca", images=False)
    html = out.read_text()
    payload = json.loads(re.search(r"var D = (\{.*?\}), decisions = \{\}", html,
                                   re.S).group(1))

    assert payload["first"] == "hand"
    # hand + FINCH's discovered clusters, so ground truth can be flipped against
    # what the model found on the SAME projection
    assert len(payload["schemes"]) >= 2
    assert any("cluster" in n for n in payload["schemes"])
    for sc in payload["schemes"].values():
        assert len(sc["colors"]) == payload["schemes"][payload["first"]]["colors"].__len__()

    assert 'id="scheme"' in html                 # the picker
    assert 'id="unl"' in html                    # the show/hide toggle
    assert "data-unl=" in html                   # crossed-through unattributed dots
    assert "<path d=" in html                    # ... drawn as an actual cross
    # unlabeled keeps the neutral grey, so the palette is spent on real hands
    from mole.viz.scatter import _UNLABELED_GREY
    assert _UNLABELED_GREY in json.dumps(payload["schemes"]["hand"]["colors"])
