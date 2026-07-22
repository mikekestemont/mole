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


def _payload(path):
    return json.loads(re.search(r"var D = (\{.*?\}), decisions", path.read_text(),
                                re.S).group(1))


def test_sheet_is_a_single_self_contained_file(tmp_path):
    """The SVG build must contain no external reference of any kind."""
    npy = _corpus(tmp_path)
    out, summary = render_review(npy, out=tmp_path / "r.html", method="pca", limit=10,
                                 map_backend="svg")
    html = out.read_text()
    external = re.findall(r'(?:src|href)\s*=\s*"(?!data:|#|file://)[^"\']+"', html)
    assert not external, external
    assert "<link" not in html
    assert "data:image/" in html                    # the charters really are inline
    assert "page images" in summary


def test_bokeh_build_fetches_nothing_of_ours(tmp_path):
    """BokehJS is inlined, so the sheet still opens with no network.

    BokehJS's own bundle contains a jsdelivr URL for MathJax, which it fetches
    lazily ONLY for LaTeX labels — these figures have none, and a browser check of
    the built page recorded zero non-localhost requests. So the assertion is
    scoped to markup we emit rather than loosened to nothing.
    """
    pytest.importorskip("bokeh")
    npy = _corpus(tmp_path)
    out, _ = render_review(npy, out=tmp_path / "b.html", method="pca", limit=10,
                           map_backend="bokeh", max_mb=0)
    html = out.read_text()
    assert not re.findall(r'<script[^>]+src\s*=\s*"http', html)   # no fetched script tag
    assert not re.findall(r'<link[^>]+href\s*=\s*"http', html)    # no fetched stylesheet
    assert "data:image/webp" in html or "data:image/png" in html   # charters inline


def test_size_cap_is_enforced_not_merely_advertised(tmp_path):
    npy = _corpus(tmp_path, n_hands=6, docs=4)
    big, _ = render_review(npy, out=tmp_path / "big.html", method="pca", max_mb=0,
                           map_backend="svg")
    small, summary = render_review(npy, out=tmp_path / "small.html", method="pca",
                                   max_mb=0.02, map_backend="svg")
    assert small.stat().st_size < big.stat().st_size
    assert small.stat().st_size < 1024 * 1024        # the cap really bit
    assert "omitted" in summary                      # ... and it said so


def test_bokeh_overhead_is_charged_to_the_size_budget(tmp_path):
    """--max-mb must stay honest once ~4 MB of BokehJS shares the file."""
    pytest.importorskip("bokeh")
    npy = _corpus(tmp_path, n_hands=4, docs=4)
    with pytest.raises(RuntimeError, match="BokehJS"):
        render_review(npy, out=tmp_path / "x.html", method="pca",
                      map_backend="bokeh", max_mb=1.0)


def test_no_images_mode_is_small_and_still_useful(tmp_path):
    npy = _corpus(tmp_path)
    out, _ = render_review(npy, out=tmp_path / "t.html", method="pca", images=False,
                           map_backend="svg")
    assert not _payload(out)["images"]               # no charter was embedded
    assert out.stat().st_size < 400 * 1024
    assert "Scribe review" in out.read_text()        # the lists are still there


def test_uncalibrated_lists_ask_questions_rather_than_assert(tmp_path):
    """Merges/splits/new hands have no ground truth — they must not sound certain."""
    npy = _corpus(tmp_path, n_hands=6, docs=5)
    out, _ = render_review(npy, out=tmp_path / "q.html", method="pca", images=False,
                           map_backend="svg")
    kinds = {s["kind"]: s for s in _payload(out)["sections"]}
    for kind in ("merges", "splits", "new_hands"):
        for row in kinds.get(kind, {}).get("rows", []):
            assert "?" in row["text"] or "may" in row["text"], (kind, row["text"])
    for row in kinds.get("attributions", {}).get("rows", []):
        assert "correct" in row["text"] or "confident" in row["text"] or \
               "No confidence" in row["text"]


def test_local_file_links_by_default_and_template_when_given(tmp_path):
    npy = _corpus(tmp_path)
    out, _ = render_review(npy, out=tmp_path / "a.html", method="pca", images=False,
                           map_backend="svg")
    assert "file://" in out.read_text()
    out2, _ = render_review(npy, out=tmp_path / "b.html", method="pca", images=False,
                            map_backend="svg",
                            image_url="https://arch.example/{filename}")
    assert "https://arch.example/h0_d0_x.png" in out2.read_text()


def test_colour_schemes_and_the_unlabeled_toggle_survive(tmp_path):
    """`mole viz`'s affordances: scheme switching and show/hide-unattributed."""
    npy = _corpus(tmp_path, n_hands=5, docs=4)
    out, _ = render_review(npy, out=tmp_path / "s.html", method="pca", images=False,
                           map_backend="svg")
    html = out.read_text()
    payload = _payload(out)

    assert payload["first"] == "hand"
    assert len(payload["schemes"]) >= 2              # hand + discovered clusters
    assert any("cluster" in n for n in payload["schemes"])
    n = len(payload["schemes"]["hand"]["colors"])
    assert all(len(sc["colors"]) == n for sc in payload["schemes"].values())

    assert 'id="scheme"' in html                     # the picker
    assert 'id="unl"' in html                        # the show/hide toggle
    assert "data-unl=" in html                       # crossed-through dots (svg)
    assert "<path d=" in html
    from mole.viz.scatter import _UNLABELED_GREY
    assert _UNLABELED_GREY in json.dumps(payload["schemes"]["hand"]["colors"])


def test_inspector_and_expert_view_exist(tmp_path):
    npy = _corpus(tmp_path, n_hands=4, docs=4)
    out, _ = render_review(npy, out=tmp_path / "i.html", method="pca", images=False,
                           map_backend="svg")
    html = out.read_text()
    assert 'id="inspect"' in html                    # the side panel
    assert "MOLE.onTap(showDoc)" in html             # ... fed by map taps
    assert 'id="expert"' in html
    assert "body.expert .panel,body.expert .bar{display:none}" in html


def test_the_divider_between_map_and_viewer_is_draggable(tmp_path):
    npy = _corpus(tmp_path, n_hands=4, docs=4)
    out, _ = render_review(npy, out=tmp_path / "d.html", method="pca", images=False,
                           map_backend="svg")
    html = out.read_text()
    assert 'id="split"' in html
    assert "cursor:col-resize" in html
    assert "mousemove" in html and "col-resize" in html
    # Bokeh lays out from a ResizeObserver; the drag must nudge it when it settles
    assert "new Event('resize')" in html


def test_image_scope_all_covers_every_document(tmp_path):
    """Expert mode clicks arbitrary points, so every page must be embedded."""
    npy = _corpus(tmp_path, n_hands=4, docs=4)          # 16 documents
    listed, _ = render_review(npy, out=tmp_path / "l.html", method="pca", max_mb=0,
                              map_backend="svg")
    every, _ = render_review(npy, out=tmp_path / "a.html", method="pca", max_mb=0,
                             image_scope="all", map_backend="svg")
    assert len(_payload(every)["images"]) == 16
    assert len(_payload(every)["images"]) >= len(_payload(listed)["images"])


def test_both_map_backends_expose_the_same_interface(tmp_path):
    """The page talks to `window.MOLE`, never to a backend directly."""
    npy = _corpus(tmp_path, n_hands=4, docs=4)
    calls = ("MOLE.setColors", "MOLE.setAlphas", "MOLE.showImage", "MOLE.onTap")
    svg, _ = render_review(npy, out=tmp_path / "svg.html", method="pca",
                           images=False, map_backend="svg")
    html = svg.read_text()
    assert all(c in html for c in calls) and "window.MOLE" in html
    # the svg build carries no BokehJS at all (a shared CSS comment mentions the
    # name, so test for the runtime rather than the word)
    assert "Bokeh.documents" not in html
    assert svg.stat().st_size < 1024 * 1024

    pytest.importorskip("bokeh")
    bk, _ = render_review(npy, out=tmp_path / "bk.html", method="pca",
                          images=False, map_backend="bokeh", max_mb=0)
    bhtml = bk.read_text()
    assert all(c in bhtml for c in calls) and "window.MOLE" in bhtml
    assert bk.stat().st_size > svg.stat().st_size    # ~4 MB of inlined BokehJS
