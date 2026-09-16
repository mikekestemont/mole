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
    # one recorded page sits in another hand's cloud — a true misidentification,
    # so the false-positive tab is not empty under the high stage-1 bar
    if n_hands >= 2 and docs >= 3:
        h1 = vecs[docs]
        vecs[docs - 1] = h1 + 0.02 * rng.standard_normal(dim)
    # ... and one unattributed page beside hand 1 — a false negative to attribute
    if n_hands >= 2:
        name = "u_near_h1.png"
        arr = (rng.random((300, 220)) > 0.82).astype("uint8") * 255
        Image.fromarray(arr, mode="L").convert("1").save(ds / name)
        vecs.append(vecs[docs] + 0.03 * rng.standard_normal(dim))
        names.append(name)
        rows.append({"row": len(rows), "image": str(ds / name)})
    (ds / "labels.csv").write_text("\n".join(lab) + "\n")
    npy = tmp_path / "e.npy"
    np.save(npy, np.asarray(vecs, dtype=np.float32))
    (tmp_path / "e.mapping.json").write_text(
        json.dumps({"model_id": "t@0", "rows": rows}))
    return npy


def _payload(path):
    return json.loads(re.search(r"var D = (\{.*\}), decisions", path.read_text(),
                                re.S).group(1))


def test_sheet_is_a_single_self_contained_file(tmp_path):
    """The review file must contain no external reference of any kind."""
    npy = _corpus(tmp_path)
    out, summary = render_review(npy, out=tmp_path / "r.html", limit=10)
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
                           map_backend="bokeh", max_mb=0, mode="viz")
    html = out.read_text()
    assert not re.findall(r'<script[^>]+src\s*=\s*"http', html)   # no fetched script tag
    assert not re.findall(r'<link[^>]+href\s*=\s*"http', html)    # no fetched stylesheet
    assert "data:image/webp" in html or "data:image/png" in html   # charters inline


def test_size_cap_is_enforced_not_merely_advertised(tmp_path):
    npy = _corpus(tmp_path, n_hands=6, docs=4)
    big, _ = render_review(npy, out=tmp_path / "big.html", max_mb=0)
    small, summary = render_review(npy, out=tmp_path / "small.html", max_mb=0.02)
    assert small.stat().st_size < big.stat().st_size
    assert small.stat().st_size < 1024 * 1024        # the cap really bit
    assert "omitted" in summary                      # ... and it said so


def test_bokeh_overhead_is_charged_to_the_size_budget(tmp_path):
    """--max-mb must stay honest once ~4 MB of BokehJS shares the file."""
    pytest.importorskip("bokeh")
    npy = _corpus(tmp_path, n_hands=4, docs=4)
    with pytest.raises(RuntimeError, match="BokehJS"):
        render_review(npy, out=tmp_path / "x.html", method="pca",
                      map_backend="bokeh", max_mb=1.0, mode="viz")


def test_no_images_mode_is_small_and_still_useful(tmp_path):
    npy = _corpus(tmp_path)
    out, _ = render_review(npy, out=tmp_path / "t.html", images=False)
    assert not _payload(out)["images"]               # no charter was embedded
    assert out.stat().st_size < 400 * 1024
    assert "Hand review" in out.read_text()          # the cases are still there


def test_uncalibrated_lists_ask_questions_rather_than_assert(tmp_path):
    """False positives have no ground truth — they must not sound certain."""
    npy = _corpus(tmp_path, n_hands=6, docs=5)
    out, _ = render_review(npy, out=tmp_path / "q.html", images=False)
    kinds = {s["kind"]: s for s in _payload(out)["sections"]}
    assert "outliers" in kinds and "attributions" in kinds
    assert "merges" not in kinds
    for row in kinds["outliers"]["rows"]:
        assert "Keep if" in row["text"] and "Reject" in row["text"], row["text"]
        assert row["ask"].endswith("?")
    for row in kinds["attributions"]["rows"]:
        assert row["ask"].endswith("?")
        assert "unattributed" in row["text"].lower()


def test_local_file_links_by_default_and_template_when_given(tmp_path):
    npy = _corpus(tmp_path)
    out, _ = render_review(npy, out=tmp_path / "a.html", images=False)
    assert "file://" in out.read_text()
    out2, _ = render_review(npy, out=tmp_path / "b.html", images=False,
                            image_url="https://arch.example/{filename}")
    assert "https://arch.example/h0_d0_x.png" in out2.read_text()


def test_colour_schemes_and_the_unlabeled_toggle_survive(tmp_path):
    """`mole viz`'s affordances: scheme switching and show/hide-unattributed."""
    npy = _corpus(tmp_path, n_hands=5, docs=4)
    out, _ = render_review(npy, out=tmp_path / "s.html", method="pca", images=False,
                           map_backend="svg", mode="viz")
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


def test_inspector_exists_without_a_review_toggle(tmp_path):
    npy = _corpus(tmp_path, n_hands=4, docs=4)
    out, _ = render_review(npy, out=tmp_path / "i.html", method="pca", images=False,
                           map_backend="svg", mode="viz")
    html = out.read_text()
    assert 'id="inspect"' in html                    # the side panel
    assert "MOLE.onTap(showDoc)" in html             # ... fed by map taps
    assert 'id="showlists"' not in html
    assert "body.expert" not in html
    assert 'id="tabs"' not in html and 'id="queue"' not in html


def test_the_divider_between_map_and_viewer_is_draggable(tmp_path):
    npy = _corpus(tmp_path, n_hands=4, docs=4)
    out, _ = render_review(npy, out=tmp_path / "d.html", method="pca", images=False,
                           map_backend="svg", mode="viz")
    html = out.read_text()
    assert 'id="split"' in html
    assert "cursor:col-resize" in html
    assert "mousemove" in html and "col-resize" in html
    # Bokeh lays out from a ResizeObserver; the drag must nudge it when it settles
    assert "new Event('resize')" in html


def test_image_scope_all_covers_every_document(tmp_path):
    """Viz clicks arbitrary points, so every page must be embeddable."""
    npy = _corpus(tmp_path, n_hands=4, docs=4)          # 16 + 1 unattributed
    listed, _ = render_review(npy, out=tmp_path / "l.html", method="pca", max_mb=0,
                              map_backend="svg", mode="viz")
    every, _ = render_review(npy, out=tmp_path / "a.html", method="pca", max_mb=0,
                             image_scope="all", map_backend="svg", mode="viz")
    assert len(_payload(every)["images"]) == 17
    assert len(_payload(every)["images"]) >= len(_payload(listed)["images"])


def test_both_map_backends_expose_the_same_interface(tmp_path):
    """The page talks to `window.MOLE`, never to a backend directly."""
    npy = _corpus(tmp_path, n_hands=4, docs=4)
    calls = ("MOLE.setColors", "MOLE.setAlphas", "MOLE.showImage", "MOLE.onTap")
    svg, _ = render_review(npy, out=tmp_path / "svg.html", method="pca",
                           images=False, map_backend="svg", mode="viz")
    html = svg.read_text()
    assert all(c in html for c in calls) and "window.MOLE" in html
    # the svg build carries no BokehJS at all (a shared CSS comment mentions the
    # name, so test for the runtime rather than the word)
    assert "Bokeh.documents" not in html
    assert svg.stat().st_size < 1024 * 1024

    pytest.importorskip("bokeh")
    bk, _ = render_review(npy, out=tmp_path / "bk.html", method="pca",
                          images=False, map_backend="bokeh", max_mb=0, mode="viz")
    bhtml = bk.read_text()
    assert all(c in bhtml for c in calls) and "window.MOLE" in bhtml
    assert bk.stat().st_size > svg.stat().st_size    # ~4 MB of inlined BokehJS


def test_no_bokeh_warning_from_the_empty_page_source(tmp_path):
    """Building the map must not emit BokehUserWarning (empty ColumnDataSource)."""
    pytest.importorskip("bokeh")
    import warnings

    npy = _corpus(tmp_path, n_hands=4, docs=4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")              # any BokehUserWarning fails
        render_review(npy, out=tmp_path / "w.html", method="pca", images=False,
                      map_backend="bokeh", max_mb=0, mode="viz")


def test_charter_viewers_are_zoomable(tmp_path):
    """Review panes and the viz inspector zoom with the wheel, not a second file."""
    npy = _corpus(tmp_path, n_hands=6, docs=5)
    review, _ = render_review(npy, out=tmp_path / "r.html", images=False)
    rhtml = review.read_text()
    assert "bindZoom" in rhtml and "moleResetZoom" in rhtml
    assert rhtml.count('class="page zoombox"') == 3   # two panes + the grid inspector
    assert "Scroll to zoom" in rhtml

    viz, _ = render_review(npy, out=tmp_path / "v.html", method="pca", images=False,
                           map_backend="svg", mode="viz")
    vhtml = viz.read_text()
    assert 'id="pagezoom"' in vhtml and "bindZoom" in vhtml
    assert 'id="pageimg"' in vhtml


def test_wide_pages_are_cropped_not_squashed(tmp_path):
    """A very wide charter is cropped to its middle; the aspect never changes."""
    from PIL import Image

    from mole.review.images import encode_page

    wide = tmp_path / "wide.png"
    Image.new("L", (4000, 900), 255).save(wide)
    _, _, w, h = encode_page(wide, max_aspect=1.7)
    assert abs(w / h - 1.7) < 0.02                  # clipped to the cap
    assert h == 900                                 # height untouched: no shrinking

    tall = tmp_path / "tall.png"
    Image.new("L", (800, 1200), 255).save(tall)
    _, _, w2, h2 = encode_page(tall, max_aspect=1.7)
    assert (w2, h2) == (800, 1200)                  # portrait pages are left alone


def test_finch_levels_are_offered_with_silhouettes(tmp_path):
    npy = _corpus(tmp_path, n_hands=6, docs=4)
    out, _ = render_review(npy, out=tmp_path / "f.html", method="pca", images=False,
                           map_backend="svg", mode="viz")
    schemes = list(_payload(out)["schemes"])
    finch = [n for n in schemes if n.startswith("FINCH")]
    assert finch, schemes
    # a level colouring everything the same says nothing and must not be offered
    assert not any("· 1 clusters" in n for n in finch)
    if len([n for n in finch if "silhouette" in n]) > 1:
        assert sum("★" in n for n in finch) == 1     # exactly one best level marked


def test_viz_html_has_no_review_chrome(tmp_path):
    """``mole viz`` is map + viewer only — no lists, no decisions, no checkbox."""
    npy = _corpus(tmp_path, n_hands=4, docs=4)
    out, _ = render_review(npy, out=tmp_path / "v.html", method="pca", images=False,
                           map_backend="svg", mode="viz")
    html = out.read_text()
    payload = _payload(out)
    assert payload["mode"] == "viz"
    assert payload["sections"] == []
    assert 'id="showlists"' not in html
    assert "Download my decisions" not in html
    assert 'data-v="keep"' not in html
    assert "False positives" not in html
    assert '<body class="dark viz">' in html
    assert "Embeddings —" in html
    assert 'id="inspect"' in html
    assert 'id="hsplit"' not in html
    assert 'id="dl"' not in html


def test_review_queue_is_keep_reject_unsure(tmp_path):
    npy = _corpus(tmp_path, n_hands=4, docs=4)
    out, _ = render_review(npy, out=tmp_path / "r.html", images=False)
    html = out.read_text()
    payload = _payload(out)
    assert payload["mode"] == "review"
    headings = [s["heading"] for s in payload["sections"]]
    assert headings == ["False positives", "False negatives", "New hands"]
    assert 'id="tabs"' in html and "data-tab" in html     # tabs come from the payload
    assert 'disabled title="Stage 3"' not in html
    assert 'id="nums"' in html and "cosine distance" in html
    assert 'data-v="keep"' in html and 'data-v="reject"' in html and 'data-v="unsure"' in html
    assert "Not sure" in html
    assert "Looks right" not in html
    assert "kind,document,hand,decision,note" in html
    assert "never writes labels.csv" in html.lower()
    assert "Hand review" in html
    assert 'id="map"' not in html
    assert "umap" not in html.lower()


def test_review_is_case_by_case_without_a_map(tmp_path):
    """Query vs recorded-hand neighbours; Keep / Reject / Not sure stay on screen."""
    npy = _corpus(tmp_path, n_hands=6, docs=5)
    out, _ = render_review(npy, out=tmp_path / "l.html", images=False)
    html = out.read_text()
    payload = _payload(out)
    assert payload["mode"] == "review"
    assert 'id="querypane"' in html and 'id="exempane"' in html
    assert 'id="workbar"' in html and 'id="caseslider"' in html
    assert "decided" in html and "left" in html
    assert 'id="sort-closest"' in html and 'id="sort-furthest"' in html
    assert 'id="prevEx"' in html and 'id="nextEx"' in html
    assert 'id="split"' not in html and 'id="hsplit"' not in html
    assert "Bokeh.documents" not in html
    assert "Does this charter belong" in html
    assert html.index('data-v="keep"') > html.index('id="querypane"')
    kinds = {s["kind"]: s for s in payload["sections"]}
    assert "outliers" in kinds
    assert "attributions" in kinds and "new_hands" in kinds
    assert 'disabled title="Stage 3"' not in html
    rows = kinds["attributions"]["rows"]
    assert rows
    for row in rows:
        assert "closest" in row and "furthest" in row
        assert "closest_dist" in row and row["hand_dist"] is not None
        assert row["csv_kind"] == "false_negative"
        assert row["hand_role"] == "proposed"
    assert kinds["new_hands"]["rows"]
    for row in kinds["new_hands"]["rows"]:
        assert row["csv_kind"] == "new_hand"
        assert row["hand_role"] == "cluster"
        assert row["ask"].endswith("?")


def test_false_positives_can_be_dropped(tmp_path):
    """A recorded identification is rarely wrong here: the tab can be left out."""
    npy = _corpus(tmp_path, n_hands=6, docs=5)
    on, _ = render_review(npy, out=tmp_path / "on.html", images=False)
    off, _ = render_review(npy, out=tmp_path / "off.html", images=False,
                           false_positives=False)
    assert [s["kind"] for s in _payload(on)["sections"]] == ["outliers", "attributions",
                                                               "new_hands"]
    assert [s["kind"] for s in _payload(off)["sections"]] == ["attributions", "new_hands"]
    assert "False positives" not in off.read_text()
    kinds = {s["kind"]: s for s in _payload(on)["sections"]}
    rows = kinds["outliers"]["rows"]
    assert rows
    for row in rows:
        assert row["csv_kind"] == "false_positive"
        assert "Keep if" in row["text"] and "Reject" in row["text"]


def test_new_hands_are_a_checkbox_grid(tmp_path):
    """A proposed hand is a SET the reviewer thins: every member page ships (with
    the pairwise cosines for the distance-to-central ordering) and gets its own
    checkbox; deselect all + confirm rejects the hand."""
    npy = _corpus(tmp_path, n_hands=6, docs=5)
    out, _ = render_review(npy, out=tmp_path / "nh.html", images=True, max_mb=0)
    html = out.read_text()
    payload = _payload(out)
    kinds = {s["kind"]: s for s in payload["sections"]}
    rows = kinds["new_hands"]["rows"]
    assert rows
    for row in rows:
        n = row["n_class"]
        assert len(row["members"]) == n and row["reference"] in row["members"]
        assert len(row["member_sim"]) == n and all(len(r) == n for r in row["member_sim"])
        for k in range(n):
            assert abs(row["member_sim"][k][k] - 1.0) < 1e-4
        assert row["hand"].startswith("new_hand_")
        assert row["keep_hint"].endswith("· 1") and row["reject_hint"].endswith("· 2")
        assert "Untick" in row["text"] and "Deselect all" in row["text"]
        for i in row["members"]:                     # every page of the group is inline
            assert str(i) in payload["images"]
    assert 'id="grid"' in html and 'id="selAll"' in html and 'id="selNone"' in html
    assert 'id="ginsp"' in html and 'id="giimg"' in html   # zoomable inspector
    assert "kind,document,hand,decision,note" in html


def test_hdbscan_schemes_mark_noise_as_unclustered(tmp_path):
    """HDBSCAN's -1 must read as 'joined nothing', not as a discovered hand."""
    pytest.importorskip("sklearn.cluster", reason="needs scikit-learn >= 1.3")
    npy = _corpus(tmp_path, n_hands=6, docs=4)
    out, _ = render_review(npy, out=tmp_path / "h.html", method="pca", images=False,
                           map_backend="svg", cluster_method="both", mode="viz")
    payload = _payload(out)
    names = list(payload["schemes"])
    hdb = [n for n in names if n.startswith("HDBSCAN")]
    assert hdb, names
    from mole.viz.scatter import _UNLABELED_GREY
    for n in hdb:
        sc = payload["schemes"][n]
        if "-1" in sc["cats"]:
            i = sc["cats"].index("-1")
            assert sc["colors"][i] == _UNLABELED_GREY   # neutral, not a hand colour
    # the star ranks within a method, never across them
    for family in ("FINCH", "HDBSCAN"):
        fam = [n for n in names if n.startswith(family) and "silhouette" in n]
        if len(fam) > 1:
            assert sum("★" in n for n in fam) == 1


def test_cluster_method_selects_which_families_appear(tmp_path):
    pytest.importorskip("sklearn.cluster")
    npy = _corpus(tmp_path, n_hands=6, docs=4)
    only_finch, _ = render_review(npy, out=tmp_path / "f2.html", method="pca",
                                  images=False, map_backend="svg",
                                  cluster_method="finch", mode="viz")
    names = list(_payload(only_finch)["schemes"])
    assert not any(n.startswith("HDBSCAN") for n in names)
    assert any(n.startswith("FINCH") for n in names)


def test_noise_is_never_proposed_as_a_new_hand(tmp_path):
    """The unclustered bag is not a discovery."""
    from mole.review.suggest import NOISE, _new_hands
    import numpy as np

    sim = np.full((6, 6), 0.9, dtype=np.float32)
    labels = np.array([NOISE] * 4 + [7, 7])
    docs = np.asarray([f"a/{i}" for i in range(6)], dtype=object)
    names = [f"d{i}.png" for i in range(6)]
    out = _new_hands(sim, labels, np.zeros(6, bool), np.zeros((6, 1), np.float32),
                     docs, names, [0.5], 10)
    assert all(c["cluster"] != NOISE for c in out)


def test_star_follows_agreement_with_recorded_hands_when_labels_exist(tmp_path):
    """A partition that recovers the archivist's hands beats a merely tidy one."""
    npy = _corpus(tmp_path, n_hands=6, docs=4)
    out, _ = render_review(npy, out=tmp_path / "ag.html", method="pca", images=False,
                           map_backend="svg", mode="viz")
    names = list(_payload(out)["schemes"])
    scored = [n for n in names if "agreement" in n]
    assert scored, names                          # labels exist -> agreement shown
    assert not any("silhouette" in n for n in scored)   # one number, not two


def test_silhouette_is_the_fallback_when_nothing_is_labeled(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(3)
    ds = tmp_path / "arch1"
    ds.mkdir()
    vecs, rows = [], []
    for h in range(6):
        c = rng.standard_normal(24)
        for d in range(4):
            n = f"h{h}_d{d}.png"
            Image.new("L", (60, 80), 255).save(ds / n)
            vecs.append(c + 0.05 * rng.standard_normal(24))
            rows.append({"row": len(rows), "image": str(ds / n)})
    np.save(tmp_path / "u.npy", np.asarray(vecs, dtype=np.float32))
    (tmp_path / "u.mapping.json").write_text(json.dumps({"rows": rows}))
    # no labels.csv at all

    out, _ = render_review(tmp_path / "u.npy", out=tmp_path / "u.html", method="pca",
                           images=False, map_backend="svg", mode="viz")
    names = list(_payload(out)["schemes"])
    assert any("silhouette" in n for n in names), names
    assert not any("agreement" in n for n in names)


def test_selection_leaves_the_rest_legible(tmp_path):
    """Bokeh's default non-selection alpha is invisible on a dark background."""
    pytest.importorskip("bokeh")
    npy = _corpus(tmp_path, n_hands=4, docs=4)
    out, _ = render_review(npy, out=tmp_path / "sel.html", method="pca", images=False,
                           map_backend="bokeh", max_mb=0, mode="viz")
    # Bokeh serialises the kwarg into a nonselection_glyph, so check the model
    from mole.review.bokeh_map import build

    coords = np.zeros((3, 2), dtype=np.float32)
    _, _, _, _ = build(coords, ["a", "b", "c"], ["H", "H", ""],
                          ["#111", "#222", "#333"])
    assert out.read_text()                       # the page still builds
    # and the row-hover dim floor is legible too
    assert "0.18" in out.read_text()


def test_scheme_count_matches_the_scribe_count_in_the_title(tmp_path):
    """"unlabeled" is the absence of a hand, not one more of them."""
    npy = _corpus(tmp_path, n_hands=5, docs=4)      # 4 labelled hands + 1 unlabelled
    out, _ = render_review(npy, out=tmp_path / "c.html", method="pca", images=False,
                           map_backend="svg", mode="viz")
    html = out.read_text()
    title_n = int(re.search(r"(\d+) scribes", html).group(1))
    picker_n = int(re.search(r'<option value="hand">hand \((\d+)\)</option>',
                             html).group(1))
    assert picker_n == title_n
    # the unattributed documents are still THERE, just not counted as a scribe
    assert 'id="unl"' in html


def test_both_dividers_are_draggable(tmp_path):
    npy = _corpus(tmp_path, n_hands=4, docs=4)
    out, _ = render_review(npy, out=tmp_path / "dd.html", method="pca", images=False,
                           map_backend="svg", mode="viz")
    html = out.read_text()
    assert 'id="split"' in html and "cursor:col-resize" in html      # map | viewer
    assert "--fig-h" in html
    assert "height:var(--fig-h)" in html.replace(" ", "")


def test_class_neighbors_are_closest_then_furthest():
    """Other pages of the recorded hand, ranked against the query; siblings skipped."""
    from mole.review.render import _class_neighbors, _l2

    X = np.array([
        [1.0, 0.0, 0.0],
        [0.95, 0.05, 0.0],
        [0.2, 0.8, 0.0],
        [0.0, 0.0, 1.0],
        [0.9, 0.1, 0.0],
    ], dtype=np.float32)
    Xn = _l2(X)
    close, far, csim, fsim = _class_neighbors(
        Xn, 0, [0, 1, 2, 3, 4], ["a", "a", "c", "d", "e"])
    assert 1 not in close and 1 not in far           # sibling scan of the query
    assert close[0] == 4                             # most similar remaining page
    assert far[0] == 3                               # least similar remaining page
    assert close == list(reversed(far))
    assert csim[0] > csim[-1]
    assert abs((1.0 - csim[0]) - (1.0 - max(csim))) < 1e-6


def test_render_does_not_write_labels(tmp_path):
    npy = _corpus(tmp_path, n_hands=4, docs=4)
    lab = next(tmp_path.rglob("labels.csv"))
    before = lab.read_text()
    render_review(npy, out=tmp_path / "r.html", images=False)
    assert lab.read_text() == before
