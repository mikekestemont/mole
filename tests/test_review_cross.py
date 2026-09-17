"""Cross-archive lists checked against a pool built to contain the answer.

Three archives share one embedding space. Each archive carries its own
"scanner" offset (a shared vector added to every page), so raw cosine ranks
archives first; centering must undo that. Planted: one scribe recorded under
two names in two archives, one unlabeled page in a third archive drawn from a
known hand elsewhere, and one label-free pair of pages from a scribe nobody
has named.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mole.review.cross import build_cross, center_by_archive, csls_matrix, format_report


def _pool(tmp_path: Path, offset: float = 1.5):
    rng = np.random.default_rng(0)
    dim = 48
    centers = {k: rng.standard_normal(dim) for k in ("SHARED", "A1", "A2", "B1", "C1", "LOOSE", "NAMELESS")}
    scanner = {"arch1": rng.standard_normal(dim) * offset,
               "arch2": rng.standard_normal(dim) * offset,
               "arch3": rng.standard_normal(dim) * offset}
    vecs, rows, labels = [], [], {a: ["filename,hand_id"] for a in scanner}

    def add(archive, name, center, hand, spread=0.08):
        v = center + spread * rng.standard_normal(dim) + scanner[archive]
        vecs.append(v)
        ds = tmp_path / archive
        ds.mkdir(exist_ok=True)
        (ds / name).touch()
        rows.append({"row": len(rows), "image": str(ds / name)})
        if hand:
            labels[archive].append(f"{name},{hand}")

    for i in range(4):
        add("arch1", f"s{i}_1.png", centers["SHARED"], "X")       # the shared scribe, name X here
    for i in range(4):
        add("arch2", f"t{i}_1.png", centers["SHARED"], "Y")       # ... and Y over there
    for i in range(4):
        add("arch1", f"a{i}_1.png", centers["A1"], "A1")
    for i in range(4):
        add("arch1", f"b{i}_1.png", centers["A2"], "A2")
    for i in range(4):
        add("arch2", f"c{i}_1.png", centers["B1"], "B1")
    for i in range(5):
        add("arch3", f"d{i}_1.png", centers["C1"], "C1")
    for i in range(4):
        add("arch3", f"e{i}_1.png", centers["LOOSE"], "LOOSE", spread=0.4)
    add("arch3", "unl_1.png", centers["A1"], "")                  # unlabeled, really A1
    add("arch1", "nn1_1.png", centers["NAMELESS"], "")            # a scribe nobody named,
    add("arch3", "nn2_1.png", centers["NAMELESS"], "")            # once in each of two archives
    for i in range(3):                                            # filler, unlabeled
        add("arch2", f"f{i}_1.png", rng.standard_normal(dim), "")

    for a, lines in labels.items():
        (tmp_path / a / "labels.csv").write_text("\n".join(lines) + "\n")
    npy = tmp_path / "pool.npy"
    np.save(npy, np.asarray(vecs, dtype=np.float32))
    (tmp_path / "pool.mapping.json").write_text(json.dumps({"model_id": "t@0", "rows": rows}))
    return npy


def test_centering_removes_the_scanner_and_the_lists_find_the_plants(tmp_path):
    npy = _pool(tmp_path)
    report, table = build_cross([npy], limit=20, out=tmp_path / "pool.cross.json")

    # the scanner offset dominates raw neighbours; centering breaks it
    assert report.gap_raw["purity@1"] > report.gap_centered["purity@1"]
    assert report.gap_centered["purity@1"] < 0.9

    # A: X (arch1) and Y (arch2) are one scribe
    top = report.hand_pairs[0]
    assert {top["hand_a"], top["hand_b"]} == {"arch1/X", "arch2/Y"}
    assert top["rank_ab"] == 1 and top["rank_ba"] == 1
    assert top["same_hand_pct"] is not None
    assert top["score"] >= top["cross_mean"]          # two strongest pairs, not the average
    # never two hands of one archive
    assert all(d["archive_a"] != d["archive_b"] for d in report.hand_pairs)

    # B: the unlabeled arch3 page is proposed to arch1/A1, above anything at home
    unl = [d for d in report.page_to_hand if d["document"] == "arch3/unl_1.png"]
    assert unl and unl[0]["hand"] == "arch1/A1" and unl[0]["delta"] > 0
    assert all(d["archive"] != d["hand"].split("/", 1)[0] for d in report.page_to_hand)

    # C: the nameless scribe's two pages are mutual cross-archive neighbours
    pp = report.page_pairs
    assert any({d["document_a"], d["document_b"]} == {"arch1/nn1_1.png", "arch3/nn2_1.png"}
               for d in pp[:3])
    assert all(d["archive_a"] != d["archive_b"] for d in pp)

    # the JSON round-trips and the terminal report renders
    saved = json.loads((tmp_path / "pool.cross.json").read_text())
    assert saved["archives"] == {"arch1": 13, "arch2": 11, "arch3": 11}
    assert "purity@1" in format_report(report)


def test_reference_shares_are_shares(tmp_path):
    npy = _pool(tmp_path)
    report, _ = build_cross([npy], limit=20)
    for d in report.hand_pairs + report.page_to_hand + report.page_pairs:
        for k in ("same_hand_pct", "diff_hand_pct"):
            assert d[k] is None or 0.0 <= d[k] <= 1.0
    ref = report.reference
    assert ref["n_same_hand_pairs"] > 0 and ref["n_diff_hand_pairs"] > 0
    # same-hand pairs sit above different-hand pairs inside an archive
    assert ref["same_hand_quartiles"][1] > ref["diff_hand_quartiles"][1]


def test_concatenation_refuses_two_models(tmp_path):
    npy = _pool(tmp_path)
    other = tmp_path / "other.npy"
    np.save(other, np.load(npy))
    (tmp_path / "other.mapping.json").write_text(
        json.dumps({"model_id": "different@1",
                    "rows": json.loads((tmp_path / "pool.mapping.json").read_text())["rows"]}))
    import pytest
    with pytest.raises(ValueError, match="different models"):
        build_cross([npy, other])


def test_csls_is_symmetric_hubness_correction():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((12, 8)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    arch = np.asarray(["a"] * 6 + ["b"] * 6, dtype=object)
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)
    cs = csls_matrix(sim, arch, k=3)
    # same-archive entries are unusable; cross entries are 2cos - r_B(i) - r_A(j)
    assert np.isinf(cs[0, 1]) and np.isfinite(cs[0, 7])
    rb0 = np.sort(sim[0, 6:])[-3:].mean()
    ra7 = np.sort(sim[7, :6])[-3:].mean()
    assert np.isclose(cs[0, 7], 2 * sim[0, 7] - rb0 - ra7, atol=1e-5)
    assert np.allclose(cs[:6, 6:], cs[6:, :6].T, atol=1e-5)


def test_centering_is_per_archive_and_unit_norm():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((10, 5)).astype(np.float32) + 3.0
    arch = ["p"] * 5 + ["q"] * 5
    Xc = center_by_archive(X, arch)
    assert np.allclose(np.linalg.norm(Xc, axis=1), 1.0, atol=1e-5)
    raw_mean = X[:5].mean(0)
    assert np.linalg.norm(raw_mean) > 1.0          # there was an offset to remove


def _pool_with_images(tmp_path: Path):
    """The planted pool, with real bilevel pages so the sheet has images."""
    pytest = __import__("pytest")
    pytest.importorskip("PIL")
    from PIL import Image

    npy = _pool(tmp_path)
    rng = np.random.default_rng(3)
    rows = json.loads((tmp_path / "pool.mapping.json").read_text())["rows"]
    for r in rows:
        arr = (rng.random((120, 90)) > 0.85).astype("uint8") * 255
        Image.fromarray(arr, mode="L").convert("1").save(r["image"])
    return npy


def test_sheet_is_self_contained_and_carries_all_tabs(tmp_path):
    import re
    from mole.review.render import render_cross

    npy = _pool_with_images(tmp_path)
    path, summary = render_cross([npy], limit=10, out=tmp_path / "x.cross.html")
    html = path.read_text()
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert (tmp_path / "x.cross.json").exists()
    D = json.loads(re.search(r"var D = (\{.*\}), decisions", html, re.S).group(1))
    kinds = [s["kind"] for s in D["sections"]]
    assert kinds[:3] == ["page_pairs", "page_to_hand", "hand_pairs"]
    # the hand-pair row flips through the left hand's pages; every row ships images
    hp = next(s for s in D["sections"] if s["kind"] == "hand_pairs")["rows"][0]
    assert len(hp["query_alts"]) >= 2 and hp["closest"]
    assert all(str(i) in D["images"] for i in hp["query_alts"] + hp["closest"])
    # names and hands carry the archive, so the reviewer always sees where a page is from
    assert all("/" in n for n in D["names"])
    assert "gap" in D and "centering" in D["gap"]
    # nothing in the sheet asserts a cross-archive identity: every case asks
    for s in D["sections"]:
        for r in s["rows"]:
            assert r["ask"].endswith("?")
    assert "labels.csv" not in summary
