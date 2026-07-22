"""PAGE XML parsing, oracle zones, and the asymmetric zone-quality metrics."""

from __future__ import annotations

import json

import pytest

from mole.prep.detect import box_iou, excess_area, pad_bbox, text_coverage
from mole.prep.pagexml import pagexml_to_zones, read_page, read_page_dir

PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Page imageFilename="{img}" imageWidth="1000" imageHeight="800">
    <TextRegion id="tr_1" custom="structure {{type:paragraph;}}">
      <Coords points="{pts}"/>
      <TextLine id="l1"><Coords points="110,210 300,215"/></TextLine>
    </TextRegion>
    {extra}
  </Page>
</PcGts>
"""


def _write(tmp_path, name, pts="100,200 900,205 895,600 105,595", extra="", img=None):
    d = tmp_path / "page"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.xml"
    p.write_text(PAGE.format(img=img or f"{name}.jpg", pts=pts, extra=extra))
    return p


def test_reads_region_and_page_size(tmp_path):
    layout = read_page(_write(tmp_path, "a"))
    assert (layout.width, layout.height) == (1000, 800)
    assert layout.image == "a.jpg"
    assert len(layout.text_regions()) == 1
    r = layout.text_regions()[0]
    assert r.bbox == (100, 200, 900, 600)          # bbox of a skewed quadrilateral
    assert r.kind == "paragraph"                   # pulled out of `custom`
    assert len(r.polygon) == 4


def test_text_bbox_unions_multiple_regions(tmp_path):
    extra = ('<TextRegion id="tr_2"><Coords points="50,650 400,650 400,700 50,700"/>'
             '</TextRegion>')
    layout = read_page(_write(tmp_path, "b", extra=extra))
    assert len(layout.text_regions()) == 2
    assert layout.text_bbox() == (50, 200, 900, 700)


def test_non_text_regions_are_excluded_from_the_text_bbox(tmp_path):
    """A seal or decoration must not drag the main text zone outward."""
    extra = ('<ImageRegion id="im_1"><Coords points="10,10 90,10 90,90 10,90"/>'
             '</ImageRegion>')
    layout = read_page(_write(tmp_path, "c", extra=extra))
    assert len(layout.regions) == 2                # both parsed …
    assert layout.text_bbox() == (100, 200, 900, 600)   # … only TextRegion counts


def test_read_page_dir_keys_on_stem_not_extension(tmp_path):
    """XMLs record .jpg while a binarized copy on disk is .png — match on stem."""
    _write(tmp_path, "d", img="d.jpg")
    layouts = read_page_dir(tmp_path / "page")
    assert set(layouts) == {"d"}


def test_malformed_xml_is_skipped_not_fatal(tmp_path):
    _write(tmp_path, "good")
    (tmp_path / "page" / "bad.xml").write_text("<PcGts><unclosed>")
    assert set(read_page_dir(tmp_path / "page")) == {"good"}


def test_oracle_zones_match_images_by_stem(tmp_path):
    _write(tmp_path, "e")
    imgs = tmp_path / "img"
    imgs.mkdir()
    (imgs / "e.png").touch()                       # .png on disk vs .jpg in the XML
    out = tmp_path / "zones.json"
    manifest = pagexml_to_zones(tmp_path / "page", imgs, out)
    assert set(manifest["images"]) == {"e.png"}
    entry = manifest["images"]["e.png"]
    assert entry["boxes"][0]["bbox"] == [100, 200, 900, 600]
    assert entry["size"] == [1000, 800]
    assert json.loads(out.read_text())["detector"] == "pagexml-oracle"


def test_oracle_zone_padding_is_clamped_to_the_page(tmp_path):
    _write(tmp_path, "f")
    imgs = tmp_path / "img"
    imgs.mkdir()
    (imgs / "f.jpg").touch()
    m = pagexml_to_zones(tmp_path / "page", imgs, padding=250)
    assert m["images"]["f.jpg"]["boxes"][0]["bbox"] == [0, 0, 1000, 800]


# ------------------------------------------------------- asymmetric metrics
def test_coverage_and_excess_separate_the_two_failure_modes():
    """IoU alone hides WHICH way a zone went wrong; these do not."""
    truth = (0, 0, 100, 100)
    clipped = (0, 0, 50, 100)       # lost half the text — unrecoverable
    generous = (-50, -50, 150, 150)  # kept everything, plus background — cheap

    assert text_coverage(clipped, truth) == pytest.approx(0.5)
    assert text_coverage(generous, truth) == pytest.approx(1.0)
    assert excess_area(clipped, truth) == pytest.approx(0.5)
    assert excess_area(generous, truth) == pytest.approx(4.0)
    # …and IoU rates them almost identically, which is exactly the problem.
    assert box_iou(clipped, truth) == pytest.approx(0.5)
    assert box_iou(generous, truth) == pytest.approx(0.25)


def test_padding_can_rescue_a_clipped_zone():
    truth = (100, 100, 200, 200)
    tight = (110, 110, 190, 190)                  # clips 10px on every side
    assert text_coverage(tight, truth) < 0.7
    assert text_coverage(pad_bbox(tight, 10), truth) == pytest.approx(1.0)


def test_pad_bbox_clamps_and_never_shrinks():
    assert pad_bbox((5, 5, 10, 10), 10, 12, 12) == (0, 0, 12, 12)
    assert pad_bbox((5, 5, 10, 10), 0) == (5, 5, 10, 10)


def test_zone_family_match_is_case_insensitive():
    """REGRESSION: a locally fine-tuned detector emits 'text', not 'Text'.

    ZONE_FAMILIES is ("Text",). A case-sensitive match dropped every detection
    from such a model, main_text_zone returned None, and prep silently fell back
    to the whole page on every image — no error, no warning, a wasted training
    run. Casing is the weights author's convention, not semantics.
    """
    from mole.prep.detect import Detection, main_text_zone

    lower = [Detection((10, 20, 90, 60), "text", 0.9)]
    upper = [Detection((10, 20, 90, 60), "Text", 0.9)]
    assert main_text_zone(lower) == main_text_zone(upper) == (10, 20, 90, 60)

    # …but unrelated families must still be excluded, in any casing.
    assert main_text_zone([Detection((0, 0, 5, 5), "paratext", 0.9)]) is None
    assert main_text_zone([Detection((0, 0, 5, 5), "Decoration", 0.9)]) is None
