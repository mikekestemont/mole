"""Tests for per-archive document-id grouping (mole.data.docids)."""

from __future__ import annotations

from mole.data.docids import canonical_archive, doc_id_for, doc_id_resolver


def test_canonical_archive_matches_folder_variants():
    assert canonical_archive("antwerp-bin") == "antwerp"
    assert canonical_archive("flanders-set-bin") == "flanders"
    assert canonical_archive("brackley-2350") == "brackley"
    assert canonical_archive("utrecht") == "utrecht"
    assert canonical_archive("leroy-bin") == "leroy"
    assert canonical_archive("some-new-archive") == "?"


def test_doc_id_for_filename_rules():
    # Antwerp: leading 0-NNNN is unique -> whole stem, NO stripping of -NN
    assert doc_id_for("0-0449_XX_XX_1348-02.png", "antwerp-bin") == "0-0449_XX_XX_1348-02"
    # Flanders: siblings share the leading number
    assert doc_id_for("134_2_RAGent K21_98.jpeg", "flanders-set-bin") == "134"
    assert doc_id_for("134_3_RAGent K21_98.jpeg", "flanders-set-bin") == "134"
    # Utrecht: drop a trailing " adjusted"
    assert doc_id_for("0985.06.26a adjusted.jpg", "utrecht") == "0985.06.26a"
    # Brackley: whole stem
    assert doc_id_for("Brackley_D4.jpg", "brackley-set") == "Brackley_D4"
    # Unknown archive: safe fallback to the whole stem (each image its own doc)
    assert doc_id_for("weird_name_1.tif", "mystery") == "weird_name_1"


def test_flanders_siblings_group_but_distinct_charters_do_not():
    a = doc_id_for("134_2_RAGent K21_98.jpeg", "flanders-set-bin")
    b = doc_id_for("134_3_RAGent K21_98.jpeg", "flanders-set-bin")
    c = doc_id_for("135_1_RAGent K21_99.jpeg", "flanders-set-bin")
    assert a == b       # two scans of charter 134 collapse
    assert a != c       # a different charter does not


def test_leroy_resolver_uses_gysseling_column(tmp_path):
    ds = tmp_path / "leroy-bin"
    ds.mkdir()
    (ds / "labels.csv").write_text(
        "filename,hand_id,match_score,gysseling_nr\n"
        "117o.png,87,0.9,170\n"
        "118o.png,87,0.9,170\n"     # same charter 170 as 117o
        "956o.png,86,0.8,1146\n"
        "orphan.png,5,0.7,\n")      # blank column -> filename fallback
    resolve = doc_id_resolver(ds)
    assert resolve("117o.png") == resolve("118o.png")   # siblings collapse
    assert resolve("117o.png") != resolve("956o.png")   # different charter
    assert resolve("orphan.png") == "orphan"            # blank -> filename stem


def test_doc_ids_csv_override_wins(tmp_path):
    ds = tmp_path / "flanders-set-bin"
    ds.mkdir()
    # override says these two are the SAME doc despite different leading numbers
    (ds / "doc_ids.csv").write_text(
        "filename,doc_id\n"
        "134_2_RAGent K21_98.jpeg,charterX\n"
        "999_1_RAGent K21_98.jpeg,charterX\n")
    resolve = doc_id_resolver(ds)
    assert resolve("134_2_RAGent K21_98.jpeg") == "charterX"
    assert resolve("999_1_RAGent K21_98.jpeg") == "charterX"
    # a filename not in the override falls back to the archive rule
    assert resolve("200_1_RAGent K30_5.jpeg") == "200"
