"""Per-archive sibling-scan grouping into *document ids*.

Cross-document retrieval (``mole eval --cross-doc-only`` and, later, the
supervised positive rule) must treat sibling scans of ONE physical charter as
the same document, so a model earns no credit for merely re-finding a sibling
scan of the query. Which images are siblings is archive-specific; the rules
below were resolved against the real filenames (see ``SUPERVISED_PLAN.md`` D3,
2026-07-19):

* **antwerp**  — one charter per image (the leading ``0-NNNN`` is a unique
  counter; ``XX_XX`` marks an undated day/month and ``-NN`` ranks *distinct*
  undated charters, NOT scans of one charter). doc id = filename stem; no
  sibling collapsing.
* **utrecht**  — one image per doc; doc id = stem minus a trailing
  ``" adjusted"`` (all stems already unique).
* **brackley** — one image per charter; doc id = stem.
* **flanders** — ``<n>_<scan>_<shelfmark>``; siblings share the leading ``<n>``
  (e.g. ``134_2_RAGent K21_98`` and ``134_3_RAGent K21_98`` → doc ``134``).
* **leroy**    — doc id = the ``gysseling_nr`` **column** (charter edition
  number); the filename carries no doc grouping. Column-based, so it is applied
  by :func:`doc_id_resolver`, not by the pure :func:`doc_id_for`.

A dataset folder may override everything with a ``doc_ids.csv``
(``filename,doc_id``). Unknown archives fall back to the filename stem — each
image is then its own document, so ``--cross-doc-only`` degrades to the standard
metric (a safe no-op, never a false collapse).
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

DOC_IDS_FILENAME = "doc_ids.csv"

# substring of the dataset folder name -> canonical archive key. Folder names
# vary (antwerp-bin, flanders-set-bin, brackley-2350, utrecht, leroy-bin), so we
# match on a stable substring rather than the exact name.
_ARCHIVE_ALIASES: tuple[tuple[str, str], ...] = (
    ("antwerp", "antwerp"),
    ("utrecht", "utrecht"),
    ("brackley", "brackley"),
    ("flanders", "flanders"),
    ("leroy", "leroy"),
)

# archives whose doc id comes from a labels.csv column, not the filename.
_DOC_ID_COLUMN: dict[str, str] = {"leroy": "gysseling_nr"}


def canonical_archive(archive: str) -> str:
    """Map a dataset folder name to its canonical archive key ('?' if unknown)."""
    a = archive.lower()
    for needle, key in _ARCHIVE_ALIASES:
        if needle in a:
            return key
    return "?"


def doc_id_for(filename: str, archive: str) -> str:
    """Document id for *filename* under *archive*, from the filename alone.

    This is the pure, filename-only rule. Column-based archives (leroy) return
    the filename stem here; their real grouping lives in a labels.csv column and
    is applied by :func:`doc_id_resolver`. Callers that need correct grouping for
    every archive should go through :func:`doc_id_resolver`.
    """
    key = canonical_archive(archive)
    stem = Path(filename).stem
    if key == "flanders":
        return stem.split("_", 1)[0] or stem
    if key == "utrecht":
        suffix = " adjusted"
        return stem[: -len(suffix)] if stem.endswith(suffix) else stem
    # antwerp, brackley, leroy (filename fallback), unknown -> whole stem
    return stem


def _read_doc_ids_csv(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fn = (row.get("filename") or "").strip()
            did = (row.get("doc_id") or "").strip()
            if fn and did:
                mapping[fn] = did
    return mapping


def _read_column(path: Path, column: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fn = (row.get("filename") or "").strip()
            if fn:
                mapping[fn] = (row.get(column) or "").strip()
    return mapping


def doc_id_resolver(dataset_root: str | Path) -> Callable[[str], str]:
    """Return a ``basename -> doc_id`` function for one dataset folder.

    Resolution order:

    1. an explicit ``doc_ids.csv`` (``filename,doc_id``) in the folder wins;
    2. a column-based archive (leroy → ``gysseling_nr``) reads that column from
       ``labels.csv``, falling back to the filename rule where the column is
       blank;
    3. otherwise the per-archive filename rule (:func:`doc_id_for`).

    The returned ids are NOT namespaced by dataset — callers pooling several
    datasets should prefix with the dataset name so ids never collide across
    archives.
    """
    root = Path(dataset_root)
    archive = root.name
    key = canonical_archive(archive)

    override = root / DOC_IDS_FILENAME
    if override.is_file():
        m = _read_doc_ids_csv(override)
        return lambda fn: m.get(fn) or doc_id_for(fn, archive)

    column = _DOC_ID_COLUMN.get(key)
    labels_csv = root / "labels.csv"
    if column and labels_csv.is_file():
        m = _read_column(labels_csv, column)
        return lambda fn: (
            f"{column}:{m[fn]}" if m.get(fn) else doc_id_for(fn, archive))

    return lambda fn: doc_id_for(fn, archive)
