"""Dataset discovery, manifests, and (optional, partial) label loading.

A *dataset* is minimally a folder of freely named images — no filename
conventions, fully unsupervised by default. Subfolders are treated as named
datasets (``data/pretrain/<name>/``); flat folders also work.

A dataset folder MAY contain a ``labels.csv`` recording hand identifications,
which may cover any subset of the images::

    filename,hand_id[,confidence][,source][,notes]

Labels NEVER influence self-supervised training. They are consumed only by
``mole eval`` and the later ``mole.supervised`` phase. Unlisted images are
simply unlabeled; orphan rows (label rows with no matching file) are reported,
not fatal.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
)
LABEL_FILENAME = "labels.csv"


@dataclass
class LabelTable:
    """Parsed ``labels.csv`` for one dataset, with coverage bookkeeping."""

    hand_by_filename: dict[str, str] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    source: dict[str, str] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    orphan_rows: list[str] = field(default_factory=list)  # rows with no matching image

    @property
    def n_labeled(self) -> int:
        return len(self.hand_by_filename)

    @property
    def hands(self) -> set[str]:
        return set(self.hand_by_filename.values())


@dataclass
class DatasetManifest:
    """Provenance + coverage record for a single dataset folder."""

    name: str
    root: Path
    n_images: int
    label_file_present: bool
    label_file_hash: str | None
    n_labeled: int
    labels: LabelTable | None = None

    @property
    def label_coverage(self) -> float:
        """Fraction of images that carry a hand_id label (0.0 if none)."""
        return self.n_labeled / self.n_images if self.n_images else 0.0


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_labels(dataset_root: str | Path) -> LabelTable:
    """Load and validate ``labels.csv`` against the images actually present.

    Missing optional columns are tolerated. Rows whose ``filename`` is not in the
    folder are collected in ``orphan_rows`` rather than raising.
    """
    root = Path(dataset_root)
    table = LabelTable()
    label_path = root / LABEL_FILENAME
    if not label_path.is_file():
        return table

    present = {p.name for p in _list_images(root)}
    with label_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fname = (row.get("filename") or "").strip()
            hand = (row.get("hand_id") or "").strip()
            if not fname or not hand:
                continue
            if fname not in present:
                table.orphan_rows.append(fname)
                continue
            table.hand_by_filename[fname] = hand
            if row.get("confidence"):
                try:
                    table.confidence[fname] = float(row["confidence"])
                except ValueError:
                    pass
            if row.get("source"):
                table.source[fname] = row["source"].strip()
            if row.get("notes"):
                table.notes[fname] = row["notes"].strip()
    return table


def _build_manifest(name: str, folder: Path) -> DatasetManifest:
    images = _list_images(folder)
    label_path = folder / LABEL_FILENAME
    has_labels = label_path.is_file()
    labels = load_labels(folder) if has_labels else None
    return DatasetManifest(
        name=name,
        root=folder,
        n_images=len(images),
        label_file_present=has_labels,
        label_file_hash=_sha256(label_path) if has_labels else None,
        n_labeled=labels.n_labeled if labels else 0,
        labels=labels,
    )


def discover_datasets(root: str | Path) -> list[DatasetManifest]:
    """Scan ``root`` for datasets and report per-dataset coverage.

    * If ``root`` directly contains images, it is treated as one flat dataset.
    * Otherwise each immediate subfolder containing images becomes a named
      dataset (subfolders are the dataset names).

    Never fails on partial or missing labels.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    manifests: list[DatasetManifest] = []
    if _list_images(root):
        manifests.append(_build_manifest(root.name, root))

    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if _list_images(sub):
            manifests.append(_build_manifest(sub.name, sub))
    return manifests
