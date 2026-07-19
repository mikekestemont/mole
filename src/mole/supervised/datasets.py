"""Labeled-dataset ingestion for the supervised module (Phase 1).

Builds a :class:`SupervisedIndex` over the per-dataset ``labels.csv`` defined in
:mod:`mole.data.datasets`, handling PARTIAL coverage natively (any subset of
images may be labeled; the rest are kept only as an ``unlabeled`` pool for the
later ``suggest`` path and are structurally absent from any sampler).

Three invariants make the supervised *negative rule* enforceable downstream
(see ``SUPERVISED_PLAN.md`` §0):

* **hands are namespaced** ``f"{archive}/{raw_hand}"`` so identical raw hand
  strings in two archives never collide into false positives;
* **documents are namespaced** ``f"{archive}/{doc_id}"`` (doc ids from
  :mod:`mole.data.docids`) so sibling scans of one charter share a doc and are
  never treated as cross-document positives;
* a **confidence floor** demotes low-trust rows to *unlabeled* — they then act
  as neither positives nor negatives (mirrors ``mole eval --min-confidence``).

:func:`pair_masks` is where the negative rule physically lives: positives are
same-hand / different-document, negatives are different confirmed hands, and
everything else (same document, the diagonal) is ignored. Because the sampler
draws only labeled items, an ``(labeled, unlabeled)`` pair is unrepresentable.

The window-level :class:`HandBatchSampler` is intentionally deferred to Phase 2:
it draws from the feature cache (window row-ids), which does not exist until
``build_feature_cache`` lands.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mole.data.datasets import IMAGE_EXTENSIONS, discover_datasets
from mole.data.docids import doc_id_resolver


@dataclass
class SupItem:
    """One labeled image."""

    path: Path
    archive: str                 # dataset folder name (from discover_datasets)
    hand: str                    # NAMESPACED: f"{archive}/{raw_hand}"
    doc: str                     # NAMESPACED: f"{archive}/{doc_id}"
    confidence: float | None = None


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


@dataclass
class SupervisedIndex:
    """Labeled images grouped by (namespaced) hand and document."""

    items: list[SupItem] = field(default_factory=list)
    unlabeled: list[tuple[str, Path]] = field(default_factory=list)  # (archive, path)
    # derived (rebuilt by _reindex)
    by_hand: dict[str, list[int]] = field(default_factory=dict)
    docs_by_hand: dict[str, set[str]] = field(default_factory=dict)

    def _reindex(self) -> "SupervisedIndex":
        by_hand: dict[str, list[int]] = defaultdict(list)
        docs_by_hand: dict[str, set[str]] = defaultdict(set)
        for i, it in enumerate(self.items):
            by_hand[it.hand].append(i)
            docs_by_hand[it.hand].add(it.doc)
        self.by_hand = dict(by_hand)
        self.docs_by_hand = {h: set(d) for h, d in docs_by_hand.items()}
        return self

    @property
    def hands(self) -> list[str]:
        return sorted(self.by_hand)

    @property
    def archives(self) -> list[str]:
        return sorted({it.archive for it in self.items})

    def retrievable_hands(self, min_docs: int = 2) -> list[str]:
        """Hands that appear in at least ``min_docs`` *distinct* documents."""
        return sorted(h for h, docs in self.docs_by_hand.items()
                      if len(docs) >= min_docs)

    def subset(self, hands: set[str], *, keep_unlabeled: bool = False
               ) -> "SupervisedIndex":
        """A new index restricted to ``hands`` (indices are rebuilt)."""
        items = [it for it in self.items if it.hand in hands]
        idx = SupervisedIndex(
            items=items,
            unlabeled=list(self.unlabeled) if keep_unlabeled else [])
        return idx._reindex()

    def split_hands(self, holdout_frac: float = 0.2, seed: int = 0,
                    stratify_by_archive: bool = True
                    ) -> tuple["SupervisedIndex", "SupervisedIndex"]:
        """Partition *retrievable* hands into (train, holdout) sub-indices.

        Only hands with ≥2 documents are eligible for holdout (an unseen class
        must be queryable); every other hand — and all non-holdout retrievable
        hands — goes to train. The split is seeded and, by default, stratified
        so each archive contributes ~``holdout_frac`` of its retrievable hands.
        """
        rng = random.Random(seed)
        retrievable = self.retrievable_hands(min_docs=2)
        holdout: set[str] = set()

        groups: dict[str, list[str]]
        if stratify_by_archive:
            groups = defaultdict(list)
            for h in retrievable:
                groups[h.split("/", 1)[0]].append(h)
        else:
            groups = {"_all": list(retrievable)}

        for hs in groups.values():
            hs = sorted(hs)
            rng.shuffle(hs)
            k = round(len(hs) * holdout_frac)
            k = min(len(hs), max(1, k)) if hs else 0
            holdout.update(hs[:k])

        all_hands = set(self.by_hand)
        return self.subset(all_hands - holdout), self.subset(holdout)

    def write_holdout_split(self, path: str | Path, *, holdout_frac: float = 0.2,
                            seed: int = 0, stratify_by_archive: bool = True
                            ) -> tuple["SupervisedIndex", "SupervisedIndex"]:
        """Write a frozen split file that ``mole eval --holdout-hands`` consumes."""
        train, hold = self.split_hands(holdout_frac, seed, stratify_by_archive)
        Path(path).write_text(json.dumps({
            "seed": seed,
            "holdout_frac": holdout_frac,
            "stratify_by_archive": stratify_by_archive,
            "holdout_hands": hold.hands,
            "train_hands": train.hands,
        }, indent=2))
        return train, hold

    def stats(self) -> str:
        """Human-readable census, incl. sample doc groupings for D3 sign-off."""
        arch_imgs: Counter[str] = Counter(it.archive for it in self.items)
        arch_hands: dict[str, set[str]] = defaultdict(set)
        arch_docs: dict[str, set[str]] = defaultdict(set)
        for it in self.items:
            arch_hands[it.archive].add(it.hand)
            arch_docs[it.archive].add(it.doc)

        lines = [
            f"SupervisedIndex: {len(self.items)} labeled images, "
            f"{len(self.by_hand)} hands, {len(self.unlabeled)} unlabeled",
            "  archive: images / hands / retrievable(≥2 docs) / docs",
        ]
        for a in sorted(arch_imgs):
            retr = sum(1 for h in arch_hands[a] if len(self.docs_by_hand[h]) >= 2)
            lines.append(f"    {a}: {arch_imgs[a]} / {len(arch_hands[a])} / "
                         f"{retr} / {len(arch_docs[a])}")

        census = Counter(len(d) for d in self.docs_by_hand.values())
        lines.append("  hands by #distinct docs: "
                     + ", ".join(f"{k}→{v}" for k, v in sorted(census.items())))

        # R5 guard: show doc ids that collapse >1 image, so the grouping rules
        # can be eyeballed before any training consumes them.
        docs_to_imgs: dict[str, list[str]] = defaultdict(list)
        for it in self.items:
            docs_to_imgs[it.doc].append(it.path.name)
        multi = sorted((d, ns) for d, ns in docs_to_imgs.items() if len(ns) > 1)
        lines.append(f"  multi-image documents (siblings collapsed): {len(multi)}")
        for d, ns in multi[:10]:
            lines.append(f"    {d}: {ns}")
        if not multi:
            lines.append("    (none — every document is a single image here)")
        return "\n".join(lines)


def load_labeled_pairs(labels_root: str | Path,
                       min_confidence: float | None = None) -> SupervisedIndex:
    """Build a :class:`SupervisedIndex` from one archive folder or a pooled root.

    Walks :func:`discover_datasets` (works on the pooled symlink dir), namespaces
    hands and documents by archive, resolves doc ids via
    :func:`mole.data.docids.doc_id_resolver`, and applies the confidence floor
    (rows below it drop to the ``unlabeled`` pool — neither positive nor
    negative). Only labeled images become :class:`SupItem`\\ s.
    """
    index = SupervisedIndex()
    for m in discover_datasets(labels_root):
        archive = m.name
        table = m.labels
        resolve = doc_id_resolver(m.root)
        for img in _list_images(m.root):
            fname = img.name
            raw = table.hand_by_filename.get(fname) if table else None
            conf = table.confidence.get(fname) if table else None
            demoted = (min_confidence is not None and conf is not None
                       and conf < min_confidence)
            if raw is not None and not demoted:
                index.items.append(SupItem(
                    path=img, archive=archive,
                    hand=f"{archive}/{raw}", doc=f"{archive}/{resolve(fname)}",
                    confidence=conf))
            else:
                index.unlabeled.append((archive, img))
    return index._reindex()


def pair_masks(hands: list[str], docs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Positive / negative pair masks for a batch of labeled units.

    ``pos[i, j]`` = same hand AND different document (cross-document positives
    only). ``neg[i, j]`` = different hand (both confirmed by construction of the
    batch). Everything else — same document (incl. the diagonal) — is IGNORED:
    excluded from both masks. Unlabeled units cannot appear here at all; the
    sampler only ever draws labeled items.
    """
    h = np.asarray(hands, dtype=object)
    d = np.asarray(docs, dtype=object)
    same_hand = h[:, None] == h[None, :]
    same_doc = d[:, None] == d[None, :]
    pos = same_hand & ~same_doc          # diagonal is same_doc -> excluded
    neg = ~same_hand                     # diagonal is same_hand -> excluded
    return pos, neg


class HandBatchSampler:  # pragma: no cover - Phase 2
    """P×D×W batches over LABELED windows (Phase 2 — needs the feature cache).

    Deferred here on purpose: the sampling unit is a *window* (a cache row-id),
    so this lands with ``build_feature_cache``. The negative rule it must respect
    is already implemented and unit-tested via :func:`pair_masks`.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "HandBatchSampler lands in Phase 2 with build_feature_cache "
            "(it samples window row-ids from the feature cache).")
