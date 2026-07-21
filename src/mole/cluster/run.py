"""Run FINCH over a ``mole embed`` output and report/serialise the hierarchy.

Writes ``<embeddings>.clusters.json`` — one entry per FINCH level with the per-document
cluster ids, plus agreement against whatever partial ground truth exists. ``mole viz
--clusters`` consumes that file to add a colour scheme per level, so discovered
clusters can be flipped against the known hands in the same scatter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mole.cluster.finch import cluster_agreement, finch


def _load(embeddings: Path):
    npy = embeddings if embeddings.suffix == ".npy" else embeddings.with_suffix(".npy")
    x = np.load(npy)
    sidecar = npy.with_suffix(".mapping.json")
    meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    rows = meta.get("rows") or [{"row": i, "image": str(i)} for i in range(len(x))]
    if len(rows) != len(x):
        rows = [{"row": i, "image": str(i)} for i in range(len(x))]
    return x, meta, rows


def _hands(rows: list[dict]) -> list[str | None]:
    """Ground-truth hand per row, ``None`` where the document is unlabeled."""
    from mole.data.datasets import load_labels

    cache: dict[Path, object] = {}
    out: list[str | None] = []
    for r in rows:
        img = Path(r["image"])
        if img.parent not in cache:
            try:
                cache[img.parent] = load_labels(img.parent)
            except Exception:
                cache[img.parent] = None
        table = cache[img.parent]
        hand = table.hand_by_filename.get(img.name) if table is not None else None
        out.append(hand or None)
    return out


def cluster_embeddings(embeddings: str | Path, out: str | Path | None = None,
                       metric: str = "cosine") -> dict:
    """FINCH over an embeddings file; returns (and writes) the hierarchy report."""
    embeddings = Path(embeddings)
    x, meta, rows = _load(embeddings)
    res = finch(x, metric=metric)
    hands = _hands(rows)
    n_hands = len({h for h in hands if h})

    levels = []
    for i, (labels, k) in enumerate(zip(res.partitions, res.n_clusters)):
        entry = {"level": i, "n_clusters": int(k), "labels": [int(v) for v in labels]}
        entry.update(cluster_agreement(hands, labels))
        levels.append(entry)

    report = {
        "model_id": meta.get("model_id"), "metric": metric,
        "n_points": int(len(x)), "n_known_hands": n_hands,
        "images": [str(r["image"]) for r in rows],
        "levels": levels,
    }
    out = Path(out) if out else embeddings.with_suffix(".clusters.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["_path"] = str(out)
    return report


def format_report(report: dict) -> str:
    """Compact per-level table: size of the partition and how well it matches truth."""
    head = (f"FINCH — {report['n_points']} documents, metric {report['metric']}"
            + (f", {report['n_known_hands']} known hands" if report["n_known_hands"] else ""))
    lines = [head, "", "  level  clusters  purity     NMI     ARI   (labeled)", ]
    for lv in report["levels"]:
        if lv["purity"] is None:
            lines.append(f"  {lv['level']:>5}  {lv['n_clusters']:>8}        --      --      --")
        else:
            lines.append(f"  {lv['level']:>5}  {lv['n_clusters']:>8}    {lv['purity']:.3f}   "
                         f"{lv['nmi']:.3f}   {lv['ari']:.3f}   ({lv['n_labeled']})")
    if report["n_known_hands"]:
        lines += ["", f"  (a partition near {report['n_known_hands']} clusters is the one to compare "
                      "against the known hands; purity rises trivially as clusters shrink, so read "
                      "it together with NMI/ARI)"]
    return "\n".join(lines)
