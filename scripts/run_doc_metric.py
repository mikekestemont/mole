#!/usr/bin/env python
"""Leave-one-archive-out test of supervision applied AFTER aggregation.

Three spaces are compared on every held-out archive, all evaluated identically
(own gallery, all hands as queries, cross-document relevance):

  raw        the universal-codebook VLAD vector as it ships          (38,400-d)
  pca        PCA-whitened to out_dim, fitted on the OTHER archives   (   128-d)
  supervised PCA-whitened to whiten_dim then a masked-SupCon
             projection to out_dim, both fitted on the OTHER archives(   128-d)

`pca` is the control that matters: whitening a VLAD descriptor is a strong
unsupervised trick on its own, so the supervised column only earns credit for
`supervised - pca`, measured at the same output dimensionality.

--whiten-dim is the knob that decides whether this works at all, and its safe
direction FLIPS with the sample/dimension ratio. VLAD is 38,400-d and the pool
has 3,392 documents, so near full rank the smallest eigenvalues are estimation
noise and whitening divides by them: measured on the real archives, full-rank
whitening cost -0.031 macro alone and -0.104 with a supervised layer on top.
Truncation regularises here. (On a corpus with far more documents than
dimensions the opposite holds — truncation discards low-variance writer signal.)
Sweep it; one SVD per fold is sliced for every candidate, so it is nearly free.

    python scripts/run_doc_metric.py outputs/universal_full/*.npy

⚠ Every input must share ONE codebook (mole codebook / mole embed --codebook-from).
Per-archive transductive embeddings (outputs/pooled_final) are NOT comparable
across archives and the script will refuse dimension mismatches, but it cannot
detect two different codebooks of the same size — that one is on you.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("embeddings", nargs="+", type=Path,
                    help="One .npy per archive, ALL from the same codebook.")
    ap.add_argument("--whiten-dim", type=int, nargs="+", default=[256],
                    help="Components kept before the supervised layer; repeat to "
                         "sweep (e.g. --whiten-dim 64 128 256 512).")
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="Write the table as JSON.")
    args = ap.parse_args()

    from mole.supervised.docmetric import (
        PCAWhiten, archive_macro_map, fit_doc_metric, load_archive_vectors)

    # `mole embed` drops artifacts beside its output, and a bare *.npy glob picks
    # them up; they are not embeddings and have no mapping sidecar.
    ARTIFACTS = (".codebook.npy", ".whiten.npy")
    paths = [p for p in args.embeddings if not p.name.endswith(ARTIFACTS)]
    if len(paths) != len(args.embeddings):
        dropped = [p.name for p in args.embeddings if p.name.endswith(ARTIFACTS)]
        print(f"  (ignoring {len(dropped)} embed artifact(s): {', '.join(dropped)})")
    if not paths:
        sys.exit("no embedding files left after dropping embed artifacts")

    X, names, hands, docs, archives = load_archive_vectors(paths)
    order = sorted(set(archives.tolist()))
    print(f"{len(X):,} documents × {X.shape[1]:,} dims across {len(order)} archives "
          f"({int(sum(1 for h in hands if h)):,} labeled)\n")

    rows, skipped = [], []
    for archive in order:
        keep = archives == archive
        raw_macro, _ = archive_macro_map(X, hands, docs, keep)
        if not np.isfinite(raw_macro):
            # no hand in this archive has two DIFFERENT charters -> nothing to
            # retrieve. Skipping loudly beats poisoning the mean with a NaN.
            skipped.append(archive)
            print(f"  {archive:<20} SKIPPED — no cross-document queries "
                  f"(every labeled hand is a single charter)")
            continue

        full = PCAWhiten().fit(X[archives != archive])       # one SVD per fold
        pca_macro, _ = archive_macro_map(
            full.truncated(args.out_dim).transform(X), hands, docs, keep)

        best = None
        for wd in args.whiten_dim:
            transform, report = fit_doc_metric(
                X, hands, docs, archives, holdout_archive=archive,
                whiten_dim=wd, out_dim=args.out_dim, epochs=args.epochs,
                lr=args.lr, seed=args.seed, pca=full)
            macro, _ = archive_macro_map(transform(X), hands, docs, keep)
            tag = report["whiten_dim"]
            if len(args.whiten_dim) > 1:
                print(f"      whiten_dim={tag:<5} supervised {macro:.4f}")
            if best is None or macro > best[0]:
                best = (macro, tag, report)
        sup_macro, used_dim, report = best

        rows.append({"archive": archive, "raw": raw_macro, "pca": pca_macro,
                     "supervised": sup_macro, "whiten_dim": used_dim,
                     "n_train_docs": report["n_train_docs"],
                     "best_epoch": report["best_epoch"]})
        print(f"  {archive:<20} raw {raw_macro:.4f} | pca {pca_macro:.4f} "
              f"({pca_macro - raw_macro:+.4f}) | supervised {sup_macro:.4f} "
              f"({sup_macro - raw_macro:+.4f})   [vs pca {sup_macro - pca_macro:+.4f}]")

    if not rows:
        print("\nnothing scoreable — every archive was skipped")
        return

    w = max(len(r["archive"]) for r in rows)
    print(f"\n  {'archive':<{w}}  {'raw':>7} {'pca':>7} {'sup':>7} "
          f"{'Δsup-raw':>9} {'Δsup-pca':>9}  {'wdim':>5}")
    for r in rows:
        print(f"  {r['archive']:<{w}}  {r['raw']:>7.4f} {r['pca']:>7.4f} "
              f"{r['supervised']:>7.4f} {r['supervised'] - r['raw']:>+9.4f} "
              f"{r['supervised'] - r['pca']:>+9.4f}  {r['whiten_dim']:>5}")
    ds = np.asarray([r["supervised"] - r["raw"] for r in rows])
    dp = np.asarray([r["supervised"] - r["pca"] for r in rows])
    dw = np.asarray([r["pca"] - r["raw"] for r in rows])
    print(f"\n  mean Δ vs raw : {ds.mean():+.4f}   (whitening alone: {dw.mean():+.4f})")
    print(f"  mean Δ vs pca : {dp.mean():+.4f}   <- what supervision actually adds")
    print(f"  guardrail     : " + ("all archives ≥ -0.01 vs raw"
                                   if ds.min() >= -0.01 else
                                   f"FAILED — worst {ds.min():+.4f}"))
    if skipped:
        print(f"  skipped       : {', '.join(skipped)} (no cross-document queries)")
    if len(args.whiten_dim) > 1:
        print("  NB whiten_dim was SWEPT and the best kept per archive — that is a "
              "tuned\n     number, not a clean held-out one. Re-run with the single "
              "chosen dim to confirm.")

    if args.json:
        import json
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"\n  → {args.json}")


if __name__ == "__main__":
    sys.exit(main())
