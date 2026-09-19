#!/usr/bin/env python
"""Sum pooling vs generalized max pooling (GMP) inside VLAD — CPU, from a token cache.

Tim Raven (mail, 2026-09-19): his VLAD class defaults to GMP with gamma=1000, which the
paper never mentions (the released inference script hardcodes sum). GMP replaces each
cluster's residual SUM with the ridge solution of ``R xi = 1`` (Murray & Perronnin,
CVPR 2014), so a burst of near-identical patches — a repeated letterform, a ruled
margin — can no longer dominate a cluster block. Power-norm and intra-normalisation
patch the same burstiness after the fact, so GMP may or may not stack with them.

Everything here is one codebook per archive (its own transductive fit from the cache,
seeded), encoded once per pooling variant, evaluated with the same `mole eval`
settings, and compared with the §4.2 rule (`compare_evals_multi`). Only the pooling
changes between the two sides of each pair — same tokens, same centres, same K.

    python scripts/run_gmp_ab.py runs/sup_tokens_full
    python scripts/run_gmp_ab.py runs/sup_tokens_full --gammas 100 1000 10000
    python scripts/run_gmp_ab.py runs/sup_tokens_full --archives antwerp flanders --intra-norm

Prints, per gamma, the per-archive macro-mAP for sum vs gmp and the multi-archive
verdict. A gain that survives the guardrail on every archive is the one to deploy
(`mole embed --vlad-pooling gmp --gmp-gamma G`); remember every embedding in an
index must then be re-encoded the same way, and any PCA/whitening refit.

GMP is a per-cluster [dim, dim] solve, so encoding costs a few ms per cluster per
page; an archive of a few hundred pages takes minutes, not hours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", type=Path, help="Token cache dir (`mole sup tokens`).")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs/gmp_ab"))
    ap.add_argument("--archives", nargs="*", default=None)
    ap.add_argument("--gammas", type=float, nargs="+", default=[1000.0],
                    help="GMP ridge gammas to try (Raven's default 1000). Larger = closer "
                         "to plain sum; smaller = stronger equalisation.")
    ap.add_argument("--clusters", type=int, default=100)
    ap.add_argument("--codebook-descriptors", type=int, default=0,
                    help="Descriptors used to fit each archive's codebook. 0 = ALL "
                         "(the deployed default).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--intra-norm", action="store_true",
                    help="Intra-normalise BOTH sides (does GMP stack with the skewed-"
                         "collection recipe?). Default: plain VLAD both sides.")
    ap.add_argument("--no-cross-doc-only", dest="cross_doc_only", action="store_false",
                    default=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from mole.embed.vlad import fit_codebook
    from mole.eval.compare import compare_evals_multi, format_multi_compare
    from mole.eval.retrieval import evaluate
    from mole.supervised.netvlad import vlad_page_vectors, write_embeddings
    from mole.supervised.tokens import TokenCache, descriptor_pool

    cache = TokenCache.load(args.cache)
    print(f"[mole] token cache: {cache.stats()}")
    print(f"[mole] cap = {cache.meta.get('max_tokens_per_page')} tokens/page  |  "
          f"intra_norm = {args.intra_norm}\n")
    args.out.mkdir(parents=True, exist_ok=True)

    def encode(A, page_rows, codebook, pooling, gamma):
        tag = "sum" if pooling == "sum" else f"gmp{gamma:g}"
        npy = args.out / f"{A}.{tag}.npy"
        write_embeddings(
            npy, vlad_page_vectors(cache, codebook, page_rows, intra_norm=args.intra_norm,
                                   pooling=pooling, gmp_gamma=gamma),
            cache, page_rows,
            {"pooling": "vlad", "aggregator": "hard-kmeans", "vlad_pooling": pooling,
             "vlad_gmp_gamma": gamma if pooling == "gmp" else None,
             "vlad_intra_norm": args.intra_norm,
             "vlad_clusters": int(codebook.shape[0]), "vlad_seed": args.seed,
             "note": "transductive codebook refit from the token cache"},
            dataset_dir=args.data / A)
        res = evaluate(npy, args.data / A, topk=(1, 5), cross_doc_only=args.cross_doc_only,
                       out=args.out / f"{A}.{tag}.eval.json")
        return res.overall

    table: dict[str, dict[str, dict]] = {}     # archive -> tag -> scores
    for A in (args.archives or cache.archives):
        print(f"\n== {A}")
        page_rows = cache.rows_for(archive=A)
        pool = descriptor_pool(cache, page_rows, max_descriptors=args.codebook_descriptors,
                               seed=args.seed)
        codebook = fit_codebook(pool, n_clusters=args.clusters, seed=args.seed)
        del pool
        scores = {"sum": encode(A, page_rows, codebook, "sum", 0.0)}
        for g in args.gammas:
            scores[f"gmp{g:g}"] = encode(A, page_rows, codebook, "gmp", g)
        table[A] = {t: {"macro": s.macro_map, "map": s.mean_ap, "top1": s.top1}
                    for t, s in scores.items()}
        line = "  ".join(f"{t} {s.macro_map:.4f}" for t, s in scores.items())
        print(f"[mole] {A}: {line}")

    tags = ["sum"] + [f"gmp{g:g}" for g in args.gammas]
    w = max(len(a) for a in table)
    print(f"\n\n{'=' * 72}\n== macro-mAP per archive (same codebook per row; only the pooling "
          f"differs)\n{'=' * 72}")
    print(f"  {'archive':<{w}}  " + "  ".join(f"{t:>10}" for t in tags))
    for A, row in table.items():
        print(f"  {A:<{w}}  " + "  ".join(f"{row[t]['macro']:>10.4f}" for t in tags))
    print(f"\n  (mAP / Top-1)")
    for A, row in table.items():
        print(f"  {A:<{w}}  " + "  ".join(
            f"{row[t]['map']:.3f}/{row[t]['top1']:.3f}".rjust(10) for t in tags))

    verdicts = {}
    for g in args.gammas:
        tag = f"gmp{g:g}"
        pairs = [(args.out / f"{A}.sum.eval.json", args.out / f"{A}.{tag}.eval.json")
                 for A in table]
        r = compare_evals_multi(pairs, seed=args.seed)
        print(f"\n{'=' * 72}\n== sum → {tag} (§4.2 rule)\n{'=' * 72}")
        print(format_multi_compare(r))
        verdicts[tag] = {"mean_delta": r.mean_delta, "ci": [r.ci_low, r.ci_high],
                         "guardrail_ok": r.guardrail_ok, "passes": r.passes,
                         "worst": [r.worst_label, r.worst_delta]}
        print("   " + ("DEPLOY-GRADE gain" if r.passes else
                       "not a clean win — see per-archive deltas above"))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"cache": str(args.cache), "intra_norm": args.intra_norm, "seed": args.seed,
             "gammas": args.gammas, "per_archive": table, "verdicts": verdicts}, indent=2))
        print(f"[mole] ✓ {args.json}")


if __name__ == "__main__":
    main()
