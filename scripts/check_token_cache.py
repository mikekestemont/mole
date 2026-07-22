#!/usr/bin/env python
"""Does the CAPPED token cache still reproduce the deployed VLAD numbers?

`mole sup tokens --max-tokens-per-page N` keeps a seeded subsample of each page's
foreground tokens. On the real pool that cap binds on essentially every page
(~6.8M of ~17.8M tokens kept), so "VLAD is saturated in token count" stops being
a citation and becomes a claim about THIS cache that has to be checked before any
NetVLAD number built on it means anything.

The check: refit each archive's OWN transductive codebook from the cache, encode
its pages, evaluate — and compare against the embeddings that shipped
(`outputs/pooled_final`, whose codebook saw every token). Same backbone, same
geometry, same foreground rule, same K, plain VLAD both sides; the ONLY
difference is which tokens the aggregation and the codebook fit got to see.

    python scripts/check_token_cache.py runs/sup_tokens

A mean Δmacro within about ±0.005 says the cap is harmless and the LOAO
experiment can be trusted. A systematic loss says lower `--max-tokens-per-page`
was too aggressive — rebuild the cache with a larger cap (or 0 for all tokens)
before reading anything into a trained aggregator.
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
    ap.add_argument("--reference", type=Path, default=Path("outputs/pooled_final"),
                    help="Per-archive embeddings to reproduce (the deployed ones).")
    ap.add_argument("--out", type=Path, default=Path("outputs/cache_check"))
    ap.add_argument("--archives", nargs="*", default=None)
    ap.add_argument("--clusters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cross-doc-only", action="store_true", default=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from mole.embed.vlad import fit_codebook
    from mole.eval.compare import compare_evals_multi, format_multi_compare
    from mole.eval.retrieval import evaluate
    from mole.supervised.netvlad import vlad_page_vectors, write_embeddings
    from mole.supervised.tokens import TokenCache, descriptor_pool

    cache = TokenCache.load(args.cache)
    cap = cache.meta.get("max_tokens_per_page")
    print(f"[mole] token cache: {cache.stats()}")
    print(f"[mole] cap = {cap} tokens/page\n")
    args.out.mkdir(parents=True, exist_ok=True)

    pairs, rows = [], []
    for A in (args.archives or cache.archives):
        ref = args.reference / f"{A}.npy"
        if not ref.is_file():
            print(f"[mole] skipping {A}: no reference at {ref}")
            continue
        print(f"\n== {A}")
        page_rows = cache.rows_for(archive=A)
        # each archive's OWN codebook, exactly as the transductive deploy does
        pool = descriptor_pool(cache, page_rows, max_descriptors=4_000_000,
                               seed=args.seed)
        codebook = fit_codebook(pool, n_clusters=args.clusters, seed=args.seed)
        mat = vlad_page_vectors(cache, codebook, page_rows)

        npy = args.out / f"{A}.cache.npy"
        write_embeddings(npy, mat, cache, page_rows,
                         {"pooling": "vlad", "aggregator": "hard-kmeans",
                          "vlad_clusters": int(codebook.shape[0]),
                          "note": "transductive, from the capped token cache"},
                         dataset_dir=args.data / A)
        got = evaluate(npy, args.data / A, topk=(1, 5),
                       cross_doc_only=args.cross_doc_only,
                       out=args.out / f"{A}.cache.eval.json")
        want = evaluate(ref, args.data / A, topk=(1, 5),
                        cross_doc_only=args.cross_doc_only,
                        out=args.out / f"{A}.reference.eval.json")
        pairs.append((args.out / f"{A}.reference.eval.json",
                      args.out / f"{A}.cache.eval.json"))
        rows.append({"archive": A, "reference": want.overall.macro_map,
                     "from_cache": got.overall.macro_map,
                     "delta": got.overall.macro_map - want.overall.macro_map,
                     "tokens": sum(cache.pages[i]["count"] for i in page_rows)})
        print(f"[mole] {A}: reference {want.overall.macro_map:.4f} → from cache "
              f"{got.overall.macro_map:.4f} ({rows[-1]['delta']:+.4f})")

    if not pairs:
        print("\n[mole] nothing to compare — check --reference")
        return

    print(f"\n\n{'=' * 72}\n== Is the token cap harmless?\n"
          f"   (same backbone/geometry/K; only the visible tokens differ)\n{'=' * 72}")
    r = compare_evals_multi(pairs, seed=args.seed)
    print(format_multi_compare(r))
    verdict = ("HARMLESS — the capped cache reproduces the deployed space"
               if abs(r.mean_delta) <= 0.005 else
               "⚠ NOT harmless — rebuild with a larger --max-tokens-per-page "
               "before trusting any NetVLAD result")
    print(f"\n   {verdict}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"cap": cap, "per_archive": rows, "mean_delta": r.mean_delta,
             "ci": [r.ci_low, r.ci_high]}, indent=2))
        print(f"[mole] ✓ {args.json}")


if __name__ == "__main__":
    main()
