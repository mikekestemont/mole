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
experiment can be trusted. A systematic loss says `--max-tokens-per-page` was
too aggressive — rebuild the cache with a larger cap (or 0 for all tokens)
before reading anything into a trained aggregator.

MEASURED 2026-07-22, cap 2048: mean Δmacro **-0.0107**, CI [-0.0221, -0.0006],
guardrail failed on Flanders (-0.0259), Utrecht -0.0127, Antwerp -0.0115. So the
cap was NOT free on these archives. The saturation result it was extrapolated
from (HWI: 8,800 -> 2,900 tokens/page = +0.0006) never tested below ~2,900,
while these pages average ~5,250 foreground tokens — 2,048 sits past the end of
the evidence. Run uncapped; the check then doubles as the float16 fidelity test,
since the cap was the only other discrepancy from the deployed embeddings.
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
    ap.add_argument("--codebook-descriptors", type=int, default=0,
                    help="Descriptors used to fit each archive's codebook. 0 = ALL, "
                         "matching `mole embed`'s default; anything else adds a "
                         "subsample confound on top of the cap being tested.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-floor", type=int, nargs="*", default=None,
                    metavar="SEED",
                    help="Extra codebook seeds to refit the SAME cache at, e.g. "
                         "--noise-floor 1 2 3. MiniBatchKMeans is order- and "
                         "seed-sensitive, so two honest fits of identical data give "
                         "different centres; this measures how much macro-mAP moves "
                         "for that reason alone. Any Δ smaller than this spread — "
                         "here or in the LOAO experiment — is not a result.")
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

    seeds = [args.seed] + list(args.noise_floor or [])

    def encode_at(A, page_rows, seed, tag):
        """Fit this archive's own codebook from the cache at `seed`, encode, eval."""
        pool = descriptor_pool(cache, page_rows,
                               max_descriptors=args.codebook_descriptors, seed=seed)
        codebook = fit_codebook(pool, n_clusters=args.clusters, seed=seed)
        npy = args.out / f"{A}.{tag}.npy"
        write_embeddings(npy, vlad_page_vectors(cache, codebook, page_rows),
                         cache, page_rows,
                         {"pooling": "vlad", "aggregator": "hard-kmeans",
                          "vlad_clusters": int(codebook.shape[0]), "vlad_seed": seed,
                          "note": "transductive, refit from the token cache"},
                         dataset_dir=args.data / A)
        return evaluate(npy, args.data / A, topk=(1, 5),
                        cross_doc_only=args.cross_doc_only,
                        out=args.out / f"{A}.{tag}.eval.json").overall.macro_map

    pairs, rows = [], []
    for A in (args.archives or cache.archives):
        ref = args.reference / f"{A}.npy"
        if not ref.is_file():
            print(f"[mole] skipping {A}: no reference at {ref}")
            continue
        print(f"\n== {A}")
        page_rows = cache.rows_for(archive=A)
        want = evaluate(ref, args.data / A, topk=(1, 5),
                        cross_doc_only=args.cross_doc_only,
                        out=args.out / f"{A}.reference.eval.json")

        macros = [encode_at(A, page_rows, s, "cache" if s == args.seed else f"seed{s}")
                  for s in seeds]
        pairs.append((args.out / f"{A}.reference.eval.json",
                      args.out / f"{A}.cache.eval.json"))
        row = {"archive": A, "reference": want.overall.macro_map,
               "from_cache": macros[0],
               "delta": macros[0] - want.overall.macro_map,
               "seed_macros": macros,
               "tokens": sum(cache.pages[i]["count"] for i in page_rows)}
        if len(macros) > 1:
            row["seed_spread"] = max(macros) - min(macros)
        rows.append(row)
        note = (f"  |  across {len(macros)} codebook seeds: "
                f"{min(macros):.4f}–{max(macros):.4f} (spread {row['seed_spread']:.4f})"
                if len(macros) > 1 else "")
        print(f"[mole] {A}: reference {want.overall.macro_map:.4f} → from cache "
              f"{macros[0]:.4f} ({row['delta']:+.4f}){note}")

    if not pairs:
        print("\n[mole] nothing to compare — check --reference")
        return

    print(f"\n\n{'=' * 72}\n== Is the token cap harmless?\n"
          f"   (same backbone/geometry/K; only the visible tokens differ)\n{'=' * 72}")
    r = compare_evals_multi(pairs, seed=args.seed)
    print(format_multi_compare(r))
    spreads = [x["seed_spread"] for x in rows if "seed_spread" in x]
    if spreads:
        print(f"\n{'=' * 72}\n== Noise floor — the SAME cache refit at "
              f"{len(seeds)} codebook seeds\n"
              f"   (identical data; only k-means initialisation differs)\n{'=' * 72}")
        w = max(len(x["archive"]) for x in rows)
        print(f"  {'archive':<{w}}  {'spread':>8}  {'|Δ vs ref|':>10}  verdict")
        for x in rows:
            sp = x.get("seed_spread", 0.0)
            flag = ("within noise" if abs(x["delta"]) <= sp
                    else "EXCEEDS noise floor")
            print(f"  {x['archive']:<{w}}  {sp:>8.4f}  {abs(x['delta']):>10.4f}  {flag}")
        worst = max(spreads)
        print(f"\n  Largest per-archive seed spread: {worst:.4f} — no Δ below this, "
              f"here or in\n  the LOAO experiment, is a result.")

    # The mean can look fine while one archive quietly fails the guardrail, which
    # is exactly the case this check exists to catch — so BOTH have to pass.
    mean_ok = abs(r.mean_delta) <= 0.005
    verdict = ("HARMLESS — the cache reproduces the deployed space"
               if mean_ok and r.guardrail_ok else
               f"⚠ mean Δ {r.mean_delta:+.4f} "
               f"({'ok' if mean_ok else 'too large'}), guardrail "
               f"{'ok' if r.guardrail_ok else f'FAILED on {r.worst_label} {r.worst_delta:+.4f}'}"
               + ("\n   → compare against the noise floor above before rebuilding: a "
                  "single-archive\n     miss inside the seed spread is k-means "
                  "initialisation, not lost information."
                  if spreads else
                  "\n   → re-run with --noise-floor 1 2 to see whether that is real "
                  "or k-means noise."))
    print(f"\n   {verdict}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"cap": cap, "seeds": seeds, "per_archive": rows,
             "mean_delta": r.mean_delta, "ci": [r.ci_low, r.ci_high],
             "guardrail_ok": r.guardrail_ok}, indent=2))
        print(f"[mole] ✓ {args.json}")


if __name__ == "__main__":
    main()
