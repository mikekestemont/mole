#!/usr/bin/env python
"""Leave-one-archive-out test of a TRAINABLE aggregator (Tier 2 / NetVLAD).

Phase 2 established that supervision transfers across collections but cannot
survive hard-assignment VLAD (SUPERVISED_PLAN.md §0a, F1–F3). This script asks
the follow-up directly: if the loss runs *through* the aggregation, does the
supervision survive to the space retrieval actually ranks?

For each archive A:

  1. fit a K-means codebook C_A on the FOUR OTHER archives' cached tokens
     — A contributes nothing, not even unsupervised;
  2. baseline  = plain hard VLAD over A's tokens with C_A;
  3. candidate = NetVLAD initialised from C_A and trained on the four others
     (model-selected on a held-out slice of THEIR hands: train/select/test are
     three disjoint hand sets);
  4. evaluate both on A's own gallery, all hands as queries, cross-document
     relevance — the same protocol as `scripts/run_loao.sh`.

Three spaces are evaluated per fold, not two, because the equivalence between
soft and hard VLAD holds only at a sufficiently sharp alpha and that is a
measured property, not an assumed one:

  frozen    hard-assignment VLAD with C_A                     the baseline
  init      the UNTRAINED NetVLAD (same C_A, same alpha)      the control
  netvlad   the trained aggregator                            the candidate

`init` costs one extra CPU pass and removes the entire confound: `netvlad-init`
isolates training at any alpha, while `init-frozen` measures how much the soft
aggregator itself moved. If the latter is not ~0, alpha is too low -- raise it or
read the headline against `init`.

Three verdicts are printed, and they answer different questions:

  init vs frozen   is the aggregator unchanged at init?    sanity: ~0
  netvlad vs init  does learning the aggregator help?      bar: >= +0.02, CI > 0
  vs transductive  is it shippable?                        bar: >= 0

The second bar is the harsh one: deployment refits a codebook per archive, worth
about +0.032 macro over a frozen one, and a learned codebook is inherently
frozen. NetVLAD has to pay that back before it is a deployment win.

    python scripts/run_netvlad_loao.py runs/sup_tokens --out outputs/netvlad_loao

Everything here is CPU and reads ONE token cache (`mole sup tokens`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", type=Path, help="Token cache dir (`mole sup tokens`).")
    ap.add_argument("--data", type=Path, default=Path("data"),
                    help="Root holding the per-archive dataset dirs (for labels).")
    ap.add_argument("--out", type=Path, default=Path("outputs/netvlad_loao"))
    ap.add_argument("--runs", type=Path, default=Path("runs/netvlad_loao"))
    ap.add_argument("--archives", nargs="*", default=None,
                    help="Default: every archive in the cache.")
    ap.add_argument("--transductive", type=Path, default=Path("outputs/pooled_final"),
                    help="Dir of per-archive deployed embeddings for the second "
                         "verdict; skipped if absent.")
    ap.add_argument("--clusters", type=int, default=100)
    ap.add_argument("--learn", default="both", choices=["both", "assign", "centroids"],
                    help="Ablation: train the assignment, the centres, or both.")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Softmax sharpness, absolute. Default = calibrated so the "
                         "UNTRAINED module reproduces hard VLAD.")
    ap.add_argument("--alpha-scale", type=float, default=1.0,
                    help="Multiply the calibrated alpha. Values BELOW 1 trade "
                         "fidelity for capacity, and that trade is the experiment: "
                         "at the calibrated alpha the softmax is a hard argmax, so "
                         "assign_c is piecewise constant (no gradient) and the "
                         "centroids can only apply an occupancy-scaled offset -- the "
                         "module is pinned to the baseline it was initialised from. "
                         "Softening alpha buys capacity and costs init fidelity, "
                         "which is exactly why the driver evaluates @init separately: "
                         "netvlad-vs-init stays valid, while netvlad-vs-frozen says "
                         "whether the capacity paid for the fidelity.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--tokens-per-page", type=int, default=512)
    ap.add_argument("--select-max-tokens", type=int, default=0,
                    help="Cap tokens per page in the PER-EPOCH model-selection pass "
                         "(0 = all). This runs every epoch over the holdout pages and "
                         "is the dominant training cost; it only picks the best epoch, "
                         "so capping it trades a slightly noisier stopping rule for a "
                         "much shorter run. The reported numbers always use all tokens.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--batches-per-epoch", type=int, default=100)
    ap.add_argument("--hands-per-batch", type=int, default=16)
    ap.add_argument("--docs-per-hand", type=int, default=2)
    ap.add_argument("--same-archive-frac", type=float, default=0.5)
    ap.add_argument("--codebook-descriptors", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from mole.embed.vlad import fit_codebook
    from mole.eval.compare import compare_evals_multi, format_multi_compare
    from mole.eval.retrieval import evaluate
    from mole.supervised.netvlad import (
        NetVLAD, netvlad_page_vectors, save_netvlad, train_netvlad,
        vlad_page_vectors, write_embeddings)
    from mole.supervised.tokens import TokenCache, descriptor_pool

    cache = TokenCache.load(args.cache)
    print(f"[mole] token cache: {cache.stats()}\n")
    archives = args.archives or cache.archives
    args.out.mkdir(parents=True, exist_ok=True)

    init_pairs: list[tuple[Path, Path]] = []
    frozen_pairs: list[tuple[Path, Path]] = []
    trans_pairs: list[tuple[Path, Path]] = []
    summary: list[dict] = []

    for A in archives:
        print(f"\n{'=' * 72}\n== fold: {A} held out\n{'=' * 72}")
        test_rows = cache.rows_for(archive=A)
        train_rows = [i for i in range(cache.n_pages) if cache.pages[i]["archive"] != A]
        if not test_rows or not train_rows:
            print(f"[mole] skipping {A}: {len(test_rows)} test / {len(train_rows)} train pages")
            continue
        dataset_dir = args.data / A
        run_dir = args.runs / A
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. the fold's frozen codebook — fit WITHOUT the held-out archive.
        cb_path = run_dir / "codebook.npy"
        if cb_path.is_file():
            codebook = np.load(cb_path)
            print(f"[mole] reusing fold codebook {cb_path}")
        else:
            pool = descriptor_pool(cache, train_rows,
                                   max_descriptors=args.codebook_descriptors,
                                   seed=args.seed)
            print(f"[mole] fitting K={args.clusters} codebook on {len(pool):,} "
                  f"descriptors from {len(train_rows)} pages (excluding {A})")
            codebook = fit_codebook(pool, n_clusters=args.clusters, seed=args.seed)
            np.save(cb_path, codebook)

        # 2. baseline: hard VLAD with that codebook, straight from the cache.
        base_npy = args.out / f"{A}.frozen.npy"
        base_eval = args.out / f"{A}.frozen.eval.json"
        base = vlad_page_vectors(cache, codebook, test_rows)
        write_embeddings(base_npy, base, cache, test_rows,
                         {"pooling": "vlad", "aggregator": "hard-kmeans",
                          "fold": A, "vlad_clusters": int(codebook.shape[0])},
                         dataset_dir=dataset_dir)
        r_base = evaluate(base_npy, dataset_dir, topk=(1, 5), cross_doc_only=True,
                          out=base_eval)

        # 3. candidate: same codebook, made trainable, trained on the others.
        excluded = {p["hand"] for p in cache.pages
                    if p["archive"] == A and p["hand"]}
        select_hands = _select_hands(cache, exclude=excluded,
                                     frac=args.holdout_frac, seed=args.seed)
        alpha = args.alpha
        if alpha is None and args.alpha_scale != 1.0:
            from mole.supervised.netvlad import alpha_for_codebook
            rng = np.random.default_rng(args.seed)
            probe = [cache.sample_tokens(i, 256, rng)
                     for i in [r for r in train_rows if cache.pages[r]["hand"]][:32]]
            alpha = alpha_for_codebook(codebook, probe) * args.alpha_scale
            print(f"[mole] alpha = {alpha:.4g} "
                  f"({args.alpha_scale}x the fidelity-calibrated value)")
        model, report = train_netvlad(
            cache, codebook, holdout_hands=select_hands, exclude_hands=excluded,
            alpha=alpha, learn=args.learn, epochs=args.epochs,
            tokens_per_page=args.tokens_per_page, lr=args.lr, seed=args.seed,
            select_max_tokens=args.select_max_tokens, device=args.device,
            sampler_cfg={"hands_per_batch": args.hands_per_batch,
                         "docs_per_hand": args.docs_per_hand,
                         "same_archive_frac": args.same_archive_frac,
                         "batches_per_epoch": args.batches_per_epoch})
        report["holdout_archive"] = A
        save_netvlad(run_dir / "netvlad.pt", model, report)
        (run_dir / "report.json").write_text(json.dumps(report, indent=2))
        h = report["history"]
        print(f"[mole] trained: best epoch {report['best_epoch']}/{args.epochs}, "
              f"select macro {report['best_holdout_macro']:.4f}, "
              f"loss {h[0]['loss']:.3f}→{h[-1]['loss']:.3f}, "
              f"assign entropy {h[0]['assign_entropy']:.3f}→{h[-1]['assign_entropy']:.3f} "
              f"(ln K = {np.log(args.clusters):.3f})")
        print(f"[mole]   grads: assign {h[-1]['grad_assign']:.3g} vs centroids "
              f"{h[-1]['grad_centroids']:.3g}  |  SEEN-hand macro "
              f"{report['train_fit_before']:.4f}→{report['train_fit_after']:.4f} "
              f"({report['train_fit_delta']:+.4f})")

        # 3b. the control: the SAME module before any training. At a faithful α
        # this reproduces `frozen`; when it does not, it — not `frozen` — is what
        # the headline Δ must be read against.
        init_npy = args.out / f"{A}.init.npy"
        init_eval = args.out / f"{A}.init.eval.json"
        init_model = NetVLAD.from_codebook(codebook, report["alpha"], learn=args.learn)
        init = netvlad_page_vectors(init_model, cache, test_rows, device=args.device)
        write_embeddings(init_npy, init, cache, test_rows,
                         {"pooling": "vlad", "aggregator": "netvlad-init", "fold": A,
                          "vlad_clusters": int(init_model.num_clusters),
                          "alpha": report["alpha"]},
                         dataset_dir=dataset_dir)
        r_init = evaluate(init_npy, dataset_dir, topk=(1, 5), cross_doc_only=True,
                          out=init_eval)

        head_npy = args.out / f"{A}.netvlad.npy"
        head_eval = args.out / f"{A}.netvlad.eval.json"
        cand = netvlad_page_vectors(model, cache, test_rows, device=args.device)
        write_embeddings(head_npy, cand, cache, test_rows,
                         {"pooling": "vlad", "aggregator": "netvlad", "fold": A,
                          "vlad_clusters": int(model.num_clusters),
                          "alpha": report["alpha"], "learn": args.learn},
                         dataset_dir=dataset_dir)
        r_cand = evaluate(head_npy, dataset_dir, topk=(1, 5), cross_doc_only=True,
                          out=head_eval)

        init_pairs.append((base_eval, init_eval))
        frozen_pairs.append((init_eval, head_eval))
        row = {"archive": A, "frozen": r_base.overall.macro_map,
               "init": r_init.overall.macro_map,
               "netvlad": r_cand.overall.macro_map,
               "best_epoch": report["best_epoch"],
               "select_macro": report["best_holdout_macro"],
               "alpha": report["alpha"],
               "init_fidelity": report["init_fidelity"],
               "train_fit_delta": report["train_fit_delta"]}

        trans = args.transductive / f"{A}.npy"
        if trans.is_file():
            trans_eval = args.out / f"{A}.transductive.eval.json"
            r_tr = evaluate(trans, dataset_dir, topk=(1, 5), cross_doc_only=True,
                            out=trans_eval)
            trans_pairs.append((trans_eval, head_eval))
            row["transductive"] = r_tr.overall.macro_map
        else:
            print(f"[mole] note: {trans} not found — skipping the deployment verdict "
                  f"for {A}")
        summary.append(row)
        print(f"[mole] {A}: frozen {row['frozen']:.4f} → init {row['init']:.4f} "
              f"→ netvlad {row['netvlad']:.4f}  "
              f"(training {row['netvlad'] - row['init']:+.4f}, "
              f"aggregator {row['init'] - row['frozen']:+.4f}, "
              f"init fidelity {row['init_fidelity']:.4f})")

    # ------------------------------------------------------------- verdicts
    print(f"\n\n{'=' * 72}\n== SANITY — is the untrained aggregator still the baseline?\n"
          f"   (NetVLAD@init vs hard VLAD, same codebook — should be ≈ 0)\n"
          f"{'=' * 72}")
    v0 = compare_evals_multi(init_pairs, seed=args.seed)
    print(format_multi_compare(v0))
    if abs(v0.mean_delta) > 0.005:
        print(f"\n   ⚠ the soft aggregator alone moved macro by {v0.mean_delta:+.4f} — "
              f"α is too low.\n     Raise it (or read VERDICT 1, which is measured "
              f"against @init and stays valid).")

    print(f"\n\n{'=' * 72}\n== VERDICT 1 — does learning the aggregator help?\n"
          f"   (trained vs its OWN initialisation: same tokens, codebook, α, K)\n"
          f"{'=' * 72}")
    v1 = compare_evals_multi(frozen_pairs, seed=args.seed)
    print(format_multi_compare(v1))
    print("\n   bar: mean Δmacro ≥ +0.02, CI excluding 0, no archive < −0.01")

    v2 = None
    if trans_pairs:
        print(f"\n\n{'=' * 72}\n== VERDICT 2 — is it shippable?\n"
              f"   (NetVLAD vs the DEPLOYED per-archive transductive codebook)\n"
              f"{'=' * 72}")
        v2 = compare_evals_multi(trans_pairs, seed=args.seed)
        print(format_multi_compare(v2))
        print("\n   bar: mean Δmacro ≥ 0 — a learned codebook is frozen by "
              "construction,\n   so this is where the ~0.032 frozen-codebook tax "
              "has to be paid back.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "summary": summary,
            "init_vs_frozen": _verdict_dict(v0),
            "vs_init": _verdict_dict(v1),
            "vs_transductive": _verdict_dict(v2) if v2 else None,
            "args": {k: str(v) for k, v in vars(args).items()},
        }, indent=2))
        print(f"\n[mole] ✓ {args.json}")


def _select_hands(cache, *, exclude: set[str], frac: float, seed: int) -> set[str]:
    """A model-selection slice of the TRAINING archives' hands, archive-stratified.

    Selection must never touch the held-out archive — that would leak the test
    set into the stopping rule — so the slice is drawn from what remains.
    """
    by_archive: dict[str, list[str]] = {}
    for p in cache.pages:
        if p["hand"] and p["hand"] not in exclude:
            by_archive.setdefault(p["archive"], []).append(p["hand"])
    rng = np.random.default_rng(seed)
    out: set[str] = set()
    for archive, hands in sorted(by_archive.items()):
        uniq = sorted(set(hands))
        n = max(1, int(round(len(uniq) * frac)))
        out.update(rng.choice(uniq, size=min(n, len(uniq)), replace=False).tolist())
    return out


def _verdict_dict(r) -> dict:
    return {"mean_delta": r.mean_delta, "ci": [r.ci_low, r.ci_high],
            "ci_excludes_zero": r.ci_excludes_zero, "guardrail_ok": r.guardrail_ok,
            "worst": [r.worst_label, r.worst_delta],
            "per_archive": {lab: p.delta_macro for lab, p in zip(r.labels, r.pairs)}}


if __name__ == "__main__":
    main()
