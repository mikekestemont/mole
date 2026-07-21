#!/usr/bin/env python
"""GPU-free read on a trained Tier-1 head, straight from its feature cache.

The head trainer's own selection metric (``holdout_macro_map``) ranks held-out
docs against a gallery of *only* held-out docs — an easy task, pooled over all
archives. This script runs the real SUPERVISED_PLAN.md §4.2 protocol instead —
**queries = held-out-hand docs, gallery = that archive's own labeled docs** —
per archive, with and without the head, on the cached window descriptors. No
GPU, no re-embed: it answers "where does the lift land, and is it real?" before
any `mole embed --head` pass is paid for.

    python scripts/sup_probe.py runs/sup_head_v1

CAVEAT (why this is a leading indicator, not the verdict): pooling here is the
**mean** of projected window descriptors, because that is all a window cache can
express. The deployed space is VLAD over head-projected patch tokens, and mean
pooling measured ~0.16 macro *below* VLAD on this backbone. A head can win big
under mean pooling merely by suppressing nuisance directions that a flat mean
averages badly — which VLAD's cluster residuals already handle. So read a large
Δ here as "worth the GPU", not as the answer; `mole eval-compare` on real
embeddings remains the go/no-go. The gallery is also labeled-docs-only, so it
lacks the unlabeled distractors the real eval carries.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _doc_table(cache, Z: np.ndarray):
    """Doc-level L2-normalised mean embeddings + their hand / archive labels."""
    by_doc: dict[str, list[int]] = defaultdict(list)
    doc_hand: dict[str, str] = {}
    doc_arch: dict[str, str] = {}
    for i, (d, h, a) in enumerate(zip(cache.window_doc, cache.window_hand,
                                      cache.window_archive)):
        if not h:                                  # unlabeled window: not a doc here
            continue
        by_doc[d].append(i)
        doc_hand[d], doc_arch[d] = h, a
    docs = sorted(by_doc)
    emb = np.stack([Z[by_doc[d]].mean(0) for d in docs])
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return (docs, emb,
            np.asarray([doc_hand[d] for d in docs], dtype=object),
            np.asarray([doc_arch[d] for d in docs], dtype=object))


def _macro_and_per_hand(emb, hands, query_mask):
    """Held-out-hand macro-mAP over an own-archive gallery (cosine, leave-one-out)."""
    from mole.eval.retrieval import _rank_metrics

    if not query_mask.any() or len(emb) < 2:
        return None, {}
    sim = emb @ emb.T
    allow = ~np.eye(len(emb), dtype=bool)
    scores = _rank_metrics(sim, hands, allow, (1,), query_mask=query_mask)
    if scores is None:
        return None, {}
    return scores.macro_map, {h: v["ap"] for h, v in scores.per_hand.items()}


def _paired_bootstrap(deltas: list[float], n_boot: int = 10000, seed: int = 0):
    """Hand-level paired bootstrap CI on the mean per-hand ΔAP."""
    if not deltas:
        return 0.0, (0.0, 0.0)
    d = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), (float(np.percentile(means, 2.5)),
                             float(np.percentile(means, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Head run dir (cache/, split.json, head.pt).")
    ap.add_argument("--cache", type=Path, default=None, help="Override the cache dir.")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from mole.supervised.datasets import FeatureCache
    from mole.supervised.metric import build_head

    run = args.run_dir
    cache = FeatureCache.load(args.cache or run / "cache")
    holdout = set(json.loads((run / "split.json").read_text())["holdout_hands"])

    blob = torch.load(run / "head.pt", map_location="cpu", weights_only=False)
    head = build_head(blob["kind"], blob["in_dim"], blob["out_dim"])
    head.load_state_dict(blob["state_dict"])
    head.eval()
    with torch.no_grad():
        Z_head = head(torch.from_numpy(cache.descriptors)).numpy()

    print(f"cache {cache.n_windows:,} windows × {cache.dim}  |  "
          f"held-out hands {len(holdout)}  |  head {blob['kind']} "
          f"{blob['in_dim']}→{blob['out_dim']}")
    print("queries = held-out-hand docs, gallery = that archive's labeled docs, "
          "MEAN pooling\n")

    spaces = {"base": cache.descriptors, "head": Z_head}
    tables = {k: _doc_table(cache, Z) for k, Z in spaces.items()}
    _, _, hands_all, arch_all = tables["base"]
    is_holdout = np.asarray([h in holdout for h in hands_all], dtype=bool)

    all_deltas: list[float] = []
    rows: list[tuple] = []
    for archive in sorted(set(arch_all)):
        sel = arch_all == archive
        qm = is_holdout & sel
        if not qm.any():
            rows.append((archive, 0, 0, None, None, None))
            continue
        res = {}
        for key in ("base", "head"):
            _, emb, hands, _ = tables[key]
            res[key] = _macro_and_per_hand(emb[sel], hands[sel], qm[sel])
        (m_base, ph_base), (m_head, ph_head) = res["base"], res["head"]
        if m_base is None or m_head is None:
            rows.append((archive, int(qm.sum()), 0, None, None, None))
            continue
        shared = sorted(set(ph_base) & set(ph_head))
        deltas = [ph_head[h] - ph_base[h] for h in shared]
        all_deltas += deltas
        rows.append((archive, int(qm.sum()), len(shared), m_base, m_head,
                     m_head - m_base))

    w = max(len(r[0]) for r in rows)
    print(f"  {'archive':<{w}}  {'queries':>7} {'hands':>5} "
          f"{'base':>7} {'head':>7} {'Δmacro':>8}")
    for a, nq, nh, mb, mh, dl in rows:
        if mb is None:
            print(f"  {a:<{w}}  {nq:>7} {nh:>5} {'—':>7} {'—':>7} {'—':>8}")
            continue
        print(f"  {a:<{w}}  {nq:>7} {nh:>5} {mb:>7.4f} {mh:>7.4f} {dl:>+8.4f}")

    mean_d, (lo, hi) = _paired_bootstrap(all_deltas, args.n_boot, args.seed)
    ups = sum(1 for d in all_deltas if d > 0)
    downs = sum(1 for d in all_deltas if d < 0)
    verdict = "REAL — 95% CI excludes 0" if lo > 0 or hi < 0 else "INSIDE THE NOISE BAND"
    print(f"\n  pooled over {len(all_deltas)} held-out hands: "
          f"mean ΔAP {mean_d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  sign: {ups}↑ / {downs}↓   verdict: {verdict}")
    print("\n  NB mean pooling, labeled-only gallery — a leading indicator for the "
          "VLAD\n     eval, not the §4.3 go/no-go. See the module docstring.")


if __name__ == "__main__":
    main()
