"""Paired per-hand comparison of two retrieval evals.

Under partial / auto-matched labels a small macro-mAP lift (+0.03) can drown in
noise (§4.3 of ``SUPERVISED_PLAN.md``). The honest test is *paired*: the same
hands are queried both sides, so we compare per-hand AP head-to-head and ask
whether the mean difference is distinguishable from zero.

Given two ``eval.json`` files (each carrying ``overall.per_hand`` from
:mod:`mole.eval.retrieval`), this:

* pairs hands present in both, computes ΔAP = B − A per hand;
* reports Δmacro over the shared hands;
* puts a 95% **hand-level bootstrap** CI on Δmacro (resample hands with
  replacement) — a lift is called *real* iff the CI excludes 0;
* runs a two-sided **sign test** on the count of improved vs. regressed hands.

Both sides must have been produced the same way (same archive, same
``--cross-doc-only`` / ``--holdout-hands`` settings) for the pairing to be
meaningful; only the model/embedding should differ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

import numpy as np


@dataclass
class CompareReport:
    a: str
    b: str
    section: str
    n_shared: int
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)
    macro_a: float = 0.0        # mean AP over shared hands, side A
    macro_b: float = 0.0        # mean AP over shared hands, side B
    delta_macro: float = 0.0    # macro_b - macro_a
    ci_low: float = 0.0
    ci_high: float = 0.0
    ci_excludes_zero: bool = False
    n_boot: int = 0
    seed: int = 0
    n_up: int = 0               # hands where B > A
    n_down: int = 0             # hands where B < A
    n_tie: int = 0
    sign_p: float = 1.0
    per_hand_delta: dict[str, float] = field(default_factory=dict)


def _per_hand(path: str | Path, section: str) -> dict[str, float]:
    data = json.loads(Path(path).read_text())
    block = data.get(section)
    if not block or "per_hand" not in block:
        raise ValueError(
            f"{path} has no '{section}.per_hand' — re-run `mole eval` (per-hand AP "
            "is written since the Phase-0 measurement update)")
    return {h: float(s["ap"]) for h, s in block["per_hand"].items()}


def _sign_test_p(n_up: int, n_down: int) -> float:
    """Exact two-sided sign-test p-value under Binom(n_up+n_down, 0.5)."""
    n = n_up + n_down
    if n == 0:
        return 1.0
    k = min(n_up, n_down)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def compare_evals(a_path: str | Path, b_path: str | Path, *,
                  section: str = "overall", n_boot: int = 10_000,
                  seed: int = 0) -> CompareReport:
    """Paired per-hand comparison of eval B against baseline eval A."""
    a = _per_hand(a_path, section)
    b = _per_hand(b_path, section)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("the two evals share no hand — nothing to compare "
                         "(are they the same archive / same section?)")

    deltas = np.array([b[h] - a[h] for h in shared])
    per_hand_delta = {h: float(b[h] - a[h]) for h in shared}
    macro_a = float(np.mean([a[h] for h in shared]))
    macro_b = float(np.mean([b[h] for h in shared]))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    boot = deltas[idx].mean(axis=1)
    ci_low, ci_high = (float(np.percentile(boot, 2.5)),
                       float(np.percentile(boot, 97.5)))

    n_up = int((deltas > 0).sum())
    n_down = int((deltas < 0).sum())
    n_tie = int((deltas == 0).sum())

    return CompareReport(
        a=str(a_path), b=str(b_path), section=section, n_shared=len(shared),
        only_a=sorted(set(a) - set(b)), only_b=sorted(set(b) - set(a)),
        macro_a=macro_a, macro_b=macro_b, delta_macro=macro_b - macro_a,
        ci_low=ci_low, ci_high=ci_high,
        ci_excludes_zero=(ci_low > 0.0 or ci_high < 0.0),
        n_boot=n_boot, seed=seed,
        n_up=n_up, n_down=n_down, n_tie=n_tie,
        sign_p=_sign_test_p(n_up, n_down),
        per_hand_delta=per_hand_delta,
    )


def format_compare(r: CompareReport, *, top: int = 5) -> str:
    verdict = ("REAL — 95% CI excludes 0" if r.ci_excludes_zero
               else "not distinguishable from 0 (CI includes 0)")
    arrow = "↑" if r.delta_macro > 0 else ("↓" if r.delta_macro < 0 else "=")
    out = [
        f"eval-compare [{r.section}]  B vs A",
        f"  A: {r.a}",
        f"  B: {r.b}",
        f"  shared hands: {r.n_shared}"
        + (f"  (only A: {len(r.only_a)}, only B: {len(r.only_b)})"
           if r.only_a or r.only_b else ""),
        "",
        f"  macro A → B : {r.macro_a:.4f} → {r.macro_b:.4f}",
        f"  Δmacro {arrow}   : {r.delta_macro:+.4f}   "
        f"95% CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}]  ({r.n_boot} boot, seed {r.seed})",
        f"  verdict     : {verdict}",
        f"  sign test   : {r.n_up}↑ / {r.n_down}↓ / {r.n_tie}=  (p={r.sign_p:.3g})",
    ]
    if r.per_hand_delta:
        ranked = sorted(r.per_hand_delta.items(), key=lambda kv: kv[1])
        regress = [kv for kv in ranked if kv[1] < 0][:top]
        improve = list(reversed([kv for kv in ranked if kv[1] > 0][-top:]))
        if improve:
            out.append("  biggest gains : "
                       + ", ".join(f"{h} {d:+.3f}" for h, d in improve))
        if regress:
            out.append("  biggest drops : "
                       + ", ".join(f"{h} {d:+.3f}" for h, d in regress))
    return "\n".join(out)
