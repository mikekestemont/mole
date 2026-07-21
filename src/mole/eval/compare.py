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


# --------------------------------------------------- several archives at once
# §4.2: "mean over the 5 archives of held-out Δmacro, bootstrap CI excluding 0,
# AND no archive worse than -0.01".
GUARDRAIL = -0.01


@dataclass
class MultiCompareReport:
    """The §4.2 decision rule over several per-archive comparisons."""

    pairs: list[CompareReport] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    mean_delta: float = 0.0      # ARCHIVE-weighted: mean of the per-archive Δmacro
    ci_low: float = 0.0
    ci_high: float = 0.0
    ci_excludes_zero: bool = False
    pooled_delta: float = 0.0    # HAND-weighted, for reference (big archives dominate)
    n_hands: int = 0
    worst_label: str = ""
    worst_delta: float = 0.0
    guardrail_ok: bool = False
    n_boot: int = 0
    seed: int = 0

    @property
    def passes(self) -> bool:
        """The full rule: positive, distinguishable from 0, and nobody regresses."""
        return self.mean_delta > 0 and self.ci_excludes_zero and self.guardrail_ok


def _dataset_label(path: str | Path) -> str:
    """Name a comparison by the archive it covers (falls back to the filename)."""
    try:
        data = json.loads(Path(path).read_text())
        ds = data.get("datasets") or []
        if ds:
            return ",".join(str(d) for d in ds)
    except (OSError, ValueError):
        pass
    return Path(path).name.split(".")[0]


def compare_evals_multi(pairs: list[tuple[str | Path, str | Path]], *,
                        section: str = "overall", n_boot: int = 10_000,
                        seed: int = 0) -> MultiCompareReport:
    """Combine per-archive paired comparisons into one verdict.

    Each archive is compared on its own (paired per-hand, own gallery — never a
    pooled gallery, which would change the task). The headline is the **mean of
    the per-archive Δmacro**, so a 90-hand archive cannot outvote a 3-hand one;
    the CI comes from resampling hands *within* each archive and re-averaging,
    which propagates small-archive noise honestly. The hand-weighted pooled Δ is
    reported alongside for contrast.
    """
    reports = [compare_evals(a, b, section=section, n_boot=n_boot, seed=seed)
               for a, b in pairs]
    labels = [_dataset_label(b) for _, b in pairs]

    per_archive = [np.array(list(r.per_hand_delta.values())) for r in reports]
    means = np.array([d.mean() for d in per_archive])

    rng = np.random.default_rng(seed)
    boot = np.zeros((n_boot, len(per_archive)))
    for j, d in enumerate(per_archive):
        boot[:, j] = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    boot_mean = boot.mean(axis=1)
    ci_low, ci_high = (float(np.percentile(boot_mean, 2.5)),
                       float(np.percentile(boot_mean, 97.5)))

    all_deltas = np.concatenate(per_archive)
    worst = int(np.argmin(means))
    return MultiCompareReport(
        pairs=reports, labels=labels,
        mean_delta=float(means.mean()), ci_low=ci_low, ci_high=ci_high,
        ci_excludes_zero=(ci_low > 0.0 or ci_high < 0.0),
        pooled_delta=float(all_deltas.mean()), n_hands=int(all_deltas.size),
        worst_label=labels[worst], worst_delta=float(means[worst]),
        guardrail_ok=bool(means.min() >= GUARDRAIL),
        n_boot=n_boot, seed=seed,
    )


def format_multi_compare(r: MultiCompareReport) -> str:
    w = max((len(x) for x in r.labels), default=7)
    out = [f"eval-compare ({r.pairs[0].section})  B vs A over {len(r.pairs)} archives",
           "",
           f"  {'archive':<{w}}  {'hands':>5} {'A':>7} {'B':>7} {'Δmacro':>8}"]
    for label, p in zip(r.labels, r.pairs):
        flag = "  ⚠" if p.delta_macro < GUARDRAIL else ""
        out.append(f"  {label:<{w}}  {p.n_shared:>5} {p.macro_a:>7.4f} "
                   f"{p.macro_b:>7.4f} {p.delta_macro:>+8.4f}{flag}")
    verdict = ("REAL — 95% CI excludes 0" if r.ci_excludes_zero
               else "not distinguishable from 0 (CI includes 0)")
    out += [
        "",
        f"  mean Δmacro : {r.mean_delta:+.4f}   95% CI [{r.ci_low:+.4f}, "
        f"{r.ci_high:+.4f}]  ({r.n_boot} boot, seed {r.seed})",
        f"  verdict     : {verdict}",
        f"  guardrail   : " + ("all archives ≥ -0.01" if r.guardrail_ok else
                               f"FAILED — {r.worst_label} {r.worst_delta:+.4f} "
                               f"(< {GUARDRAIL})"),
        f"  (hand-weighted Δ over all {r.n_hands} hands: {r.pooled_delta:+.4f})",
    ]
    return "\n".join(out)


def format_compare(r: CompareReport, *, top: int = 5) -> str:
    verdict = ("REAL — 95% CI excludes 0" if r.ci_excludes_zero
               else "not distinguishable from 0 (CI includes 0)")
    arrow = "↑" if r.delta_macro > 0 else ("↓" if r.delta_macro < 0 else "=")
    out = [
        f"eval-compare ({r.section})  B vs A",
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
