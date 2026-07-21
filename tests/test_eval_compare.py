"""Tests for the paired per-hand eval comparison (mole.eval.compare)."""

from __future__ import annotations

import json

from mole.eval.compare import _sign_test_p, compare_evals


def _write_eval(path, per_hand: dict[str, float], section="overall"):
    block = {"per_hand": {h: {"ap": ap, "n_queries": 2} for h, ap in per_hand.items()},
             "mean_ap": 0.0, "macro_map": 0.0, "top1": 0.0, "topk": {}}
    path.write_text(json.dumps({section: block}))


def test_sign_test_symmetric_and_extremes():
    assert _sign_test_p(0, 0) == 1.0
    assert _sign_test_p(5, 5) == 1.0             # perfectly split -> p=1
    assert _sign_test_p(10, 0) < 0.01            # all one direction -> tiny p
    assert abs(_sign_test_p(3, 7) - _sign_test_p(7, 3)) < 1e-12  # symmetric


def test_compare_detects_uniform_improvement(tmp_path):
    """Every hand improves by +0.1 -> Δmacro +0.1, CI excludes 0, all-up sign test."""
    a = tmp_path / "a.eval.json"
    b = tmp_path / "b.eval.json"
    hands = {f"h{i}": 0.5 for i in range(12)}
    _write_eval(a, hands)
    _write_eval(b, {h: v + 0.1 for h, v in hands.items()})
    r = compare_evals(a, b, n_boot=2000, seed=0)
    assert r.n_shared == 12
    assert abs(r.delta_macro - 0.1) < 1e-9
    assert r.ci_excludes_zero          # a uniform shift has zero bootstrap variance
    assert r.n_up == 12 and r.n_down == 0
    assert r.sign_p < 0.01


def test_compare_noise_does_not_exclude_zero(tmp_path):
    """Symmetric +/- deltas -> Δmacro ~0, CI includes 0."""
    a = tmp_path / "a.eval.json"
    b = tmp_path / "b.eval.json"
    base = {f"h{i}": 0.5 for i in range(10)}
    _write_eval(a, base)
    # half up 0.2, half down 0.2 -> mean delta 0
    bumped = {h: (0.7 if i % 2 == 0 else 0.3) for i, h in enumerate(base)}
    _write_eval(b, bumped)
    r = compare_evals(a, b, n_boot=2000, seed=0)
    assert abs(r.delta_macro) < 1e-9
    assert not r.ci_excludes_zero
    assert r.n_up == 5 and r.n_down == 5


def test_compare_pairs_only_shared_hands(tmp_path):
    a = tmp_path / "a.eval.json"
    b = tmp_path / "b.eval.json"
    _write_eval(a, {"x": 0.4, "y": 0.6, "onlyA": 0.9})
    _write_eval(b, {"x": 0.5, "y": 0.7, "onlyB": 0.1})
    r = compare_evals(a, b, n_boot=500, seed=1)
    assert r.n_shared == 2
    assert r.only_a == ["onlyA"] and r.only_b == ["onlyB"]
    assert set(r.per_hand_delta) == {"x", "y"}


# ------------------------------------------------- §4.2 multi-archive rule
def _pair(tmp_path, name, base: dict[str, float], delta: float):
    a, b = tmp_path / f"{name}.a.json", tmp_path / f"{name}.b.json"
    _write_eval(a, base)
    _write_eval(b, {h: v + delta for h, v in base.items()})
    return a, b


def test_multi_compare_mean_is_archive_weighted_not_hand_weighted(tmp_path):
    """A 3-hand archive counts as much as a 90-hand one (the §4.2 rule)."""
    from mole.eval.compare import compare_evals_multi

    small = _pair(tmp_path, "small", {f"s{i}": 0.5 for i in range(3)}, +0.20)
    big = _pair(tmp_path, "big", {f"b{i}": 0.5 for i in range(90)}, 0.00)
    r = compare_evals_multi([small, big], n_boot=2000, seed=0)

    assert abs(r.mean_delta - 0.10) < 1e-9        # (0.20 + 0.00) / 2
    assert abs(r.pooled_delta - (0.20 * 3 / 93)) < 1e-9   # hand-weighted differs sharply
    assert r.n_hands == 93
    assert [p.n_shared for p in r.pairs] == [3, 90]


def test_multi_compare_guardrail_catches_one_regressed_archive(tmp_path):
    """Mean can be strongly positive while one archive regresses — the rule fails."""
    from mole.eval.compare import compare_evals_multi

    good = _pair(tmp_path, "good", {f"g{i}": 0.5 for i in range(10)}, +0.10)
    bad = _pair(tmp_path, "bad", {f"d{i}": 0.5 for i in range(10)}, -0.05)
    r = compare_evals_multi([good, bad], n_boot=2000, seed=0)

    assert r.mean_delta > 0 and r.ci_excludes_zero
    assert not r.guardrail_ok
    assert r.worst_delta < -0.01
    assert not r.passes                            # positive mean is NOT enough
