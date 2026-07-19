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
