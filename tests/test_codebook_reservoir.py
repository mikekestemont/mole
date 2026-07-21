"""The reservoir behind `mole codebook`: bounded memory + an unbiased sample.

The corpus codebook exists so a frozen VLAD space can be fitted over a corpus that
does not fit in RAM. Two properties carry that promise and are worth pinning:

* memory is capped by ``capacity`` no matter how long the stream is, and
* the retained sample is *uniform* over everything seen — otherwise a codebook fitted
  over several archives would silently over-represent whichever one streams first,
  which is exactly the failure a pooled codebook is meant to avoid.
"""

from __future__ import annotations

import numpy as np

from mole.embed.extract import _Reservoir


def test_fills_then_caps():
    r = _Reservoir(capacity=100, dim=4, seed=0)
    r.add(np.ones((30, 4), np.float32))
    assert r.filled == 30 and r.seen == 30
    r.add(np.ones((500, 4), np.float32))
    assert r.filled == 100            # never grows past capacity
    assert r.seen == 530              # but the stream count is exact
    assert r.sample.shape == (100, 4)


def test_short_stream_keeps_everything():
    r = _Reservoir(capacity=1000, dim=3, seed=0)
    block = np.arange(12, dtype=np.float32).reshape(4, 3)
    r.add(block)
    assert np.array_equal(r.sample, block)


def test_empty_block_is_a_noop():
    r = _Reservoir(capacity=10, dim=2, seed=0)
    r.add(np.empty((0, 2), np.float32))
    assert r.filled == 0 and r.seen == 0


def test_sample_is_uniform_over_the_stream():
    """Late items must be as likely to survive as early ones.

    Stream 20x the capacity, tagging each descriptor with its arrival order, and check
    the retained tags spread across the whole range rather than clustering at the start
    (what a truncating cap would do) or the end. Averaged over seeds so the assertion
    is about the sampler, not one lucky draw.
    """
    cap, n = 200, 4000
    means = []
    for seed in range(12):
        r = _Reservoir(capacity=cap, dim=1, seed=seed)
        for start in range(0, n, 250):                     # arrives in blocks, as pages do
            block = np.arange(start, start + 250, dtype=np.float32).reshape(-1, 1)
            r.add(block)
        assert r.filled == cap and r.seen == n
        means.append(float(r.sample.mean()))
    # A uniform sample of 0..n-1 has mean ~ (n-1)/2; truncation would give ~cap/2.
    overall = float(np.mean(means))
    assert abs(overall - (n - 1) / 2) < 0.08 * n, overall
    assert overall > 2 * cap          # decisively not the "first cap items" behaviour


def test_every_stream_position_can_survive():
    """Across seeds, retained tags reach both the first and the last decile."""
    cap, n = 50, 2000
    lo = hi = False
    for seed in range(20):
        r = _Reservoir(capacity=cap, dim=1, seed=seed)
        r.add(np.arange(n, dtype=np.float32).reshape(-1, 1))
        tags = r.sample.ravel()
        lo |= bool((tags < 0.1 * n).any())
        hi |= bool((tags > 0.9 * n).any())
    assert lo and hi
