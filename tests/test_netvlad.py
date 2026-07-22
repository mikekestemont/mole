"""Tier-2 NetVLAD: the aggregator must start as an exact copy of the baseline.

The load-bearing test is :func:`test_init_reproduces_hard_vlad`. The whole LOAO
experiment reads Δ against the frozen-codebook baseline, and that Δ only means
"learning helped" if the untrained module is numerically the baseline. If this
test ever fails, every NetVLAD number in the plan is confounded by an aggregator
change that has nothing to do with supervision.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mole.embed.vlad import fit_codebook, vlad_encode          # noqa: E402
from mole.supervised.netvlad import (                          # noqa: E402
    NetVLAD, alpha_for_codebook, netvlad_page_vectors, train_netvlad,
    vlad_fidelity, vlad_page_vectors, write_embeddings)
from mole.supervised.tokens import TokenCache, descriptor_pool  # noqa: E402


def _clustered(n_clusters=6, per=40, dim=8, seed=0, spread=0.15):
    """Well-separated blobs: hard and soft assignment should agree on these."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim)).astype(np.float32) * 3.0
    x = np.vstack([c + rng.normal(scale=spread, size=(per, dim)) for c in centres])
    return x.astype(np.float32), centres


# -------------------------------------------------------------- the crux test
def test_init_reproduces_hard_vlad():
    """NetVLAD@init == vlad_encode at the calibrated α, so Δ starts at 0."""
    x, _ = _clustered()
    pages = [x[:80], x[80:160], x[160:]]
    codebook = fit_codebook(x, n_clusters=6, seed=0)
    alpha = alpha_for_codebook(codebook, pages)

    model = NetVLAD.from_codebook(codebook, alpha)
    got = model(torch.from_numpy(pages[0])).detach().numpy()
    want = vlad_encode(pages[0], codebook, intra_norm=False)

    assert got.shape == want.shape == (6 * 8,)
    cos = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want)))
    assert cos > 0.999, f"soft/hard VLAD disagree at init (cos={cos})"


def test_init_matches_with_intra_norm_too():
    """The mole variant of VLAD is reproduced as well, not just Raven-plain."""
    x, _ = _clustered(seed=3)
    pages = [x[:120], x[120:]]
    codebook = fit_codebook(x, n_clusters=6, seed=0)
    alpha = alpha_for_codebook(codebook, pages, intra_norm=True)
    assert vlad_fidelity(pages, codebook, alpha, intra_norm=True) > 0.999


def test_entropy_is_not_a_sufficient_fidelity_proxy():
    """REGRESSION: a near-one-hot softmax can still give the wrong descriptor.

    Calibrating α on assignment mass (entropy ≈ 0 ⇒ "basically hard assignment")
    is the intuitive move and it is wrong: the residual to a distant centre is
    large and coherent, so a fraction of a percent of leaked assignment outweighs
    the correct cluster, whose residuals largely cancel. This test pins the trap
    so nobody re-derives α from entropy later.
    """
    from mole.supervised.netvlad import assignment_gap

    x, _ = _clustered()
    pages = [x]
    codebook = fit_codebook(x, n_clusters=6, seed=0)
    # The textbook rule: pick α so the runner-up centre carries ~1% of the
    # softmax mass. That looks decisively one-hot by entropy...
    weak = -np.log(0.01) / assignment_gap(codebook, x)
    model = NetVLAD.from_codebook(codebook, weak)
    assert model.assignment_entropy(torch.from_numpy(x)) < 0.15 * np.log(6)
    # ...yet the descriptor is nowhere near hard VLAD.
    assert vlad_fidelity(pages, codebook, weak) < 0.5
    # The calibrated α fixes it, and it is strictly sharper.
    good = alpha_for_codebook(codebook, pages)
    assert good > weak
    assert vlad_fidelity(pages, codebook, good) > 0.999


def test_alpha_controls_sharpness():
    """Small α blurs the aggregator toward a mean — the statistic known to fail."""
    x, _ = _clustered()
    codebook = fit_codebook(x, n_clusters=6, seed=0)
    sharp = NetVLAD.from_codebook(codebook, alpha_for_codebook(codebook, [x]))
    blunt = NetVLAD.from_codebook(codebook, 1e-4)
    xt = torch.from_numpy(x)
    assert sharp.assignment_entropy(xt) < 0.05
    assert blunt.assignment_entropy(xt) > 0.9 * np.log(6)


def test_batched_matches_single_page():
    x, _ = _clustered()
    codebook = fit_codebook(x, n_clusters=6, seed=0)
    model = NetVLAD.from_codebook(codebook, 2.0)
    batch = torch.from_numpy(np.stack([x[:40], x[40:80]]))
    got = model(batch).detach().numpy()
    for j, block in enumerate((x[:40], x[40:80])):
        one = model(torch.from_numpy(block)).detach().numpy()
        np.testing.assert_allclose(got[j], one, atol=1e-5)


def test_token_count_invariance():
    """Power-norm + L2 make the descriptor invariant to how many tokens went in.

    This is why training on a 512-token subsample and deploying on the whole page
    is not a train/deploy mismatch.
    """
    x, _ = _clustered(per=200, seed=7)
    codebook = fit_codebook(x, n_clusters=6, seed=0)
    model = NetVLAD.from_codebook(codebook, 2.0)
    full = model(torch.from_numpy(x)).detach().numpy()
    doubled = model(torch.from_numpy(np.vstack([x, x]))).detach().numpy()
    np.testing.assert_allclose(full, doubled, atol=2e-3)


def test_learn_flag_freezes_the_right_parameters():
    codebook = fit_codebook(_clustered()[0], n_clusters=6, seed=0)
    assign = NetVLAD.from_codebook(codebook, 2.0, learn="assign")
    assert not assign.centroids.requires_grad
    assert assign.assign_c.requires_grad
    centres = NetVLAD.from_codebook(codebook, 2.0, learn="centroids")
    assert centres.centroids.requires_grad
    assert not centres.assign_c.requires_grad
    with pytest.raises(ValueError):
        NetVLAD.from_codebook(codebook, 2.0, learn="nonsense")


def test_parameter_groups_share_a_scale():
    """Both groups sit at codebook scale, so ONE learning rate serves both.

    The textbook w = 2*alpha*C init puts the assignment ~2*alpha above the
    centroids; with Adam stepping ~lr regardless of gradient size, that leaves
    the assignment effectively frozen. Measured on the first real fold: loss
    1.229 -> 1.225 over 20 epochs, i.e. no training at all.
    """
    x, _ = _clustered()
    codebook = fit_codebook(x, n_clusters=6, seed=0)
    model = NetVLAD.from_codebook(codebook, 50.0)
    cen, asg = model.centroids.norm().item(), model.assign_c.norm().item()
    assert 0.5 < asg / cen < 2.0, f"scale mismatch: assign {asg:.3g} vs centroids {cen:.3g}"


def test_gradients_reach_both_parameter_groups():
    x, _ = _clustered()
    codebook = fit_codebook(x, n_clusters=6, seed=0)
    model = NetVLAD.from_codebook(codebook, 2.0)
    model(torch.from_numpy(x)).sum().backward()
    assert model.centroids.grad is not None and model.centroids.grad.abs().sum() > 0
    assert model.assign_c.grad is not None and model.assign_c.grad.abs().sum() > 0


# ---------------------------------------------------------------- token cache
def _fake_cache(tmp_path, n_hands=8, docs_per_hand=3, tokens=60, dim=8, seed=0):
    """A cache whose hands ARE the blob structure — signal a learner can find."""
    rng = np.random.default_rng(seed)
    hand_centres = rng.normal(size=(n_hands, dim)).astype(np.float32) * 4.0
    blocks, pages, cursor = [], [], 0
    for h in range(n_hands):
        archive = f"arch{h % 2}"
        for d in range(docs_per_hand):
            block = (hand_centres[h] + rng.normal(scale=0.5, size=(tokens, dim))
                     ).astype(np.float32)
            blocks.append(block)
            pages.append({"item": f"/data/{archive}/h{h}_d{d}.png", "archive": archive,
                          "hand": f"{archive}/h{h}", "doc": f"{archive}/d{h}_{d}",
                          "start": cursor, "count": tokens})
            cursor += tokens
    cache = TokenCache(np.vstack(blocks).astype(np.float16), pages,
                       meta={"model_id": "test@0", "embed_dim": dim})
    cache.save(tmp_path / "cache")
    return cache


def test_token_cache_roundtrip(tmp_path):
    cache = _fake_cache(tmp_path)
    back = TokenCache.load(tmp_path / "cache")
    assert back.n_pages == cache.n_pages
    assert back.dim == cache.dim
    np.testing.assert_allclose(back.page_tokens(3), cache.page_tokens(3))
    assert back.meta["model_id"] == "test@0"
    assert set(back.archives) == {"arch0", "arch1"}


def test_rows_for_filters(tmp_path):
    cache = _fake_cache(tmp_path)
    a0 = cache.rows_for(archive="arch0")
    assert a0 and all(cache.pages[i]["archive"] == "arch0" for i in a0)
    one = cache.rows_for(hands={"arch0/h0"})
    assert len(one) == 3


def test_sample_tokens_shape_and_padding(tmp_path):
    cache = _fake_cache(tmp_path, tokens=5)
    rng = np.random.default_rng(0)
    assert cache.sample_tokens(0, 3, rng).shape == (3, 8)
    assert cache.sample_tokens(0, 12, rng).shape == (12, 8)   # replacement


def test_descriptor_pool_is_bounded(tmp_path):
    cache = _fake_cache(tmp_path)
    pool = descriptor_pool(cache, max_descriptors=100, seed=0)
    assert 0 < len(pool) <= 120                      # per-page rounding slack
    assert pool.dtype == np.float32


def test_vlad_page_vectors_matches_vlad_encode(tmp_path):
    cache = _fake_cache(tmp_path)
    codebook = fit_codebook(descriptor_pool(cache), n_clusters=6, seed=0)
    got = vlad_page_vectors(cache, codebook, [0, 1], progress=False)
    for j, i in enumerate((0, 1)):
        np.testing.assert_allclose(
            got[j], vlad_encode(cache.page_tokens(i), codebook, intra_norm=False),
            atol=1e-6)


def test_write_embeddings_is_eval_compatible(tmp_path):
    import json

    cache = _fake_cache(tmp_path)
    rows = cache.rows_for(archive="arch0")
    mat = np.zeros((len(rows), 4), np.float32)
    out = write_embeddings(tmp_path / "e.npy", mat, cache, rows, {"pooling": "vlad"},
                           dataset_dir="/elsewhere/arch0")
    meta = json.loads(out.with_suffix(".mapping.json").read_text())
    assert len(meta["rows"]) == len(rows) == len(np.load(out))
    assert meta["rows"][0]["image"].startswith("/elsewhere/arch0/")
    assert meta["rows"][0]["image"].endswith(".png")
    assert meta["model_id"] == "test@0"


# -------------------------------------------------------------------- training
def test_training_lifts_unseen_hands(tmp_path):
    """The end-to-end claim: SupCon through the aggregator improves HELD-OUT hands.

    Held-out hands are unseen classes, so a lift here is geometry, not
    memorization — the same standard `train_head` is held to.
    """
    from mole.supervised.netvlad import _holdout_macro_map

    cache = _fake_cache(tmp_path, n_hands=12, docs_per_hand=4, tokens=80, seed=1)
    codebook = fit_codebook(descriptor_pool(cache), n_clusters=6, seed=0)
    holdout = {"arch0/h0", "arch1/h1", "arch0/h2", "arch1/h3"}

    model, report = train_netvlad(
        cache, codebook, holdout_hands=holdout, alpha=1.0, epochs=6,
        tokens_per_page=32, lr=5e-2, seed=0, progress=False,
        sampler_cfg={"hands_per_batch": 6, "docs_per_hand": 2,
                     "same_archive_frac": 0.5, "batches_per_epoch": 10})

    base = NetVLAD.from_codebook(codebook, 1.0)
    rows = cache.rows_for(hands=holdout)
    before = _holdout_macro_map(base, cache, rows, torch.device("cpu"))
    after = _holdout_macro_map(model, cache, rows, torch.device("cpu"))
    assert after >= before, f"held-out macro-mAP fell: {before:.4f} → {after:.4f}"
    assert report["best_epoch"] >= 0
    assert report["n_excluded_hands"] == 0
    assert len(report["history"]) == 6
    assert all("assign_entropy" in h for h in report["history"])
    assert all("grad_assign" in h and "grad_centroids" in h for h in report["history"])
    assert "init_fidelity" in report


def test_excluded_archive_never_trains_or_selects(tmp_path):
    """LOAO integrity: the held-out archive touches neither training nor selection."""
    cache = _fake_cache(tmp_path, n_hands=12, docs_per_hand=3, seed=2)
    codebook = fit_codebook(descriptor_pool(cache), n_clusters=6, seed=0)
    excluded = {p["hand"] for p in cache.pages if p["archive"] == "arch1" and p["hand"]}
    holdout = {"arch0/h0", "arch0/h2"}

    _, report = train_netvlad(
        cache, codebook, holdout_hands=holdout, exclude_hands=excluded,
        alpha=1.0, epochs=2, tokens_per_page=16, seed=0, progress=False,
        sampler_cfg={"hands_per_batch": 4, "docs_per_hand": 2,
                     "same_archive_frac": 0.0, "batches_per_epoch": 4})

    assert set(report["excluded_hands"]) == excluded
    assert not (set(report["train_hands"]) & excluded)
    assert not (set(report["holdout_hands"]) & excluded)
    assert all(h.startswith("arch0/") for h in report["train_hands"])


def test_netvlad_page_vectors_are_l2_normalised(tmp_path):
    cache = _fake_cache(tmp_path)
    codebook = fit_codebook(descriptor_pool(cache), n_clusters=6, seed=0)
    model = NetVLAD.from_codebook(codebook, 2.0)
    emb = netvlad_page_vectors(model, cache, [0, 1, 2], progress=False)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_descriptor_pool_uncapped_is_exact(tmp_path):
    """max_descriptors=0 returns EVERY token, in order — no sampling, no loss.

    This is the setting the cache check runs at, because reproducing
    `outputs/pooled_final` means fitting on all descriptors the way
    `mole embed` does. Also guards the in-place fill: an uncapped pool on a real
    archive is ~8 GB, so accumulate-then-vstack would double peak RAM.
    """
    cache = _fake_cache(tmp_path, n_hands=4, docs_per_hand=2, tokens=25)
    pool = descriptor_pool(cache, max_descriptors=0)
    assert len(pool) == cache.n_tokens == 4 * 2 * 25
    np.testing.assert_allclose(pool, np.asarray(cache.tokens, dtype=np.float32))
