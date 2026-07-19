"""Tests for `mole embed --head` (Tier-1 head applied at embed time)."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image

from mole.embed.extract import embed, load_backbone


def _dataset(tmp_path, n=4):
    ds = tmp_path / "arch1"
    ds.mkdir()
    rng = np.random.default_rng(0)
    names = [f"img_{i}.png" for i in range(n)]
    for name in names:
        Image.fromarray((rng.random((260, 260, 3)) * 255).astype("uint8")).save(ds / name)
    (ds / "labels.csv").write_text(
        "filename,hand_id\n" + "".join(f"{n},H{i % 2}\n" for i, n in enumerate(names)))
    return ds


def _tiny_ckpt(tmp_path):
    from mole.selfsup.vit import build_vit
    ck = tmp_path / "ck.pth"
    torch.save({"state_dict": build_vit("vit_tiny", patch_size=16,
                                        num_class_tokens=1).state_dict()}, ck)
    return ck


def _save_head(path, in_dim, out_dim, base_model_id, kind="linear"):
    from mole.supervised.metric import build_head
    head = build_head(kind, in_dim, out_dim)
    torch.save({"state_dict": head.state_dict(), "in_dim": in_dim, "out_dim": out_dim,
                "kind": kind, "base_model_id": base_model_id}, path)


def test_embed_with_head_projects_to_out_dim(tmp_path):
    ck = _tiny_ckpt(tmp_path)
    ds = _dataset(tmp_path)
    model_id = load_backbone(ck)[1]["model_id"]
    head = tmp_path / "head.pt"
    _save_head(head, 192, 16, model_id)

    out = tmp_path / "emb.npy"
    embed(ck, ds, out, pooling="mean", head=head, invert=False, batch_size=8)
    X = np.load(out)
    assert X.shape[1] == 16                       # pooled in the projected space
    sidecar = json.loads((tmp_path / "emb.mapping.json").read_text())
    assert sidecar["head_out_dim"] == 16
    assert sidecar["head_id"].endswith(model_id)  # head_id = sha8 + base_model_id


def test_embed_head_vlad_codebook_is_in_projected_space(tmp_path):
    ck = _tiny_ckpt(tmp_path)
    ds = _dataset(tmp_path)
    model_id = load_backbone(ck)[1]["model_id"]
    head = tmp_path / "head.pt"
    _save_head(head, 192, 8, model_id)

    out = tmp_path / "v.npy"
    embed(ck, ds, out, pooling="vlad", head=head, vlad_clusters=3, invert=False, batch_size=8)
    codebook = np.load(tmp_path / "v.codebook.npy")
    assert codebook.shape[1] == 8                 # codebook fit on projected descriptors
    assert np.load(out).shape[1] == 3 * 8         # VLAD dim = K * out_dim


def test_embed_head_base_model_mismatch_is_a_hard_error(tmp_path):
    ck = _tiny_ckpt(tmp_path)
    ds = _dataset(tmp_path)
    head = tmp_path / "head.pt"
    _save_head(head, 192, 8, "some_other_model@deadbeef+step0")
    with pytest.raises(ValueError, match="only valid on the backbone"):
        embed(ck, ds, tmp_path / "e.npy", pooling="mean", head=head, invert=False)


def test_embed_head_rejects_cls_pooling(tmp_path):
    ck = _tiny_ckpt(tmp_path)
    ds = _dataset(tmp_path)
    model_id = load_backbone(ck)[1]["model_id"]
    head = tmp_path / "head.pt"
    _save_head(head, 192, 8, model_id)
    with pytest.raises(ValueError, match="projects patch tokens"):
        embed(ck, ds, tmp_path / "e.npy", pooling="cls", head=head, invert=False)
