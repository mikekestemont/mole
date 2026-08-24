#!/usr/bin/env python
"""Full-window standard eval of a joint-NetVLAD finetune — confirm the LOAO delta.

`run_joint_vlad.py`'s in-loop model-selection uses a page-level proxy capped at
`--max-windows` (8) for memory, which depresses the absolute on big-page archives
(Brackley 45 MP scans read ~0.26 there vs ~0.76 elsewhere). This re-embeds a held-out
archive at FULL windows/tokens (no grad, so no memory cap) with BOTH the trained joint
model and the untrained (== deployed frozen-VLAD) init, writes standard embeddings, and
you eval them with the ordinary `mole eval` — removing the proxy artifact.

  python scripts/run_joint_eval.py runs/jointvlad/antwerp/joint.pt \
      runs/pooled_bin_ft/checkpoint.pth outputs/universal_full/fit.codebook.npy \
      data/antwerp-bin --out outputs/joint_eval/antwerp --device cuda:5
  # then, the familiar eval on each arm:
  mole eval outputs/joint_eval/antwerp.frozen.npy  data/antwerp-bin --cross-doc-only --per-hand
  mole eval outputs/joint_eval/antwerp.trained.npy data/antwerp-bin --cross-doc-only --per-hand
  mole eval-compare outputs/joint_eval/antwerp.frozen.eval.json \
      outputs/joint_eval/antwerp.trained.eval.json    # (writing eval jsons via --out)

The frozen arm doubles as an alpha check: if frozen-NetVLAD ≈ the known hard-VLAD
number, the calibrated alpha is faithful and not the thing holding results back.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _eval_docvec(model, netvlad, crops, fwd, device, batch_size=32):
    """A page's full-window NetVLAD vector, no grad, ViT run in chunks (giant pages fit)."""
    import torch

    from mole.embed.extract import _foreground_mask
    from mole.embed.pooling import patch_descriptors

    keep = _foreground_mask(crops, fwd["patch_size"], fwd["fg_threshold"], method=fwd["fg_method"])
    fg_chunks = []
    for i in range(0, len(crops), batch_size):
        batch = torch.stack(crops[i:i + batch_size]).to(device)
        tok = model(batch, return_attention=False, return_all_tokens=True)
        patches = patch_descriptors(tok, fwd["num_class_tokens"])
        fg_chunks.append(patches[keep[i:i + batch_size].to(patches.device)])
    fg = torch.cat(fg_chunks) if fg_chunks else torch.zeros(0, fwd["embed_dim"], device=device)
    if fg.shape[0] == 0:
        fg = torch.zeros(1, fwd["embed_dim"], device=device)
    return netvlad(fg)


def _embed_archive(model, netvlad, archive, meta, fwd, window_size, overlap, device, out_npy):
    import torch

    from mole.data.patches import load_rgb
    from mole.embed.extract import _build_transform, _page_index, _write_output
    from mole.embed.pooling import Pooling
    from mole.progress import track

    pages = _page_index(Path(archive), window_size, overlap, use_zones=False)
    transform = _build_transform(meta["model_size"])
    invert = bool(meta.get("invert", False))
    k, dim = netvlad.centroids.shape
    model.eval(); netvlad.eval()
    vecs, rows = [], []
    with torch.no_grad():
        for i, entry in enumerate(track(pages, f"Embedding → {out_npy.name}", unit="page")):
            page = load_rgb(entry.path, invert=invert)
            crops = [transform(page.crop((w.x, w.y, w.x + w.size, w.y + w.size)))
                     for w in entry.windows]
            if crops:
                v = _eval_docvec(model, netvlad, crops, fwd, device).cpu().numpy()
            else:
                v = np.zeros(k * dim, np.float32)
            vecs.append(v.astype(np.float32))
            rows.append({"row": i, "image": str(entry.path)})
    matrix = np.vstack(vecs)
    _write_output(out_npy, matrix, rows, meta, Pooling.VLAD, False, netvlad.codebook(),
                  k, 0, foreground=True, foreground_threshold=fwd["fg_threshold"],
                  foreground_method=fwd["fg_method"], vlad_intra_norm=False, codebook_source="joint")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("joint", type=Path, help="Trained joint.pt from run_joint_vlad.py.")
    ap.add_argument("base_checkpoint", type=Path, help="Pooled backbone (for the frozen arm).")
    ap.add_argument("codebook", type=Path, help="Universal codebook (frozen NetVLAD init).")
    ap.add_argument("archive", type=Path, help="Held-out archive dir (data/<archive>).")
    ap.add_argument("--out", type=Path, required=True, help="Output prefix (dir/name).")
    ap.add_argument("--window-size", type=int, default=224)
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--fg-method", default="contrast")
    ap.add_argument("--fg-threshold", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    from mole.embed.extract import load_backbone
    from mole.supervised.netvlad import NetVLAD

    dev = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    blob = torch.load(args.joint, map_location="cpu", weights_only=False)
    meta = blob["meta"]
    alpha = float(blob["report"].get("alpha", 1.0))
    fwd = dict(num_class_tokens=meta["num_class_tokens"], patch_size=meta["patch_size"],
               fg_threshold=args.fg_threshold, fg_method=args.fg_method, embed_dim=meta["embed_dim"])
    codebook = np.load(args.codebook).astype(np.float32)

    # frozen arm: pooled backbone + untrained from_codebook NetVLAD (== deployed frozen VLAD)
    fmodel, _ = load_backbone(args.base_checkpoint, map_location=str(dev))
    fmodel.to(dev)
    fnet = NetVLAD.from_codebook(codebook, alpha).to(dev)
    _embed_archive(fmodel, fnet, args.archive, meta, fwd, args.window_size, args.overlap, dev,
                   Path(f"{args.out}.frozen.npy"))

    # trained arm: the finetuned backbone + trained NetVLAD
    tmodel, _ = load_backbone(args.base_checkpoint, map_location=str(dev))
    tmodel.load_state_dict(blob["backbone"]); tmodel.to(dev)
    tnet = NetVLAD.from_codebook(codebook, alpha)
    tnet.load_state_dict(blob["netvlad"]); tnet.to(dev)
    _embed_archive(tmodel, tnet, args.archive, meta, fwd, args.window_size, args.overlap, dev,
                   Path(f"{args.out}.trained.npy"))

    print(f"\n[joint-eval] ✓ {args.out}.frozen.npy + {args.out}.trained.npy")
    print("  Now eval both (full-window, standard metric):")
    a = args.archive
    print(f"    mole eval {args.out}.frozen.npy  {a} --cross-doc-only --per-hand "
          f"--out {args.out}.frozen.eval.json")
    print(f"    mole eval {args.out}.trained.npy {a} --cross-doc-only --per-hand "
          f"--out {args.out}.trained.eval.json")
    print(f"    mole eval-compare {args.out}.frozen.eval.json {args.out}.trained.eval.json")


if __name__ == "__main__":
    main()
