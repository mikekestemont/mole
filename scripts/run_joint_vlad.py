#!/usr/bin/env python
"""Fully-differentiable VLAD: joint backbone + NetVLAD finetune, one LOAO fold.

The literature-faithful NetVLAD (gradients reach the backbone), warm-started so the
UNTRAINED model == the current deployed descriptor and every epoch is measured
against it. One archive is held out: its hands are excluded from training AND used
for model-selection, so the printed delta is a clean leave-one-archive-out number.

  python scripts/run_joint_vlad.py runs/pooled_bin_ft/checkpoint.pth data/pooled-bin \
      outputs/universal_full/fit.codebook.npy --holdout-archive antwerp-bin \
      --out runs/jointvlad/antwerp --device 5

Run one fold per invocation (a joint finetune is heavy); repeat per archive for the
full LOAO. Prints frozen-vs-trained held-out macro-mAP — the verdict for that fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_trainable_backbone(ckpt, device):
    """The embed backbone, but grads ON and train mode — ready to finetune."""
    from mole.embed.extract import load_backbone
    model, meta = load_backbone(ckpt, map_location=device)
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    return model, meta


def _make_load_crops(meta, window_size, overlap, cache_windows):
    """Cached page → transformed window crops.

    Each page is loaded and transformed ONCE and memoised (the same pages recur every
    epoch, and disk I/O on big scans is the training bottleneck on a CPU-bound box).
    ``cache_windows`` caps the windows kept per page (deterministic even spacing), which
    bounds RAM (~cache_windows × 0.6 MB × #pages) and fixes each page's windows across
    epochs — cheap, and fine as sampling for a finetune.
    """
    from mole.data.patches import load_rgb, window_coords
    from mole.embed.extract import _build_transform
    transform = _build_transform(meta["model_size"])
    invert = bool(meta.get("invert", False))
    cache: dict[str, list] = {}

    def load_crops(item):
        key = str(item.path)
        hit = cache.get(key)
        if hit is not None:
            return hit
        page = load_rgb(item.path, invert=invert)
        w, h = page.size
        wins = window_coords(w, h, window_size, overlap, None)
        if cache_windows and len(wins) > cache_windows:          # even subsample, deterministic
            idx = np.linspace(0, len(wins) - 1, cache_windows).round().astype(int)
            wins = [wins[i] for i in idx]
        crops = [transform(page.crop((win.x, win.y, win.x + win.size, win.y + win.size)))
                 for win in wins]
        cache[key] = crops
        return crops
    return load_crops


def _make_load_view(meta, window_size, overlap, view_windows, preset):
    """A page → ONE augmented view (random window subset, each window augmented).

    The PIL window crops are cached (superset of ``view_windows``) so disk+crop happen
    once; each view then draws a random subset and augments it, so two calls give two
    different views (different windows AND different augmentation). Uses the SAME mild
    augmentation the backbone was pretrained with — gentle enough to preserve writer
    strokes (the make-or-break requirement for this SSL objective).
    """
    from mole.data.augment import MoleMultiCropAugmentation, resolve_config
    from mole.data.patches import load_rgb, window_coords
    aug = MoleMultiCropAugmentation(resolve_config(preset, model_size=meta["model_size"]))
    invert = bool(meta.get("invert", False))
    cap = max(view_windows * 2, view_windows)                # cache a superset per page
    crops_pil: dict[str, list] = {}

    def load_view(path, rng):
        pil = crops_pil.get(str(path))
        if pil is None:
            page = load_rgb(Path(path), invert=invert)
            w, h = page.size
            wins = window_coords(w, h, window_size, overlap, None)
            if cap and len(wins) > cap:
                idx = np.linspace(0, len(wins) - 1, cap).round().astype(int)
                wins = [wins[i] for i in idx]
            pil = [page.crop((win.x, win.y, win.x + win.size, win.y + win.size)) for win in wins]
            crops_pil[str(path)] = pil
        k = min(view_windows, len(pil))
        pick = rng.choice(len(pil), k, replace=False) if len(pil) > k else list(range(len(pil)))
        return [aug.global_transfo1(pil[i]) for i in pick]   # augment each -> tensor
    return load_view


def _sample_page_tokens(model, load_crops, items, n, fwd, device, seed=0):
    """Foreground tokens for n sample pages (no grad) — to calibrate NetVLAD's alpha."""
    import torch

    from mole.embed.extract import _foreground_mask
    from mole.embed.pooling import patch_descriptors
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(items), min(n, len(items)), replace=False)
    out = []
    model.eval()
    with torch.no_grad():
        for i in pick:
            crops = load_crops(items[int(i)])
            if not crops:
                continue
            tok = model(torch.stack(crops).to(device), return_attention=False,
                        return_all_tokens=True)
            patches = patch_descriptors(tok, fwd["num_class_tokens"])
            keep = _foreground_mask(crops, fwd["patch_size"], fwd["fg_threshold"],
                                    method=fwd["fg_method"])
            fg = patches[keep.to(patches.device)].cpu().numpy().astype(np.float32)
            if len(fg):
                out.append(fg)
    model.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, help="Warm-start backbone (pooled SSL).")
    ap.add_argument("root", type=Path, help="Pooled labels root (all archives).")
    ap.add_argument("codebook", type=Path, help="Universal VLAD codebook (fit.codebook.npy).")
    ap.add_argument("--holdout-archive", required=True, help="Archive to leave out (its hands).")
    ap.add_argument("--out", type=Path, required=True, help="Run dir for weights + report.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-tokens", type=int, default=768, help="Fg tokens/page into NetVLAD.")
    ap.add_argument("--max-windows", type=int, default=24, help="Windows/page through the ViT (memory).")
    ap.add_argument("--window-size", type=int, default=224)
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--fg-method", default="contrast")
    ap.add_argument("--fg-threshold", type=float, default=0.05)
    ap.add_argument("--hands-per-batch", type=int, default=8)
    ap.add_argument("--docs-per-hand", type=int, default=2)
    ap.add_argument("--batches-per-epoch", type=int, default=100)
    ap.add_argument("--learn", default="both", help="NetVLAD trainable: both|assign|centroids.")
    ap.add_argument("--self-supervised", action="store_true",
                    help="LABEL-FREE: two augmented views of a page are the positive pair "
                         "(no writer labels; labels used only for held-out selection). Trains "
                         "on ALL training-archive pages.")
    ap.add_argument("--aug-preset", default="mild", help="SSL augmentation strength (mild|default).")
    ap.add_argument("--batch-pages", type=int, default=8, help="SSL: pages per batch (×2 views).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from mole.supervised.datasets import load_labeled_pairs
    from mole.supervised.jointvlad import (holdout_doc_macro_map, train_joint_vlad,
                                           train_joint_vlad_ssl)
    from mole.supervised.netvlad import NetVLAD, alpha_for_codebook

    dev = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    model, meta = _load_trainable_backbone(args.checkpoint, dev)
    fwd = dict(num_class_tokens=meta["num_class_tokens"], patch_size=meta["patch_size"],
               fg_threshold=args.fg_threshold, fg_method=args.fg_method,
               embed_dim=meta["embed_dim"], max_tokens=args.max_tokens,
               max_windows=0)                             # the crop cache already caps windows
    load_crops = _make_load_crops(meta, args.window_size, args.overlap, args.max_windows)

    index = load_labeled_pairs(args.root)
    holdout = {h for h in index.hands if h.startswith(f"{args.holdout_archive}/")}
    if not holdout:
        raise SystemExit(f"no hands for held-out archive {args.holdout_archive!r} "
                         f"(archives: {sorted(index.archives)})")
    print(f"[joint] {len(index.items)} labeled pages, holding out {args.holdout_archive} "
          f"({len(holdout)} hands)")

    codebook = np.load(args.codebook).astype(np.float32)
    if codebook.shape[1] != meta["embed_dim"]:
        raise SystemExit(f"codebook is {codebook.shape[1]}-d, backbone is {meta['embed_dim']}-d")
    tok_sample = _sample_page_tokens(model, load_crops, index.items, 12, fwd, dev)
    alpha = alpha_for_codebook(codebook, tok_sample)
    print(f"[joint] NetVLAD alpha calibrated to {alpha:.1f} (fidelity target)")
    netvlad = NetVLAD.from_codebook(codebook, alpha, learn=args.learn).to(dev)

    # baseline: the untrained (== deployed frozen-VLAD) model on the held-out archive
    frozen = holdout_doc_macro_map(model, netvlad, index, holdout, load_crops, fwd, dev)
    print(f"[joint] frozen held-out macro-mAP: {frozen:.4f}")

    if args.self_supervised:
        from mole.data.datasets import IMAGE_EXTENSIONS, discover_datasets
        ssl_paths = []
        for m in discover_datasets(args.root):
            if m.name == args.holdout_archive:
                continue                                   # LOAO: held-out archive unseen
            ssl_paths += [p for p in sorted(m.root.iterdir())
                          if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"[joint] SELF-SUPERVISED: {len(ssl_paths)} training pages (label-free, "
              f"{args.aug_preset} aug, 2 views/page)")
        load_view = _make_load_view(meta, args.window_size, args.overlap, args.max_windows,
                                    args.aug_preset)
        model, netvlad, report = train_joint_vlad_ssl(
            model, netvlad, ssl_paths, load_view, holdout_index=index, holdout_hands=holdout,
            holdout_load_crops=load_crops, fwd=fwd, epochs=args.epochs, lr=args.lr,
            batch_pages=args.batch_pages, batches_per_epoch=args.batches_per_epoch,
            device=str(dev), seed=args.seed)
    else:
        model, netvlad, report = train_joint_vlad(
            model, netvlad, index, load_crops=load_crops, fwd=fwd, holdout_hands=holdout,
            epochs=args.epochs, lr=args.lr, device=str(dev), seed=args.seed,
            sampler_cfg=dict(hands_per_batch=args.hands_per_batch,
                             docs_per_hand=args.docs_per_hand,
                             batches_per_epoch=args.batches_per_epoch))

    report.update({"holdout_archive": args.holdout_archive, "frozen_macro": frozen,
                   "delta": report["best_holdout_macro"] - frozen, "alpha": alpha,
                   "self_supervised": bool(args.self_supervised)})
    torch.save({"backbone": model.state_dict(), "netvlad": netvlad.state_dict(),
                "meta": meta, "report": report}, args.out / "joint.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[joint] === {args.holdout_archive} ===")
    print(f"  frozen  {frozen:.4f}")
    print(f"  trained {report['best_holdout_macro']:.4f}  (best epoch {report['best_epoch']})")
    print(f"  delta   {report['delta']:+.4f}")
    print(f"[joint] ✓ {args.out}/joint.pt + report.json")


if __name__ == "__main__":
    main()
