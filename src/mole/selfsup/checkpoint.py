"""Seamless-resume checkpointing.

A checkpoint carries everything needed to continue *exactly* where training
stopped: model (student), EMA teacher, optimizer, AMP scaler, iBOT loss centers,
the global step, and ALL RNG states (python, numpy, torch CPU + CUDA). Schedules
are pure functions of the global step, so they are recomputed on resume rather
than stored.

The run directory holds a single rolling ``checkpoint.pth`` (latest) plus optional
numbered ``checkpoint_epochNNNN.pth`` snapshots.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_NAME = "checkpoint.pth"


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["torch_cuda"]])
        except Exception:
            pass  # GPU count changed between save and resume — skip cuda RNG


def _as_byte_tensor(t):
    # RNG state must be a CPU uint8 tensor; torch.load(map_location=gpu) may have
    # moved it to the training device, so force it back to CPU/uint8.
    if isinstance(t, torch.Tensor):
        return t.cpu().to(torch.uint8)
    return torch.tensor(t, dtype=torch.uint8)


def save_checkpoint(run_dir: str | Path, *, student, teacher, optimizer, ibot_loss,
                    fp16_scaler, global_step: int, config: dict, epoch_snapshot: int | None = None,
                    extra: dict | None = None) -> Path:
    """Write the rolling checkpoint (and an optional numbered epoch snapshot)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ibot_loss": ibot_loss.state_dict(),
        "fp16_scaler": fp16_scaler.state_dict() if fp16_scaler is not None else None,
        "global_step": global_step,
        "config": config,
        "rng": _rng_state(),
        **(extra or {}),
    }
    path = run_dir / CHECKPOINT_NAME
    torch.save(payload, path)
    if epoch_snapshot is not None:
        torch.save(payload, run_dir / f"checkpoint_epoch{epoch_snapshot:04d}.pth")
    return path


def _relaxed_buffer_load(module, sd: dict) -> None:
    """Assign saved buffers directly (their shapes may have drifted during
    training — e.g. the iBOT loss centers gain a class-token dim after step 1)."""
    buffers = dict(module.named_buffers())
    for name, val in sd.items():
        if name not in buffers:
            continue
        parent = module
        *parents, leaf = name.split(".")
        for p in parents:
            parent = getattr(parent, p)
        setattr(parent, leaf, val.clone().to(buffers[name].device))


def load_checkpoint(path: str | Path, *, student, teacher, optimizer, ibot_loss,
                    fp16_scaler, map_location="cpu") -> int:
    """Restore state in place and return the saved ``global_step``."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    student.load_state_dict(ckpt["student"])
    teacher.load_state_dict(ckpt["teacher"])
    optimizer.load_state_dict(ckpt["optimizer"])
    _relaxed_buffer_load(ibot_loss, ckpt["ibot_loss"])  # centers only; shapes may drift
    if fp16_scaler is not None and ckpt.get("fp16_scaler") is not None:
        fp16_scaler.load_state_dict(ckpt["fp16_scaler"])
    if "rng" in ckpt:
        _restore_rng(ckpt["rng"])
    return int(ckpt.get("global_step", 0))


def find_resume(run_dir: str | Path) -> Path | None:
    """Return the rolling checkpoint path if the run dir has one (auto-resume)."""
    p = Path(run_dir) / CHECKPOINT_NAME
    return p if p.is_file() else None


# --------------------------------------------------------------------------- interop
# Warm-starting from a foreign checkpoint. Three formats are recognised and reduced
# to a canonical BARE ViT backbone state dict (the transferable part; projection
# heads are run-specific and re-initialised):
#   1. mole            -- `config` dict + `student`/`teacher` MultiCropWrapper sds
#   2. AttMask/iBOT run -- `args` Namespace + DDP-`module.`-prefixed `student` + `teacher`
#   3. extracted backbone -- `{"state_dict": {...}}` (or a raw sd) of bare ViT weights,
#                            e.g. the original `extract_backbone_weights.py` output
# Architecture comes from `config`/`args` when present, else it is INFERRED from the
# backbone weights (embed_dim, depth, patch, num_class_tokens) — a bare checkpoint
# carries no metadata but the weights fully determine the ViT shape.

# model.* fields that determine parameter shapes — recovered from a foreign
# checkpoint's args and adopted by the warm-started run so weights load.
ARCH_FIELDS = ("arch", "patch_size", "num_class_tokens", "out_dim", "patch_out_dim",
               "shared_head", "shared_head_teacher", "norm_last_layer", "norm_in_head",
               "act_in_head", "use_masked_im_modeling")

# (embed_dim, depth) -> canonical ViT name (heads/mlp are fixed by the factory).
_ARCH_BY_SHAPE = {(192, 12): "vit_tiny", (384, 12): "vit_small",
                  (768, 12): "vit_base", (1024, 24): "vit_large"}


def is_mole_checkpoint(ckpt: dict) -> bool:
    """A mole checkpoint carries a `config` dict; a foreign one does not."""
    return isinstance(ckpt, dict) and "config" in ckpt


def _bare_backbone(sd: dict) -> dict:
    """Reduce any wrapper/DDP state dict to bare ViT backbone keys.

    Strips the DDP ``module.`` prefix and the MultiCropWrapper ``backbone.`` prefix,
    and drops the projection head (``head.*``) / ``fc.*``. Already-bare weights
    (an extracted backbone) pass through unchanged.
    """
    out = {}
    for k, v in sd.items():
        kk = k[len("module."):] if k.startswith("module.") else k
        if kk.startswith("backbone."):
            kk = kk[len("backbone."):]
        elif kk.startswith(("head.", "fc.")):
            continue  # projection head / classifier — run-specific, not transferred
        out[kk] = v
    return out


def _base_config() -> dict:
    import copy

    from mole.config import DEFAULTS
    return copy.deepcopy(DEFAULTS)


def _foreign_to_config(args) -> dict:
    """Synthesize a mole config from a foreign checkpoint's argparse args.

    Only the architecture (model.*) is taken from the checkpoint — it must match
    the weights. Everything else (data geometry, aug, optim, ...) stays at mole
    defaults, deliberately NOT adopting the original's rejected choices (e.g. its
    256 px window; see the Phase-2 resolution decision).
    """
    a = vars(args) if hasattr(args, "__dict__") else dict(args or {})
    cfg = _base_config()
    for f in ARCH_FIELDS:
        if f in a and a[f] is not None:
            cfg["model"][f] = a[f]
    return cfg


def _infer_config(backbone: dict) -> dict:
    """Infer a mole config's architecture from bare ViT backbone weights.

    A metadata-less checkpoint (extracted backbone) is fully described by its
    weights: embed_dim, depth, patch size and class-token count. Head dims stay at
    mole defaults (the head is re-initialised on warm-start).
    """
    if "cls_token" not in backbone or "patch_embed.proj.weight" not in backbone:
        raise KeyError("unrecognised checkpoint: no ViT backbone found (expected "
                       "`cls_token` / `patch_embed.proj.weight`). Not a mole/AttMask/"
                       "iBOT or extracted-backbone checkpoint.")
    embed_dim = backbone["cls_token"].shape[-1]
    nct = backbone["cls_token"].shape[1]
    patch = backbone["patch_embed.proj.weight"].shape[-1]
    depth = 1 + max(int(k.split(".")[1]) for k in backbone if k.startswith("blocks."))
    arch = _ARCH_BY_SHAPE.get((embed_dim, depth))
    if arch is None:
        raise ValueError(f"cannot map embed_dim={embed_dim}, depth={depth} to a known ViT "
                         f"(tiny/small/base/large). Non-standard architecture — extend "
                         f"_ARCH_BY_SHAPE or supply a matching config.")
    cfg = _base_config()
    cfg["model"].update(arch=arch, patch_size=int(patch), num_class_tokens=int(nct))
    return cfg


def normalize_checkpoint(ckpt: dict) -> dict:
    """Normalise a mole or foreign checkpoint to ``{backbone, config, global_step, foreign}``.

    ``backbone`` is a bare ViT state dict (heads dropped). ``config`` is the real
    one (mole), synthesized from ``args`` (AttMask run), or inferred from the weights
    (extracted backbone).
    """
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint is a {type(ckpt).__name__}, expected a dict")
    if is_mole_checkpoint(ckpt):
        src = ckpt.get("teacher") or ckpt.get("student")
        return {"backbone": _bare_backbone(src), "config": ckpt["config"],
                "global_step": int(ckpt.get("global_step", 0)), "foreign": False}
    if "teacher" in ckpt or "student" in ckpt:                 # AttMask/iBOT full run
        src = ckpt.get("teacher") or ckpt.get("student")
        backbone = _bare_backbone(src)
        config = _foreign_to_config(ckpt["args"]) if "args" in ckpt else _infer_config(backbone)
    else:                                                       # extracted backbone
        backbone = _bare_backbone(ckpt.get("state_dict", ckpt))
        config = _infer_config(backbone)
    return {"backbone": backbone, "config": config, "global_step": 0, "foreign": True}


def filtered_load(module, state_dict: dict, *, label: str = "") -> dict:
    """Load only the keys that exist in ``module`` with a matching shape (strict-safe).

    ``load_state_dict(strict=False)`` still raises on a shape mismatch of a shared
    key, so we pre-filter. Returns a report of what loaded / was skipped, letting
    a warm-start proceed while clearly flagging any re-initialised parameters.
    """
    own = module.state_dict()
    keep, shape_mismatch = {}, []
    for k, v in state_dict.items():
        if k in own:
            if own[k].shape == v.shape:
                keep[k] = v
            else:
                shape_mismatch.append(k)
    module.load_state_dict(keep, strict=False)
    return {
        "label": label,
        "loaded": len(keep),
        "total": len(own),
        "missing": [k for k in own if k not in keep],       # left at init
        "unexpected": [k for k in state_dict if k not in own],
        "shape_mismatch": shape_mismatch,                   # left at init
    }
