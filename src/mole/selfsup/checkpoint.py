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
# Warm-starting from a foreign (original AttMask/iBOT) checkpoint: those files
# store `args` (an argparse Namespace) + `epoch` instead of mole's `config` +
# `global_step`, and their DDP-wrapped student carries a `module.` prefix. We
# normalise both formats to a common shape so `train --init-from` and `embed` can
# consume either. Model architecture is authoritative in the checkpoint (the
# weights only load into a matching model), so we recover it from `args`.

# model.* fields that determine parameter shapes — recovered from a foreign
# checkpoint's args and adopted by the warm-started run so weights load.
ARCH_FIELDS = ("arch", "patch_size", "num_class_tokens", "out_dim", "patch_out_dim",
               "shared_head", "shared_head_teacher", "norm_last_layer", "norm_in_head",
               "act_in_head", "use_masked_im_modeling")


def is_mole_checkpoint(ckpt: dict) -> bool:
    """A mole checkpoint carries a `config` dict; a foreign one carries `args`."""
    return isinstance(ckpt, dict) and "config" in ckpt


def _strip_module_prefix(sd: dict) -> dict:
    """Drop the DDP ``module.`` prefix (original student is DDP-wrapped)."""
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}


def _foreign_to_config(args) -> dict:
    """Synthesize a mole config from a foreign checkpoint's argparse args.

    Only the architecture (model.*) is taken from the checkpoint — it must match
    the weights. Everything else (data geometry, aug, optim, ...) stays at mole
    defaults, deliberately NOT adopting the original's rejected choices (e.g. its
    256 px window; see the Phase-2 resolution decision).
    """
    import copy

    from mole.config import DEFAULTS

    a = vars(args) if hasattr(args, "__dict__") else dict(args or {})
    cfg = copy.deepcopy(DEFAULTS)
    for f in ARCH_FIELDS:
        if f in a and a[f] is not None:
            cfg["model"][f] = a[f]
    return cfg


def normalize_checkpoint(ckpt: dict) -> dict:
    """Normalise a mole or foreign checkpoint to ``{student, teacher, config, global_step, foreign}``.

    ``student`` may be ``None`` if the source ships only a teacher. State dicts are
    ``module.``-stripped; ``config`` is the real one (mole) or synthesized from
    ``args`` (foreign).
    """
    if is_mole_checkpoint(ckpt):
        return {"student": ckpt.get("student"), "teacher": ckpt["teacher"],
                "config": ckpt["config"], "global_step": int(ckpt.get("global_step", 0)),
                "foreign": False}
    if "teacher" not in ckpt:
        raise KeyError("checkpoint has neither a mole `config` nor a `teacher` state dict "
                       "— not a recognised AttMask/iBOT/mole checkpoint")
    return {"student": _strip_module_prefix(ckpt["student"]) if "student" in ckpt else None,
            "teacher": _strip_module_prefix(ckpt["teacher"]),
            "config": _foreign_to_config(ckpt.get("args")), "global_step": 0, "foreign": True}


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
