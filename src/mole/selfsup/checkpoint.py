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
