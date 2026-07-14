"""Minimal DDP (DistributedDataParallel) helpers for single-node multi-GPU training.

Design goals:

* **Single-GPU stays byte-for-byte the same.** When not launched under
  ``torchrun`` (``WORLD_SIZE`` unset or ``1``), :func:`setup` is a no-op and every
  helper reports rank 0 / world size 1 — the training loop takes exactly its
  previous path.
* **Launch via ``torchrun``**, which sets ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``
  in the environment. We read those; we never spawn processes ourselves.
* **NCCL backend**, one process per GPU, ``LOCAL_RANK`` selects the CUDA device.

The ViT uses LayerNorm (no BatchNorm), so no ``SyncBatchNorm`` conversion is needed.

Typical use in ``train()``::

    dist = setup()                      # reads torchrun env; no-op if single-proc
    device = dist.device                # cuda:LOCAL_RANK, or _pick_device() fallback
    ...
    if dist.is_distributed:
        student_fwd = DDP(student, device_ids=[dist.local_rank], find_unused_parameters=True)
    ...
    if dist.is_main:                    # gate all logging / checkpoint writes
        save_checkpoint(...)
    dist.barrier()                      # sync at explicit points only
    ...
    dist.cleanup()                      # in a finally
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistInfo:
    """Immutable snapshot of the process's place in the (possibly trivial) group."""

    rank: int
    local_rank: int
    world_size: int
    device: "object"  # torch.device — typed loosely to keep torch import lazy at module top

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """True on exactly one process (rank 0) — the only one that logs / checkpoints."""
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def setup() -> DistInfo:
    """Initialise the process group from ``torchrun`` env, or return a trivial group.

    Returns a :class:`DistInfo`. When ``WORLD_SIZE <= 1`` (i.e. not launched under
    ``torchrun``, or launched with one process) this does NOT init any process
    group and picks the usual device — so single-GPU / CPU / MPS runs are untouched.
    """
    import torch

    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    if world_size <= 1:
        # Not distributed: fall back to the standard device pick (cuda/mps/cpu).
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return DistInfo(rank=0, local_rank=0, world_size=1, device=device)

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed launch requested (WORLD_SIZE>1) but CUDA is unavailable.")

    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        # Pass device_id so collectives infer the right device (mutes the
        # "using the device under current context" warning on barrier/all_reduce).
        dist.init_process_group(backend="nccl", device_id=device)
    return DistInfo(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def barrier() -> None:
    """Block until all ranks reach this point (no-op if not distributed)."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def any_rank_stopping(flag: bool, device) -> bool:
    """Collective OR of ``flag`` across all ranks (passthrough if not distributed).

    Used so a SIGINT caught by *any* rank makes *every* rank stop on the SAME
    training iteration. Without this, ranks can decide to stop on different
    iterations and the one still looping deadlocks on the next gradient all-reduce
    (its peer has already left). The all-reduce is itself the rendezvous, so every
    rank leaves the loop together. One int scalar per step — negligible overhead.
    """
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return flag
    t = torch.tensor([1 if flag else 0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def cleanup() -> None:
    """Tear down the process group if one was created (safe to call unconditionally)."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
