"""Self-supervised training entry points.

Two clearly separated modes:

* ``train(..., mode="scratch")`` / ``mode="continual"`` -- advance the *base
  lineage*. Continual mixes new data with a replay buffer of previous datasets
  (compact per-dataset patch shards) at a configurable ratio, with lower peak LR
  and warmup. This is the mole metaphor: always fold in yesterday's leftovers.
* ``finetune(...)`` -- dataset-specific adaptation. Branches from a base
  checkpoint into a NEW run dir; the base model is never overwritten.

Checkpoints are seamless-resume capable: model, EMA teacher, optimizer,
scheduler, epoch/step, and ALL RNG states. Ctrl-C checkpoints cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal


def train(config_path: str | Path, output_dir: str | Path | None = None,
          mode: Literal["scratch", "continual"] = "scratch",
          resume: str | Path | None = None, overrides: list[str] | None = None):
    """Run (or resume) AttMask pretraining.

    ``mode="continual"`` mixes a replay buffer of previous datasets' patch shards
    into the new data. Auto-resumes if ``output_dir`` already holds a checkpoint.
    """
    raise NotImplementedError("Training is implemented in Phase 4.")


def finetune(config_path: str | Path, base_checkpoint: str | Path,
             output_dir: str | Path | None = None, overrides: list[str] | None = None):
    """Branch from ``base_checkpoint`` for dataset-specific adaptation.

    Never mutates the base. Full-finetune vs. LoRA/adapter is a decision surfaced
    in Phase 7.
    """
    raise NotImplementedError("Finetuning is implemented in Phase 7.")
