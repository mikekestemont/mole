"""Configuration loading for MOLE.

One YAML file carries every hyperparameter; the CLI overrides any leaf with
``--set a.b.c=value`` (values parsed as YAML scalars, so ``optim.lr=1e-4`` and
``train.fp16=true`` do the right thing). A minimal YAML works because everything
falls back to :data:`DEFAULTS`.

Groups: ``data / aug / model / mask / optim / loss / train``. Crop counts
(global/local) are NOT here — they come from the augmentation preset
(``mole.data.augment``), the single source of truth, so they can't drift.

Resolution note (see :mod:`mole.data.patches`): ``data.window_size`` is the crop
lifted from the page; ``data.model_size`` is what the ViT ingests.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "data": {
        "path": "data/samples",       # dataset root (flat folder or subfolders-as-datasets)
        "window_size": 512,           # physical crop from the page
        "model_size": 224,            # ViT input size
        "overlap": 0.5,               # window grid overlap
        "use_zones": True,            # restrict windows to prep text zones (zones.json)
        "num_workers": 8,
    },
    "aug": {
        "preset": "mild",             # mild | default | aggressive (source of crop counts)
        "overrides": {},              # optional per-field AugConfig overrides
    },
    "model": {
        "arch": "vit_small",          # vit_tiny | vit_small | vit_base | vit_large
        "patch_size": 16,
        "out_dim": 256,               # CLS projection dim
        "patch_out_dim": 256,         # patch projection dim
        "num_class_tokens": 1,
        "shared_head": False,
        "shared_head_teacher": True,
        "norm_last_layer": True,
        "norm_in_head": None,
        "act_in_head": "gelu",
        "momentum_teacher": 0.996,    # base EMA; cosine-annealed to 1
        "drop_path": 0.1,
        "use_masked_im_modeling": True,
    },
    "mask": {
        "pred_ratio": 0.3,            # fraction of tokens masked
        "pred_ratio_var": 0.0,
        "pred_shape": "attmask_high", # attmask_high | attmask_hint | attmask_low | rand | block
        "masking_prob": 0.5,
        "show_max": 0.1,
        "pred_start_epoch": 0,
    },
    "optim": {
        "optimizer": "adamw",         # adamw | sgd | lars
        "lr": 5.0e-4,                 # peak LR at reference batch 256 (linearly scaled)
        "min_lr": 1.0e-6,
        "warmup_epochs": 10,
        "weight_decay": 0.04,
        "weight_decay_end": 0.4,
        "clip_grad": 3.0,
        "freeze_last_layer": 1,
        "batch_size": 128,            # per-GPU batch size
    },
    "loss": {
        "warmup_teacher_temp": 0.04,
        "teacher_temp": 0.04,
        "warmup_teacher_patch_temp": 0.04,
        "teacher_patch_temp": 0.07,
        "warmup_teacher_temp_epochs": 0,
        "lambda1": 1.0,               # CLS (DINO) loss weight
        "lambda2": 1.0,               # patch (iBOT/MIM) loss weight
    },
    "train": {
        "epochs": 100,
        "seed": 0,
        "fp16": True,                 # mixed precision
        "output_dir": "runs/base",
        "save_every_steps": 500,      # periodic checkpoint (seamless resume)
        "saveckp_epoch_freq": 10,     # keep a numbered checkpoint every N epochs
        "tensorboard": True,          # write TensorBoard scalars into the run dir
        "tb_every_steps": 10,         # scalar logging cadence (steps)
        "projector": True,            # log document embeddings to the TB projector tab
        "projector_every_epochs": 5,  # projector logging cadence (epochs)
        "projector_max_images": 300,  # cap documents embedded per projector snapshot
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def apply_override(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``config[a][b][c] = value`` for a ``dotted_key`` of ``'a.b.c'``."""
    keys = dotted_key.split(".")
    node = config
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise KeyError(f"override path {dotted_key!r} traverses a non-dict at {k!r}")
    node[keys[-1]] = value


def _parse_override_value(raw: str) -> Any:
    """Parse a ``--set`` value as a YAML scalar, coercing numeric strings.

    PyYAML's 1.1 resolver does NOT treat dot-less scientific notation (``1e-5``,
    ``2e+3``) as a number, leaving it a string — which then breaks arithmetic on
    e.g. ``optim.lr``. We recover those by trying int then float before giving up.
    """
    import yaml

    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config over :data:`DEFAULTS` and apply ``key.path=value`` overrides."""
    import yaml

    config = copy.deepcopy(DEFAULTS)
    if path:
        user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        _deep_merge(config, user)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set expects key.path=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        apply_override(config, key.strip(), _parse_override_value(raw))
    return config


def config_hash(config: dict[str, Any]) -> str:
    """Stable short hash of a config (for run manifests / lineage)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
