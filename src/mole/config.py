"""Configuration loading for MOLE.

One YAML file carries every hyperparameter; the CLI can override any leaf with
``--set a.b.c=value``. The full, validated config *schema* (which fields exist,
their types and defaults) is a deliberate Phase-4 decision and is NOT frozen
here — this module only provides the load + override plumbing so the shape of
the API is fixed early.

Design note (resolution): two independent sizes live in the config and must not
be conflated:

* ``data.window_size``  -- physical crop lifted from the page (default 256 px).
* ``data.model_size``   -- what the ViT ingests (default 224 px).

Training resizes window -> model_size; embedding resizes window -> model_size
deterministically (no positional-encoding interpolation at inference).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config and apply ``key.path=value`` overrides.

    Parameters
    ----------
    path:
        Path to a YAML config file.
    overrides:
        List of ``dotted.key=value`` strings (from ``--set`` CLI flags). Values
        are parsed as YAML scalars, so ``optim.lr=1e-4`` and ``train.fp16=true``
        do the right thing.

    Returns
    -------
    dict
        The merged configuration.
    """
    raise NotImplementedError("Config loading is implemented in Phase 4 (mole train).")


def apply_override(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``config[a][b][c] = value`` for a ``dotted_key`` of ``'a.b.c'``."""
    raise NotImplementedError("Config loading is implemented in Phase 4 (mole train).")
