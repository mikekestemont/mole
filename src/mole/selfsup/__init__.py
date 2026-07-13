"""Self-supervised pretraining: AttMask training, continual updates, finetuning.

Ported AttMask / iBOT / DINO machinery. Labels are never used here.
"""

from __future__ import annotations

from mole.selfsup.attmask import AttMask
from mole.selfsup.head import iBOTHead
from mole.selfsup.vit import VIT_ARCHS, build_vit
from mole.selfsup.wrapper import MultiCropWrapper, get_params_groups, has_batchnorms

__all__ = [
    "AttMask", "iBOTHead", "VIT_ARCHS", "build_vit",
    "MultiCropWrapper", "get_params_groups", "has_batchnorms",
]
