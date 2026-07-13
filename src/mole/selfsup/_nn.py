"""Small shared nn helpers for the self-supervised backbone."""

from __future__ import annotations

import torch.nn as nn


def trunc_normal_(tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    """Truncated normal init (thin wrapper over ``torch.nn.init``)."""
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
