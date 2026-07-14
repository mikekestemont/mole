"""Embedding extraction: pooling (mean/cls/patches), VLAD, output formats."""

from __future__ import annotations

from mole.embed.extract import embed, load_backbone
from mole.embed.pooling import Pooling

__all__ = ["Pooling", "embed", "load_backbone"]
