"""MOLE — continual self-supervised embeddings for premodern handwriting.

Named after *mole*, the Mexican sauce continually remade from the previous day's
leftovers: the model is continually re-pretrained on a mix of old and new data.

Built on Tim Raven's adaptation of AttMask (Kakogeorgiou et al.), itself in the
DINO / iBOT lineage.

This top-level module deliberately imports nothing heavy (no torch) so that
``import mole`` and ``mole --help`` stay fast. Submodules import their heavy
dependencies lazily, inside functions.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]
