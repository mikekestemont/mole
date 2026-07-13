"""Pooling strategies for turning patch tokens into a page embedding.

* ``mean``    -- mean over patch tokens (cheap NLP-style default). DEFAULT.
* ``cls``     -- the [CLS] token.
* ``vlad``    -- VLAD aggregation (optional; reproducible codebook, see vlad.py).
* ``patches`` -- raw per-patch embeddings, no pooling.

An optional PCA-whitening flag (a classic retrieval trick) is exposed at the
``mole embed`` level.
"""

from __future__ import annotations

from enum import Enum


class Pooling(str, Enum):
    MEAN = "mean"
    CLS = "cls"
    VLAD = "vlad"
    PATCHES = "patches"
