"""Pooling strategies for turning patch tokens into a page embedding.

* ``mean``    -- L2-normed mean of the (foreground) patch tokens. Codebook-free,
                 384-dim, incremental (embed one document without any refit).
* ``meanstd`` -- mean concatenated with the per-dimension std of the (foreground)
                 tokens (768-dim). A cheap second-order descriptor that captures
                 the token *distribution* VLAD exploits, still codebook-free.
* ``cls``     -- the [CLS] token(s), flattened when there is more than one.
* ``vlad``    -- VLAD aggregation (reproducible codebook, see vlad.py). Strongest,
                 but 38400-dim and needs a fitted codebook (transductive / per-set).
* ``patches`` -- raw per-patch embeddings, no pooling.

``mean``/``meanstd`` honour the foreground filter (they aggregate the same
foreground patch descriptors VLAD does); ``cls`` cannot (a window has one class
token, not per-patch ones), so foreground filtering does not apply to it.

The ViT returns, per window, a token sequence ``[num_class_tokens + num_patches,
dim]`` (class tokens first, exactly as the training model lays them out). The
helpers below split that sequence and pool it two ways:

* per **window** -> a single vector (``mean`` / ``cls``), then windows of a page
  are averaged into the page embedding by the extraction driver;
* per **window** -> its raw patch descriptors (``patches`` / ``vlad`` input),
  which the driver concatenates across a page.

An optional PCA-whitening flag (a classic retrieval trick) is exposed at the
``mole embed`` level and applied to the finished page matrix.
"""

from __future__ import annotations

from enum import Enum


class Pooling(str, Enum):
    MEAN = "mean"
    MEANSTD = "meanstd"
    CLS = "cls"
    VLAD = "vlad"
    PATCHES = "patches"


def split_tokens(tokens, num_class_tokens: int):
    """Split a ``[B, seq, dim]`` token batch into (class_tokens, patch_tokens)."""
    return tokens[:, :num_class_tokens], tokens[:, num_class_tokens:]


def pool_window(tokens, num_class_tokens: int, strategy: "Pooling | str"):
    """Reduce a ``[B, seq, dim]`` window-token batch to one ``[B, D]`` vector.

    ``mean`` averages the patch tokens; ``cls`` takes the class token(s) and
    flattens them to ``num_class_tokens * dim`` (matching how the training model
    forms its CLS representation).
    """
    strategy = Pooling(strategy)
    cls, patches = split_tokens(tokens, num_class_tokens)
    if strategy is Pooling.MEAN:
        return patches.mean(dim=1)
    if strategy is Pooling.CLS:
        return cls.reshape(cls.shape[0], -1)
    raise ValueError(f"pool_window does not handle {strategy!r} (patch-level strategy)")


def patch_descriptors(tokens, num_class_tokens: int):
    """Return the raw patch tokens ``[B, num_patches, dim]`` (drops class tokens)."""
    _, patches = split_tokens(tokens, num_class_tokens)
    return patches
