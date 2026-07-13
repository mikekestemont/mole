"""VLAD encoding with a reproducible, versioned codebook.

Fix (vs. original code): the k-means codebook uses a fixed, configurable seed;
fitting is reproducible and the fitted codebook is SAVED with the run so
embeddings are stable across invocations. Codebooks are bound to and versioned
with the model ID whose descriptors trained them.
"""

from __future__ import annotations

from pathlib import Path


def fit_codebook(descriptors, n_clusters: int = 100, seed: int = 0):
    """Fit a reproducible k-means codebook on patch descriptors."""
    raise NotImplementedError("VLAD is implemented in Phase 5.")


def vlad_encode(descriptors, codebook, powernorm: bool = True):
    """VLAD-encode a page's descriptors against a fitted codebook."""
    raise NotImplementedError("VLAD is implemented in Phase 5.")
