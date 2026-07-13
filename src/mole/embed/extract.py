"""Embedding extraction driver.

Loads a versioned checkpoint, samples patch windows from each image, runs the
backbone, pools (mean/cls/vlad/patches), and writes ``.npy``/parquet plus a
sidecar mapping (image path -> row) stamped with the producing model ID and
embedding dim — designed so a FAISS index can be built on top later.

Warns loudly if the output directory already contains embeddings from a
different model version (mixed-version indexes are the failure mode the lineage
stamping exists to prevent).
"""

from __future__ import annotations

from pathlib import Path

from mole.embed.pooling import Pooling


def embed(checkpoint: str | Path, input_dir: str | Path, output: str | Path,
          pooling: Pooling | str = Pooling.MEAN, whiten: bool = False,
          overrides: list[str] | None = None):
    """Extract page embeddings for a folder of images.

    Grayscale inputs are replicated to 3 channels transparently. Output is a
    ``.npy``/parquet array plus a sidecar mapping file recording image paths,
    embedding dim, and the producing model ID.
    """
    raise NotImplementedError("Embedding extraction is implemented in Phase 5.")
