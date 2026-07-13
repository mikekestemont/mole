"""Linear / classification probes on frozen embeddings (SCAFFOLD).

No training logic in this phase — interface only.
"""

from __future__ import annotations

from pathlib import Path


def train_probe(embeddings_path: str | Path, labels_root: str | Path,
                output_dir: str | Path | None = None):
    """Fit a linear/classification probe on frozen embeddings."""
    raise NotImplementedError("Probes are scaffolded now, implemented later.")
