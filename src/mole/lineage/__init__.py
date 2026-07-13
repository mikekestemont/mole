"""Model lineage & versioning (a core feature, not an afterthought).

The base lineage advances linearly (``base@v1``, ``base@v2``, ...) via
``mole train --mode continual``; ``mole finetune`` creates named branches
(``base@v3/stgallen@v1``) that never advance the base. An append-only registry
records per version: parent ID, config hash, dataset manifest(s), replay
composition, seed, date, and eval scores if available.
"""

from __future__ import annotations

__all__: list[str] = []
