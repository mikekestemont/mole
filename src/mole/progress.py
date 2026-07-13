"""Shared progress-bar helper for MOLE.

CONVENTION (project-wide): every long-running loop — prep detection, augview
rendering, training epochs/steps, embedding extraction, k-means, eval — wraps its
iterable in :func:`track` so the user always gets a live progress bar. Do not use
bare ``for`` loops for anything that can take more than a second or two; do not
call ``tqdm`` directly. Route everything through here so the look is consistent
and the backend is swappable in one place.

Built on ``tqdm`` (``tqdm.auto`` picks the right renderer for terminal vs.
notebook). Bars write to stderr, so ``✓`` status lines on stdout stay clean.
"""

from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def track(iterable: Iterable[T], description: str = "Working", *,
          total: int | None = None, unit: str = "it", disable: bool = False,
          leave: bool = True) -> Iterator[T]:
    """Wrap ``iterable`` in a progress bar.

    Parameters
    ----------
    iterable:
        Anything iterable. Pass ``total`` for generators without ``len``.
    description:
        Short label shown to the left of the bar (imperative, e.g. "Detecting
        text zones").
    unit:
        Noun for one step ("page", "img", "batch", "step").
    disable:
        Silence the bar (e.g. inside library calls / tests).
    leave:
        Keep the finished bar on screen so its final elapsed time / rate stays
        visible (default). Set ``False`` to clear it on completion.
    """
    from tqdm.auto import tqdm

    return tqdm(iterable, desc=description, total=total, unit=unit,
                disable=disable, leave=leave, dynamic_ncols=True)
