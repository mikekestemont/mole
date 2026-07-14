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
          leave: bool = True, position: int | None = None, initial: int = 0) -> Iterator[T]:
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
        visible (default). Set ``False`` to clear it on completion (e.g. an inner
        bar nested under an outer one).
    position:
        Row for nested bars — ``0`` for the outer bar, ``1`` for the inner.
    initial:
        Starting count (e.g. resuming an epoch bar mid-training).
    """
    from tqdm.auto import tqdm

    return tqdm(iterable, desc=description, total=total, unit=unit, disable=disable,
                leave=leave, dynamic_ncols=True, position=position, initial=initial)


def write(message: str) -> None:
    """Print a line without corrupting an active progress bar.

    tqdm bars live on stderr and are redrawn continuously; a bare ``print`` in the
    middle of a loop tears the bar. Route any in-loop status line through here
    (tqdm's own ``write``) so it lands cleanly above the running bar.
    """
    from tqdm.auto import tqdm

    tqdm.write(message)


def progress_bar(total: int | None = None, description: str = "Working", *, unit: str = "it",
                 position: int | None = None, leave: bool = True, initial: int = 0,
                 disable: bool = False, bar_format: str | None = None):
    """A manually-updated progress bar (call ``.update(n)`` / ``.set_postfix(...)``).

    Use for loops that aren't a simple ``for x in track(iterable)`` — e.g. two
    fixed nested bars where the inner one is ``reset()`` each outer step (the
    reliable way to render stacked bars in a terminal; recreating the inner bar
    each iteration breaks the nesting). Pass ``bar_format`` with equal-width
    ``{desc}`` fields to align stacked bars vertically.
    """
    from tqdm.auto import tqdm

    return tqdm(total=total, desc=description, unit=unit, position=position, leave=leave,
                dynamic_ncols=True, initial=initial, disable=disable, bar_format=bar_format)
