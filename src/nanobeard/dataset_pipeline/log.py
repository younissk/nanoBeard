"""Shared rich console + step logging for the dataset pipeline.

A full end-to-end build (stream Wikipedia, download cosmopedia, piratize
millions of rows, train a tokenizer, write bins) runs for a long time. These
helpers give every phase a timestamped, visually distinct line so it is always
clear what the pipeline is doing right now and how long it has been running.
"""

from __future__ import annotations

import time

from rich.console import Console

console = Console()
_T0 = time.monotonic()


def _elapsed() -> str:
    s = int(time.monotonic() - _T0)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def step(msg: str) -> None:
    """A major phase boundary — rendered as a full-width rule."""
    console.rule(f"[bold cyan]{msg}[/]  [dim](+{_elapsed()})[/]")


def info(msg: str) -> None:
    """A sub-step / progress line under the current phase."""
    console.print(f"[dim]+{_elapsed()}[/] {msg}")
