"""Wall-clock timing and storage accounting."""
from __future__ import annotations

import time
from typing import Callable

BYTES_PER_VALUE = 4  # representations are stored as float32


class Timer:
    """Context manager storing the elapsed seconds in ``self.elapsed``.

    ``sync`` runs before the clock stops; GPU code passes ``torch.cuda.synchronize`` so pending
    work is included.
    """

    def __init__(self, sync: Callable[[], None] | None = None) -> None:
        self.elapsed = 0.0
        self._sync = sync
        self._start = 0.0

    def __enter__(self) -> "Timer":
        if self._sync:
            self._sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self._sync:
            self._sync()
        self.elapsed = time.perf_counter() - self._start


def representation_bytes(n_images: int, dim: int, bytes_per_value: int = BYTES_PER_VALUE) -> int:
    """Size of the corpus representation: n_images x dim x bytes per value."""
    return n_images * dim * bytes_per_value
