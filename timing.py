"""Temporal-persistence helper shared by every detector."""
from __future__ import annotations

from typing import Optional


class PersistenceTimer:
    """
    Turns a noisy per-frame boolean into a stable, time-filtered state.

    * The state becomes ACTIVE only after `condition` has been True for at least
      `on_after_s` seconds.
    * Once running, a False reading does not cancel anything until `condition`
      has been False for `off_after_s` seconds (a small debounce / grace period).
      With off_after_s = 0 the timer resets the moment the condition ends.

    Uses timestamps (not frame counts) so behaviour is independent of FPS.
    """

    def __init__(self, on_after_s: float, off_after_s: float = 0.0) -> None:
        self._on_after = on_after_s
        self._off_after = off_after_s
        self._true_since: Optional[float] = None
        self._last_true: float = 0.0

    def update(self, condition: bool, now: float) -> bool:
        """Feed one observation; returns whether the filtered state is active."""
        if condition:
            if self._true_since is None:
                self._true_since = now
            self._last_true = now
        elif self._true_since is not None and now - self._last_true >= self._off_after:
            self._true_since = None
        return self.active

    @property
    def elapsed(self) -> float:
        """Seconds the condition has been (continuously, ignoring grace) true."""
        if self._true_since is None:
            return 0.0
        return max(0.0, self._last_true - self._true_since)

    @property
    def active(self) -> bool:
        return self._true_since is not None and self.elapsed >= self._on_after

    def reset(self) -> None:
        self._true_since = None
        self._last_true = 0.0
