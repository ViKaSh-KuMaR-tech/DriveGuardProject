"""Driver-absence detection with a grace period."""
from __future__ import annotations

from dataclasses import dataclass

from config import PresenceConfig
from utils.timing import PersistenceTimer


@dataclass(frozen=True)
class PresenceState:
    face_visible: bool            # a face is visible THIS frame
    absent: bool                  # no face for longer than the grace period
    missing_duration_s: float


class PresenceMonitor:
    def __init__(self, cfg: PresenceConfig) -> None:
        # Driver counts as present again the instant a face reappears (off_after = 0).
        self._timer = PersistenceTimer(cfg.absence_grace_s, 0.0)

    def update(self, face_visible: bool, now: float) -> PresenceState:
        absent = self._timer.update(not face_visible, now)
        return PresenceState(face_visible, absent, self._timer.elapsed)
