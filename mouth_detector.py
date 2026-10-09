"""Mouth Aspect Ratio (MAR) yawn detection with temporal logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import MouthConfig
from utils.geometry import mouth_aspect_ratio
from utils.timing import PersistenceTimer

# Inner-lip landmarks in the order mouth_aspect_ratio() expects:
# left corner, right corner, then three (upper, lower) vertical pairs.
MOUTH_IDX = (78, 308, 82, 87, 13, 14, 312, 317)


@dataclass(frozen=True)
class MouthState:
    mar: Optional[float]
    mouth_open: bool              # instantaneous
    yawning: bool                 # persistent
    open_duration_s: float


class MouthDetector:
    def __init__(self, cfg: MouthConfig) -> None:
        self._cfg = cfg
        self._timer = PersistenceTimer(cfg.yawn_duration_s, cfg.close_grace_s)

    def update(self, landmarks: Optional[np.ndarray], now: float) -> MouthState:
        if landmarks is None:
            yawning = self._timer.update(False, now)
            return MouthState(None, False, yawning, self._timer.elapsed)

        mar = mouth_aspect_ratio(landmarks[list(MOUTH_IDX)])
        is_open = mar > self._cfg.mar_threshold
        yawning = self._timer.update(is_open, now)   # talking/laughing is too brief or too small
        return MouthState(mar, is_open, yawning, self._timer.elapsed)
