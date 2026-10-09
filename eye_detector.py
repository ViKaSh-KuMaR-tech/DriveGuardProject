"""Eye Aspect Ratio (EAR) drowsiness detection with temporal logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import EyeConfig
from utils.geometry import eye_aspect_ratio
from utils.timing import PersistenceTimer

# MediaPipe Face Mesh indices, ordered p1..p6 as eye_aspect_ratio() expects
# (outer corner, upper, upper, inner corner, lower, lower).
RIGHT_EYE_IDX = (33, 160, 158, 133, 153, 144)    # the driver's right eye
LEFT_EYE_IDX = (362, 385, 387, 263, 373, 380)    # the driver's left eye


@dataclass(frozen=True)
class EyeState:
    ear: Optional[float]          # None when no face is available
    eyes_closed: bool             # instantaneous: EAR below threshold
    drowsy: bool                  # persistent: closed for >= closed_duration_s
    closed_duration_s: float


class EyeDetector:
    def __init__(self, cfg: EyeConfig) -> None:
        self._cfg = cfg
        self._timer = PersistenceTimer(cfg.closed_duration_s, cfg.open_grace_s)

    def update(self, landmarks: Optional[np.ndarray], now: float) -> EyeState:
        if landmarks is None:
            drowsy = self._timer.update(False, now)
            return EyeState(None, False, drowsy, self._timer.elapsed)

        # Average both eyes: more robust than either alone (one may be occluded
        # or partly outside the frame when the head is turned).
        ear_right = eye_aspect_ratio(landmarks[list(RIGHT_EYE_IDX)])
        ear_left = eye_aspect_ratio(landmarks[list(LEFT_EYE_IDX)])
        ear = (ear_right + ear_left) / 2.0

        closed = ear < self._cfg.ear_threshold
        drowsy = self._timer.update(closed, now)    # a normal blink (~0.1-0.4 s) never reaches 1 s
        return EyeState(ear, closed, drowsy, self._timer.elapsed)
