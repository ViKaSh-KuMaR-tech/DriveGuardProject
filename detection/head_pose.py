"""Head-pose (yaw / pitch / roll) estimation with solvePnP + distraction logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config import HeadPoseConfig
from utils.timing import PersistenceTimer

# MediaPipe landmark ids used for pose: nose tip, chin, outer corner of the
# driver's right eye, outer corner of the driver's left eye, mouth corners.
LANDMARK_IDS = (1, 152, 33, 263, 61, 291)

# A generic 3-D face model (millimetre-ish units) for those six landmarks.
# Model frame = camera frame for a driver looking straight at the camera:
#   +x -> image right, +y -> down, +z -> away from the camera.
# The nose tip is the origin and is the part of the face closest to the camera,
# so every other point sits at positive z.
# Order matches LANDMARK_IDS. Landmark 33 / 61 appear on the image-LEFT of an
# un-mirrored frame, 263 / 291 on the image-RIGHT.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, 330.0, 65.0),       # chin
    (-225.0, -170.0, 135.0),  # right-eye outer corner (image left)
    (225.0, -170.0, 135.0),   # left-eye outer corner  (image right)
    (-150.0, 150.0, 125.0),   # right mouth corner (image left)
    (150.0, 150.0, 125.0),    # left mouth corner  (image right)
], dtype=np.float64)

_AXIS_LENGTH = 150.0


@dataclass(frozen=True)
class HeadPoseState:
    yaw: Optional[float]                  # degrees, relative to the calibrated neutral pose
    pitch: Optional[float]
    roll: Optional[float]
    looking_away: bool                    # instantaneous: any angle beyond its limit
    distracted: bool                      # persistent
    nose_point: Optional[Tuple[int, int]]  # for drawing
    axis_point: Optional[Tuple[int, int]]  # tip of the "where the face points" line


class HeadPoseEstimator:
    def __init__(self, cfg: HeadPoseConfig) -> None:
        self._cfg = cfg
        self._timer = PersistenceTimer(cfg.distraction_duration_s, cfg.return_grace_s)
        self._smoothed: Optional[np.ndarray] = None    # [yaw, pitch, roll], raw (uncalibrated)
        self._offset = np.zeros(3)                      # neutral pose set by calibrate()
        self._guess: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # ------------------------------------------------------------------ public
    def update(self, landmarks: Optional[np.ndarray], frame_size: Tuple[int, int],
               now: float) -> HeadPoseState:
        raw = self._estimate(landmarks, frame_size) if landmarks is not None else None
        if raw is None:
            self._smoothed = None
            self._guess = None
            distracted = self._timer.update(False, now)
            return HeadPoseState(None, None, None, False, distracted, None, None)

        angles, nose_pt, axis_pt = raw
        a = self._cfg.angle_smoothing                  # exponential moving average vs. jitter
        self._smoothed = angles if self._smoothed is None else a * angles + (1 - a) * self._smoothed
        yaw, pitch, roll = self._smoothed - self._offset

        looking_away = (
            abs(yaw) > self._cfg.yaw_max_deg
            or abs(pitch) > self._cfg.pitch_max_deg
            or abs(roll) > self._cfg.roll_max_deg
        )
        distracted = self._timer.update(looking_away, now)
        return HeadPoseState(float(yaw), float(pitch), float(roll),
                             looking_away, distracted, nose_pt, axis_pt)

    def calibrate(self) -> bool:
        """Treat the CURRENT head pose as 'looking straight at the road'."""
        if self._smoothed is None:
            return False
        self._offset = self._smoothed.copy()
        self._timer.reset()
        return True

    # ----------------------------------------------------------------- private
    def _estimate(self, landmarks: np.ndarray, frame_size: Tuple[int, int]):
        w, h = frame_size
        image_points = landmarks[list(LANDMARK_IDS)].astype(np.float64)

        # Approximate pinhole camera: focal length ~ image width, principal point at
        # the centre, no lens distortion. Good enough for a generic webcam.
        focal = float(w)
        camera_matrix = np.array([[focal, 0, w / 2.0],
                                  [0, focal, h / 2.0],
                                  [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        try:
            if self._guess is None:
                ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix,
                                              dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                # Start from last frame's pose: faster and avoids flipping solutions.
                rvec0, tvec0 = self._guess[0].copy(), self._guess[1].copy()
                ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix,
                                              dist_coeffs, rvec0, tvec0, True,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            self._guess = None
            return None

        if not ok or tvec[2, 0] <= 0:          # face must be in FRONT of the camera
            self._guess = None
            return None
        self._guess = (rvec, tvec)

        # rvec (axis-angle) -> 3x3 rotation matrix -> Euler angles in degrees.
        rotation, _ = cv2.Rodrigues(rvec)
        euler_deg = cv2.RQDecomp3x3(rotation)[0]
        pitch, yaw, roll = (self._wrap(a) for a in euler_deg)   # rotations about x, y, z

        axis_3d = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, -_AXIS_LENGTH)])  # out of the face
        projected, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
        nose_pt = tuple(int(v) for v in projected[0, 0])
        axis_pt = tuple(int(v) for v in projected[1, 0])
        return np.array([yaw, pitch, roll]), nose_pt, axis_pt

    @staticmethod
    def _wrap(angle: float) -> float:
        """Wrap to [-180, 180)."""
        return (float(angle) + 180.0) % 360.0 - 180.0
