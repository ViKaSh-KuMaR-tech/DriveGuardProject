"""MediaPipe Face Mesh wrapper: returns pixel-space landmarks for the primary driver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config import FaceConfig


@dataclass(frozen=True)
class FaceLandmarks:
    """Landmarks of one face in PIXEL coordinates."""
    points: np.ndarray                    # shape (478, 2) with refine_landmarks=True
    bbox: Tuple[int, int, int, int]       # x1, y1, x2, y2 (clipped to the frame)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def from_points(points: np.ndarray, frame_size: Tuple[int, int]) -> "FaceLandmarks":
        w, h = frame_size
        x1, y1 = np.floor(points.min(axis=0)).astype(int)
        x2, y2 = np.ceil(points.max(axis=0)).astype(int)
        bbox = (max(0, int(x1)), max(0, int(y1)), min(w - 1, int(x2)), min(h - 1, int(y2)))
        return FaceLandmarks(points=points, bbox=bbox)


@dataclass(frozen=True)
class FaceDetectionResult:
    num_faces: int                        # 0, 1 or several
    primary: Optional[FaceLandmarks]      # the driver (largest face), or None


class FaceDetector:
    """
    Wraps MediaPipe Face Mesh (468 landmarks + 10 iris landmarks).

    Driver selection: when several faces are visible, the one with the LARGEST
    bounding box is the "primary" face. In a car the driver is normally the
    person closest to a dashboard-mounted camera, hence the biggest face.
    """

    def __init__(self, cfg: FaceConfig) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Run: pip install -r requirements.txt"
            ) from exc
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This MediaPipe version no longer ships the 'solutions' API used here. "
                "Install the pinned version: pip install mediapipe==0.10.14"
            )
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,          # video mode: detect once, then track (faster)
            max_num_faces=cfg.max_faces,
            refine_landmarks=True,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> FaceDetectionResult:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # MediaPipe expects RGB
        rgb.flags.writeable = False                         # lets MediaPipe avoid a copy
        results = self._mesh.process(rgb)

        faces = results.multi_face_landmarks
        if not faces:
            return FaceDetectionResult(num_faces=0, primary=None)

        scale = np.array([w, h], dtype=np.float64)          # normalised [0,1] -> pixels
        best: Optional[FaceLandmarks] = None
        for face in faces:
            pts = np.array([(lm.x, lm.y) for lm in face.landmark], dtype=np.float64) * scale
            candidate = FaceLandmarks.from_points(pts, (w, h))
            if best is None or candidate.area > best.area:
                best = candidate
        return FaceDetectionResult(num_faces=len(faces), primary=best)

    def close(self) -> None:
        self._mesh.close()
