"""MediaPipe Face Landmarker wrapper: returns pixel-space landmarks for the primary driver."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config import FaceConfig


@dataclass(frozen=True)
class FaceLandmarks:
    """Landmarks of one face in PIXEL coordinates."""
    points: np.ndarray                    # shape (478, 2)
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
    num_faces: int
    primary: Optional[FaceLandmarks]


class FaceDetector:
    """MediaPipe Tasks Face Landmarker wrapper; selects the largest visible face."""

    def __init__(self, cfg: FaceConfig) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Run: pip install -r requirements.txt"
            ) from exc

        model_path = Path(cfg.model_path)
        if not model_path.is_file():
            raise RuntimeError(
                f"Face Landmarker model not found: {model_path}. "
                "Run run.ps1 to download the required model."
            )

        self._mp = mp
        self._last_timestamp_ms = -1
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=cfg.max_faces,
            min_face_detection_confidence=cfg.min_detection_confidence,
            min_face_presence_confidence=cfg.min_tracking_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, now: float) -> FaceDetectionResult:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = max(self._last_timestamp_ms + 1, int(now * 1000))
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        faces = result.face_landmarks
        if not faces:
            return FaceDetectionResult(num_faces=0, primary=None)

        scale = np.array([w, h], dtype=np.float64)
        candidates = []
        for face in faces:
            points = np.array([(lm.x, lm.y) for lm in face], dtype=np.float64) * scale
            candidates.append(FaceLandmarks.from_points(points, (w, h)))
        primary = max(candidates, key=lambda candidate: candidate.area)
        return FaceDetectionResult(num_faces=len(candidates), primary=primary)

    def close(self) -> None:
        self._landmarker.close()
