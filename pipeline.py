"""
pipeline.py - connects every detector to the safety engine for ONE frame.

Data flow per frame:

    frame -> FaceDetector -> landmarks (primary face) -+-> EyeDetector   -> EyeState
                                                       +-> MouthDetector -> MouthState
                                                       +-> HeadPose      -> HeadPoseState
             face present? --------------------------> PresenceMonitor  -> PresenceState
    frame -> PhoneDetector (YOLO) ---------------------------------------> PhoneState
    all persistent flags ------------------------> SafetyEngine -> SafetyAssessment
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config
from detection.eye_detector import EyeDetector, EyeState
from detection.face_detector import FaceDetectionResult, FaceDetector
from detection.head_pose import HeadPoseEstimator, HeadPoseState
from detection.mouth_detector import MouthDetector, MouthState
from detection.phone_detector import PhoneDetector, PhoneState
from detection.presence import PresenceMonitor, PresenceState
from safety.safety_engine import SafetyAssessment, SafetyEngine, SafetyInputs


@dataclass(frozen=True)
class FrameAnalysis:
    """Everything known about one frame; consumed by the UI and audio code."""
    face: FaceDetectionResult
    presence: PresenceState
    eyes: EyeState
    mouth: MouthState
    head: HeadPoseState
    phone: PhoneState
    safety: SafetyAssessment


class DriveGuardPipeline:
    def __init__(self, config: Config, enable_phone: bool = True) -> None:
        # Build the phone detector first: it is the most likely to fail (missing model)
        # and failing early means no camera / MediaPipe resources are left dangling.
        self._phone = PhoneDetector(config.phone) if enable_phone else None
        self._face = FaceDetector(config.face)
        self._eyes = EyeDetector(config.eye)
        self._mouth = MouthDetector(config.mouth)
        self._head = HeadPoseEstimator(config.head_pose)
        self._presence = PresenceMonitor(config.presence)
        self._engine = SafetyEngine(config.safety)

    def process(self, frame_bgr: np.ndarray, now: float) -> FrameAnalysis:
        h, w = frame_bgr.shape[:2]
        face = self._face.process(frame_bgr)
        landmarks = face.primary.points if face.primary is not None else None

        presence = self._presence.update(landmarks is not None, now)
        eyes = self._eyes.update(landmarks, now)
        mouth = self._mouth.update(landmarks, now)
        head = self._head.update(landmarks, (w, h), now)
        phone = self._phone.update(frame_bgr, now) if self._phone else PhoneState.disabled()

        safety = self._engine.update(
            SafetyInputs(
                drowsy=eyes.drowsy,
                yawning=mouth.yawning,
                distracted=head.distracted,
                driver_absent=presence.absent,
                phone_in_use=phone.in_use,
            ),
            now,
        )
        return FrameAnalysis(face, presence, eyes, mouth, head, phone, safety)

    def calibrate_head_pose(self) -> bool:
        return self._head.calibrate()

    def close(self) -> None:
        self._face.close()

    def __enter__(self) -> "DriveGuardPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
