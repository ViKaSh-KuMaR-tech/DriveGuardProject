"""
config.py - the ONE place for every tunable value in DriveGuard.

All thresholds, durations, weights and paths live here as small frozen
dataclasses. Components receive the section they need in their constructor,
so nothing else in the project hard-codes a number.

Time values are in SECONDS (not frames) so behaviour is identical whether your
webcam delivers 15 FPS or 60 FPS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0                  # 0 = default webcam
    width: int = 640                # requested capture size (driver may pick the closest)
    height: int = 480
    max_read_failures: int = 30     # consecutive failed reads before giving up


@dataclass(frozen=True)
class FaceConfig:
    max_faces: int = 3              # MediaPipe tracks up to this many faces
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class EyeConfig:
    ear_threshold: float = 0.21     # EAR below this => eye considered closed
    closed_duration_s: float = 1.0  # eyes must stay closed this long => DROWSY
    open_grace_s: float = 0.1       # tiny debounce so one noisy "open" frame doesn't reset the timer


@dataclass(frozen=True)
class MouthConfig:
    mar_threshold: float = 0.55     # MAR above this => mouth considered wide open
    yawn_duration_s: float = 1.5    # mouth must stay open this long => YAWN
    close_grace_s: float = 0.3


@dataclass(frozen=True)
class HeadPoseConfig:
    yaw_max_deg: float = 30.0       # looking left/right beyond this => away from road
    pitch_max_deg: float = 25.0     # looking up/down (e.g. at a phone in lap) beyond this
    roll_max_deg: float = 45.0      # head tilted sideways beyond this
    distraction_duration_s: float = 2.0
    return_grace_s: float = 0.3
    angle_smoothing: float = 0.4    # 0..1, weight of the NEW sample (lower = smoother, laggier)


@dataclass(frozen=True)
class PresenceConfig:
    absence_grace_s: float = 2.0    # face must be missing this long => driver ABSENT


@dataclass(frozen=True)
class PhoneConfig:
    model_path: str = str(BASE_DIR / "models" / "yolov8n.pt")
    confidence: float = 0.40
    class_names: Tuple[str, ...] = ("cell phone",)   # COCO class name(s) treated as a phone
    inference_every_n_frames: int = 3   # run YOLO on every Nth frame to keep FPS high
    image_size: int = 640
    device: str | None = None           # None = auto ("cpu", "cuda:0", "mps" ...)
    use_duration_s: float = 0.5         # phone must be seen this long => PHONE IN USE
    release_grace_s: float = 1.0        # keep status this long after phone disappears


@dataclass(frozen=True)
class SafetyConfig:
    # Risk points added while each (already time-filtered) behaviour is active.
    weights: Dict[str, float] = field(default_factory=lambda: {
        "drowsy": 60.0,
        "absent": 60.0,
        "phone": 45.0,
        "distracted": 40.0,
        "yawning": 20.0,
    })
    # Smoothed risk points -> level. Safety score = 100 - risk points.
    low_at: float = 5.0
    medium_at: float = 25.0
    high_at: float = 50.0
    # Smoothing time-constants: risk rises quickly, falls slowly.
    rise_tau_s: float = 0.5
    fall_tau_s: float = 2.0
    # Speak only when the level reaches this value.
    alert_min_level: str = "MEDIUM"
    # If several behaviours are active, the first one in this list is announced.
    alert_priority: Tuple[str, ...] = ("drowsy", "phone", "distracted", "absent", "yawning")
    messages: Dict[str, str] = field(default_factory=lambda: {
        "drowsy": "Please stay alert.",
        "phone": "Please do not use your phone while driving.",
        "distracted": "Please keep your eyes on the road.",
        "absent": "Driver not detected. Please face the camera.",
        "yawning": "You seem tired. Consider taking a break.",
    })


@dataclass(frozen=True)
class AudioConfig:
    enabled: bool = True
    global_cooldown_s: float = 3.0      # minimum gap between ANY two alerts
    per_event_cooldown_s: float = 10.0  # minimum gap before the SAME alert repeats
    speech_rate: int = 170              # words/min (pyttsx3 backend only)
    volume: float = 1.0                 # 0..1 (pyttsx3 backend only)


@dataclass(frozen=True)
class UIConfig:
    window_name: str = "DriveGuard"
    show_landmarks: bool = True         # toggle at runtime with 'l'
    font_scale: float = 0.55


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    eye: EyeConfig = field(default_factory=EyeConfig)
    mouth: MouthConfig = field(default_factory=MouthConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    phone: PhoneConfig = field(default_factory=PhoneConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ui: UIConfig = field(default_factory=UIConfig)


CONFIG = Config()
