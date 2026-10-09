"""OpenCV drawing code: the HUD panel, landmarks, head-pose axis and phone boxes."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np

from config import UIConfig
from detection.eye_detector import LEFT_EYE_IDX, RIGHT_EYE_IDX
from detection.mouth_detector import MOUTH_IDX
from safety.safety_engine import RiskLevel

if TYPE_CHECKING:
    from pipeline import FrameAnalysis

# BGR colours
GREEN = (0, 200, 0)
YELLOW = (0, 220, 255)
ORANGE = (0, 140, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)

LEVEL_COLORS = {RiskLevel.SAFE: GREEN, RiskLevel.LOW: YELLOW,
                RiskLevel.MEDIUM: ORANGE, RiskLevel.HIGH: RED}


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def draw_overlay(frame: np.ndarray, a: "FrameAnalysis", fps: float,
                 show_landmarks: bool, muted: bool, cfg: UIConfig) -> None:
    """Draw everything in place on `frame`."""
    _draw_scene(frame, a, show_landmarks)
    _draw_panel(frame, a, cfg)
    _draw_footer(frame, a, fps, muted, cfg)
    if a.safety.level == RiskLevel.HIGH:
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), RED, 6)


def _draw_scene(frame: np.ndarray, a: "FrameAnalysis", show_landmarks: bool) -> None:
    face = a.face.primary
    if face is not None:
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 1)
        if show_landmarks:
            eye_color = RED if a.eyes.eyes_closed else GREEN
            for idx in RIGHT_EYE_IDX + LEFT_EYE_IDX:
                cv2.circle(frame, tuple(face.points[idx].astype(int)), 1, eye_color, -1)
            mouth_color = ORANGE if a.mouth.mouth_open else GREEN
            for idx in MOUTH_IDX:
                cv2.circle(frame, tuple(face.points[idx].astype(int)), 1, mouth_color, -1)
    if a.head.nose_point is not None and a.head.axis_point is not None:
        cv2.line(frame, a.head.nose_point, a.head.axis_point, YELLOW, 2)
        cv2.circle(frame, a.head.nose_point, 3, YELLOW, -1)

    for box in a.phone.boxes:
        color = RED if a.phone.in_use else ORANGE
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        label = f"phone {box.confidence:.2f}"
        cv2.putText(frame, label, (box.x1, max(15, box.y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _draw_panel(frame: np.ndarray, a: "FrameAnalysis", cfg: UIConfig) -> None:
    drowsy, yawning = a.eyes.drowsy, a.mouth.yawning
    distracted, absent = a.head.distracted, a.presence.absent

    if not a.phone.enabled:
        phone_text, phone_bad = "DISABLED", False
    else:
        phone_text, phone_bad = ("DETECTED", True) if a.phone.in_use else ("NOT DETECTED", False)

    driver_text = "ABSENT" if absent else "PRESENT"
    if not absent and not a.presence.face_visible:
        driver_text += " (searching...)"
    if a.face.num_faces > 1:
        driver_text += f"  [{a.face.num_faces} faces]"

    level = a.safety.level
    # (text, colour) per line
    rows: List[Tuple[str, Tuple[int, int, int]]] = [
        (f"Driver: {driver_text}", RED if absent else GREEN),
        (f"Drowsiness: {'DROWSY' if drowsy else 'NORMAL'}   EAR: {_fmt(a.eyes.ear)}",
         RED if drowsy else GREEN),
        (f"Yawning: {'YES' if yawning else 'NO'}   MAR: {_fmt(a.mouth.mar)}",
         ORANGE if yawning else GREEN),
        (f"Distraction: {'YES' if distracted else 'NO'}", RED if distracted else GREEN),
        (f"  Yaw: {_fmt(a.head.yaw, 1)}  Pitch: {_fmt(a.head.pitch, 1)}  Roll: {_fmt(a.head.roll, 1)}",
         WHITE),
        (f"Phone: {phone_text}", RED if phone_bad else (GREY if not a.phone.enabled else GREEN)),
        (f"Risk Level: {level.name}", LEVEL_COLORS[level]),
        (f"Safety Score: {a.safety.safety_score}", LEVEL_COLORS[level]),
    ]

    scale = cfg.font_scale
    line_h = int(34 * scale) + 4
    title_h = 30
    panel_w = 400
    panel_h = title_h + line_h * len(rows) + 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)   # translucent background

    cv2.putText(frame, "DRIVEGUARD", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    y = title_h + line_h - 6
    for text, color in rows:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += line_h


def _draw_footer(frame: np.ndarray, a: "FrameAnalysis", fps: float, muted: bool,
                 cfg: UIConfig) -> None:
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS {fps:4.1f}", (w - 100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    if a.safety.alert_message:
        text = f"WARNING: {a.safety.alert_message}"
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        cv2.putText(frame, text, (max(10, (w - size[0]) // 2), h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, RED, 2, cv2.LINE_AA)

    help_text = "q quit | c calibrate head pose | l landmarks | m mute" + (" [MUTED]" if muted else "")
    cv2.putText(frame, help_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
