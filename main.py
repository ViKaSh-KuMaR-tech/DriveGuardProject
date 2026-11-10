"""DriveGuard Phase 1 - real-time driver monitoring. Run: python main.py"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import replace

import cv2
import numpy as np

from alerts.audio_alert import AudioAlert
from config import CONFIG, Config
from detection.phone_detector import ModelNotFoundError
from pipeline import DriveGuardPipeline
from ui.overlay import draw_overlay
from storage.event_store import EventStore

log = logging.getLogger("driveguard")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DriveGuard - AI driver monitoring (Phase 1)")
    p.add_argument("--camera", type=int, help="webcam index (default from config.py)")
    p.add_argument("--model", type=str, help="path to a YOLO .pt model (default from config.py)")
    p.add_argument("--no-phone", action="store_true", help="disable YOLO phone detection")
    p.add_argument("--no-audio", action="store_true", help="disable audio alerts")
    p.add_argument("--test-audio", action="store_true", help="play a sample alert and exit (no camera required)")
    return p.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.camera is not None:
        cfg = replace(cfg, camera=replace(cfg.camera, index=args.camera))
    if args.model:
        cfg = replace(cfg, phone=replace(cfg.phone, model_path=args.model))
    if args.no_audio:
        cfg = replace(cfg, audio=replace(cfg.audio, enabled=False))
    return cfg


def open_camera(cfg: Config) -> cv2.VideoCapture:
    index = cfg.camera.index
    # DirectShow starts much faster than the default backend on Windows.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot open webcam index {index}. Check that it is connected, not used by "
            "another app, and that camera permission is granted. Try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    return cap


def _show_last_record(record) -> None:
    """Display the latest SQLite event in a small OpenCV window."""
    panel = np.zeros((220, 560, 3), dtype=np.uint8)
    cv2.putText(panel, "LATEST SAVED EVENT", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 220, 220), 2, cv2.LINE_AA)
    if record is None:
        lines = ["No event records have been saved yet."]
    else:
        lines = [
            f"Time (UTC): {record['occurred_at']}",
            f"Event: {record['event_type']}",
            f"Risk level: {record['risk_level']}",
            f"Safety score: {record['safety_score']}",
            f"Message: {(record['message'] or '')[:48]}",
        ]
    for index, line in enumerate(lines):
        cv2.putText(panel, line, (20, 78 + index * 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.namedWindow("Last Trip Record", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Last Trip Record", panel)


def run(cfg: Config, enable_phone: bool) -> int:
    pipeline = DriveGuardPipeline(cfg, enable_phone=enable_phone)   # may raise (e.g. missing model)
    audio = AudioAlert(cfg.audio)
    event_store = EventStore()
    previous_events = set()
    cap = None
    show_landmarks = cfg.ui.show_landmarks
    fps = 0.0
    last_t = time.monotonic()
    failures = 0
    mouse_state = {"clicked": False, "width": cfg.camera.width}

    def on_mouse(event, x, y, flags, state):
        if (event == cv2.EVENT_LBUTTONDOWN
                and x >= state["width"] - 155 and x <= state["width"] - 8
                and 29 <= y <= 59):
            state["clicked"] = True

    try:
        cap = open_camera(cfg)
        cv2.namedWindow(cfg.ui.window_name)
        cv2.setMouseCallback(cfg.ui.window_name, on_mouse, mouse_state)
        log.info("Running. Press 'q' in the video window to quit.")
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= cfg.camera.max_read_failures:
                    log.error("Webcam stopped delivering frames.")
                    return 1
                time.sleep(0.03)
                continue
            failures = 0

            mouse_state["width"] = frame.shape[1]
            now = time.monotonic()
            analysis = pipeline.process(frame, now)

            current_events = set(analysis.safety.active_events)
            for event in current_events - previous_events:
                event_store.record_event(
                    event_type=event,
                    risk_level=analysis.safety.level.name,
                    safety_score=analysis.safety.safety_score,
                    message=cfg.safety.messages.get(event),
                )
            previous_events = current_events

            if analysis.safety.alert_event and analysis.safety.alert_message:
                audio.request(analysis.safety.alert_event, analysis.safety.alert_message, now)

            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            draw_overlay(frame, analysis, fps, show_landmarks, audio.muted, cfg.ui)
            cv2.imshow(cfg.ui.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if mouse_state["clicked"]:
                mouse_state["clicked"] = False
                _show_last_record(event_store.get_last_event())
            if key in (ord("q"), 27):               # q or Esc
                break
            elif key == ord("c"):
                log.info("Head pose calibrated." if pipeline.calibrate_head_pose()
                         else "Calibration needs a visible face.")
            elif key == ord("l"):
                show_landmarks = not show_landmarks
            elif key == ord("m"):
                log.info("Audio muted." if audio.toggle_mute() else "Audio unmuted.")
        return 0
    finally:
        if cap is not None:
            cap.release()
        audio.close()
        event_store.close()
        pipeline.close()
        cv2.destroyAllWindows()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    cfg = apply_overrides(CONFIG, args)
    if args.test_audio:
        audio = AudioAlert(cfg.audio)
        try:
            if not cfg.audio.enabled:
                log.error("Audio is disabled. Remove --no-audio and set AudioConfig.enabled = True.")
                return 2
            if not audio.test():
                log.error("Audio test did not complete. Check Windows volume and the audio backend log.")
                return 1
            log.info("Audio test finished. If you heard the sample, sound is working.")
            return 0
        finally:
            audio.close()
    try:
        return run(cfg, enable_phone=not args.no_phone)
    except ModelNotFoundError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
