"""DriveGuard Phase 1 - real-time driver monitoring. Run: python main.py"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import replace

import cv2

from alerts.audio_alert import AudioAlert
from config import CONFIG, Config
from detection.phone_detector import ModelNotFoundError
from pipeline import DriveGuardPipeline
from ui.overlay import draw_overlay

log = logging.getLogger("driveguard")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DriveGuard - AI driver monitoring (Phase 1)")
    p.add_argument("--camera", type=int, help="webcam index (default from config.py)")
    p.add_argument("--model", type=str, help="path to a YOLO .pt model (default from config.py)")
    p.add_argument("--no-phone", action="store_true", help="disable YOLO phone detection")
    p.add_argument("--no-audio", action="store_true", help="disable audio alerts")
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


def run(cfg: Config, enable_phone: bool) -> int:
    pipeline = DriveGuardPipeline(cfg, enable_phone=enable_phone)   # may raise (e.g. missing model)
    audio = AudioAlert(cfg.audio)
    cap = None
    show_landmarks = cfg.ui.show_landmarks
    fps = 0.0
    last_t = time.monotonic()
    failures = 0
    try:
        cap = open_camera(cfg)
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

            now = time.monotonic()
            analysis = pipeline.process(frame, now)

            if analysis.safety.alert_event and analysis.safety.alert_message:
                audio.request(analysis.safety.alert_event, analysis.safety.alert_message, now)

            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            draw_overlay(frame, analysis, fps, show_landmarks, audio.muted, cfg.ui)
            cv2.imshow(cfg.ui.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
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
        pipeline.close()
        cv2.destroyAllWindows()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    cfg = apply_overrides(CONFIG, args)
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
