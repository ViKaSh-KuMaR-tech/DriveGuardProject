"""YOLO (Ultralytics) mobile-phone detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from config import PhoneConfig
from utils.timing import PersistenceTimer


class ModelNotFoundError(FileNotFoundError):
    """Raised when the YOLO weights file does not exist."""


@dataclass(frozen=True)
class PhoneBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass(frozen=True)
class PhoneState:
    enabled: bool = True
    detected: bool = False                 # a phone was seen in the latest inference
    in_use: bool = False                   # persistent: seen long enough => unsafe
    boxes: Tuple[PhoneBox, ...] = ()

    @staticmethod
    def disabled() -> "PhoneState":
        return PhoneState(enabled=False)


def _missing_model_message(path: Path) -> str:
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    return (
        f"YOLO model file not found: {path}\n\n"
        "DriveGuard needs a pretrained COCO model (it contains the 'cell phone' class).\n"
        "Get one of these ways, then re-run:\n"
        f"  Linux/macOS : mkdir -p models && curl -L -o models/yolov8n.pt {url}\n"
        f"  Windows PS  : mkdir models -Force; Invoke-WebRequest -Uri {url} -OutFile models\\yolov8n.pt\n"
        "  Any OS      : python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"\n"
        "                (downloads yolov8n.pt into the current folder - move it into models/)\n"
        "Or point to any COCO-trained .pt file with:  python main.py --model path/to/model.pt\n"
        "Or run without phone detection:             python main.py --no-phone"
    )


class PhoneDetector:
    def __init__(self, cfg: PhoneConfig) -> None:
        self._cfg = cfg
        path = Path(cfg.model_path)
        if not path.is_file():
            raise ModelNotFoundError(_missing_model_message(path))

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

        try:
            self._model = YOLO(str(path))
        except Exception as exc:  # corrupt file, wrong format, incompatible version ...
            raise RuntimeError(
                f"Could not load YOLO model '{path}': {exc}\n"
                "Make sure it is a valid Ultralytics .pt file (e.g. yolov8n.pt).") from exc

        # Resolve class ids by NAME so any COCO-trained model works (cell phone = 67 in COCO).
        wanted = {n.lower() for n in cfg.class_names}
        self._class_ids = [i for i, name in self._model.names.items() if name.lower() in wanted]
        if not self._class_ids:
            raise RuntimeError(
                f"Model '{path.name}' has no class named {sorted(wanted)}. "
                f"Available classes: {list(self._model.names.values())[:20]}...")

        self._timer = PersistenceTimer(cfg.use_duration_s, cfg.release_grace_s)
        self._frame_counter = 0
        self._last_boxes: Tuple[PhoneBox, ...] = ()

    def update(self, frame_bgr: np.ndarray, now: float) -> PhoneState:
        """Run YOLO every Nth frame; in between, reuse the latest boxes."""
        if self._frame_counter % max(1, self._cfg.inference_every_n_frames) == 0:
            self._last_boxes = self._infer(frame_bgr)
        self._frame_counter += 1

        detected = len(self._last_boxes) > 0
        in_use = self._timer.update(detected, now)
        return PhoneState(True, detected, in_use, self._last_boxes)

    def _infer(self, frame_bgr: np.ndarray) -> Tuple[PhoneBox, ...]:
        kwargs = dict(conf=self._cfg.confidence, classes=self._class_ids,
                      imgsz=self._cfg.image_size, verbose=False)
        if self._cfg.device is not None:
            kwargs["device"] = self._cfg.device
        result = self._model.predict(frame_bgr, **kwargs)[0]   # Ultralytics accepts BGR numpy frames

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return ()
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        return tuple(PhoneBox(int(x1), int(y1), int(x2), int(y2), float(c))
                     for (x1, y1, x2, y2), c in zip(xyxy, conf))
