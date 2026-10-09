# DriveGuard - Phase 1

Real-time, webcam-based driver monitoring: drowsiness, yawning, distraction, driver absence and phone use, combined into a risk score with spoken warnings.

> Phase 1 covers milestones 1-8. SQLite trip storage and the Streamlit dashboard/report (milestones 9-10) are **not** included yet.

## 1. Project overview
DriveGuard reads frames from a webcam, extracts 478 facial landmarks with MediaPipe, derives several geometric measurements (EAR, MAR, head angles), runs a YOLO model for phones, and feeds time-filtered results into a safety engine that produces a risk level, a score and (when needed) a spoken alert.

## 2. Features
- Live webcam view with a readable HUD
- Primary-driver selection (largest face) with 0 / 1 / many faces handled
- Drowsiness via Eye Aspect Ratio (EAR) + persistence timer
- Yawning via Mouth Aspect Ratio (MAR) + persistence timer
- Head pose (yaw / pitch / roll) via `solvePnP`, with distraction timer and a calibration key
- Driver-absence detection with a grace period
- YOLO (Ultralytics) phone detection with boxes and confidence
- Weighted, smoothed risk engine: SAFE / LOW / MEDIUM / HIGH + safety score
- Non-blocking audio alerts with cooldowns and per-event messages
- Every threshold lives in `config.py`

## 3. Architecture
```
DriveGuard/
├── main.py              camera loop, keys, window, audio hookup
├── pipeline.py          runs all detectors + safety engine for one frame
├── config.py            ALL thresholds/weights/paths (frozen dataclasses)
├── requirements.txt
├── detection/
│   ├── face_detector.py   MediaPipe Face Mesh, primary-face choice
│   ├── eye_detector.py    EAR + drowsiness timer
│   ├── mouth_detector.py  MAR + yawn timer
│   ├── head_pose.py       solvePnP, yaw/pitch/roll, distraction timer
│   ├── presence.py        driver-absent logic
│   └── phone_detector.py  YOLO phone detection
├── safety/safety_engine.py  weighted risk score + alert selection
├── alerts/audio_alert.py    threaded text-to-speech with cooldowns
├── ui/overlay.py            all OpenCV drawing
├── utils/geometry.py        EAR/MAR maths (pure functions)
├── utils/timing.py          PersistenceTimer (temporal filtering)
└── models/                  put yolov8n.pt here
```
Data flow per frame: `frame -> FaceDetector -> landmarks -> {Eye, Mouth, HeadPose, Presence}`, `frame -> PhoneDetector`, then all persistent flags -> `SafetyEngine -> FrameAnalysis -> overlay + audio`. Detectors never import each other; only `pipeline.py` wires them together.

## 4-6. Installation and virtual environment
Use **Python 3.9 - 3.12** (the pinned MediaPipe 0.10.14 has no wheels for 3.13+).

Windows (PowerShell):
```powershell
cd DriveGuard
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Linux / macOS:
```bash
cd DriveGuard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
(`ultralytics` installs PyTorch, so the first install is large.)

## 7. YOLO model setup
Download the pretrained COCO model (class `cell phone`) into `models/`:
```bash
# Linux / macOS
mkdir -p models && curl -L -o models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
# Windows PowerShell
mkdir models -Force; Invoke-WebRequest -Uri https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt -OutFile models\yolov8n.pt
```
Alternative: `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"` then move the downloaded file into `models/`.
Any COCO-trained Ultralytics model works (`--model path/to/model.pt`, or edit `PhoneConfig.model_path`). If the file is missing, DriveGuard prints these instructions and exits; use `--no-phone` to run without it.

## 8. Run
```bash
python main.py
```
Options: `--camera 1`, `--model path.pt`, `--no-phone`, `--no-audio`.

## 9. Keyboard controls
| Key | Action |
|---|---|
| `q` / `Esc` | Quit (camera and windows are released) |
| `c` | Calibrate head pose: current pose becomes "looking at the road" |
| `l` | Toggle landmark dots |
| `m` | Mute / unmute audio |

Press `c` once while looking straight ahead from your normal seated position - a dashboard camera rarely sits exactly in front of the driver.

## 10. Configuration
Open `config.py`. Each section is a dataclass: `EyeConfig.ear_threshold`, `MouthConfig.yawn_duration_s`, `HeadPoseConfig.yaw_max_deg`, `PresenceConfig.absence_grace_s`, `PhoneConfig.confidence`, `SafetyConfig.weights`, `AudioConfig.per_event_cooldown_s`, and so on. Durations are in **seconds**, so behaviour does not depend on FPS. Thresholds are starting points: tune them for your camera, lighting and face (people with narrow eyes or glasses may need a lower `ear_threshold`).

## 11. How EAR works
For six landmarks around an eye (p1 outer corner, p4 inner corner, p2/p3 upper lid, p6/p5 lower lid):

`EAR = ( |p2-p6| + |p3-p5| ) / ( 2 * |p1-p4| )`

An open eye gives roughly 0.25-0.35; when it closes the vertical distances collapse and EAR drops toward 0. Dividing by eye width makes it distance-independent. EAR of both eyes is averaged. EAR below `ear_threshold` means "closed", but **drowsy** is only declared after the eyes stay closed for `closed_duration_s` (1.0 s), so ordinary blinks (0.1-0.4 s) are ignored. The timer resets as soon as the eyes open.

## 12. How MAR works
Using the inner-lip landmarks (two corners, three upper/lower pairs):

`MAR = ( |u1-l1| + |u2-l2| + |u3-l3| ) / ( 3 * |left corner - right corner| )`

Closed mouth ~0-0.1, talking ~0.2-0.4, wide yawn above ~0.55. A yawn is declared only if MAR stays above `mar_threshold` for `yawn_duration_s` (1.5 s).

## 13. How head-pose detection works
Six landmarks (nose tip, chin, outer eye corners, mouth corners) are matched to a generic 3-D face model with `cv2.solvePnP`, using an approximate camera (focal length = image width, centre principal point, no distortion). This returns a rotation vector -> `cv2.Rodrigues` -> rotation matrix -> `cv2.RQDecomp3x3` -> pitch (up/down), yaw (left/right), roll (tilt). Angles are smoothed and reported relative to the calibrated neutral pose. If |yaw|, |pitch| or |roll| exceed their limits for `distraction_duration_s` (2.0 s), the driver is "distracted". The yellow line on the video shows where the face points.

## 14. How phone detection works
A pretrained YOLO COCO model is run on every Nth frame (`inference_every_n_frames`, default 3) filtered to the `cell phone` class (resolved by name, not hard-coded id). Boxes and confidence are drawn. A phone must be seen for `use_duration_s` (0.5 s) to count as "in use", and the status is held for `release_grace_s` (1.0 s) to bridge missed detections. Note: this detects a phone *visible in the frame*, not proof that it is being used.

## 15. Safety-score logic
Each time-filtered behaviour adds risk points (defaults): drowsy 60, absent 60, phone 45, distracted 40, yawning 20 (capped at 100). The total is smoothed (rises with a 0.5 s time constant, falls with 2 s) and mapped to levels: `<5` SAFE, `5-25` LOW, `25-50` MEDIUM, `>=50` HIGH. **Safety score = 100 - risk.** When the level reaches MEDIUM the highest-priority active behaviour (drowsy > phone > distracted > absent > yawning) is spoken, subject to cooldowns (3 s between any alerts, 10 s before the same one repeats). Examples: yawning alone = LOW (silent); phone alone = MEDIUM; drowsy alone = HIGH. Edit `SafetyConfig` to change it.

## 16. Known limitations
- Generic 3-D face model and guessed camera intrinsics make head angles approximate; use `c` to calibrate.
- EAR/MAR thresholds vary by person; glasses, sunglasses, masks and strong backlight reduce accuracy. Sunglasses will look like closed eyes.
- Low light / IR-less webcams degrade MediaPipe. A real vehicle needs an IR camera.
- Phone detection depends on YOLOv8n accuracy and camera view; a phone held below the frame or against the ear may be missed. Phone *presence*, not *usage*, is detected.
- Absence and drowsiness can overlap: a driver whose face leaves the frame is reported absent.
- Primary driver = largest face; a passenger leaning close could be chosen.
- CPU-only YOLO lowers FPS; increase `inference_every_n_frames` or use a GPU.
- Audio uses OS text-to-speech; Linux needs `espeak-ng`/`spd-say` (or falls back to beeps).
- Not a certified safety device; for education/prototyping only.

## Troubleshooting
**Webcam:** "Cannot open webcam" -> close Zoom/Teams/browser tabs using it, try `--camera 1`, grant camera permission (Windows Settings > Privacy > Camera; macOS System Settings > Privacy > Camera for your terminal/IDE). Black or very slow frames on Windows -> the DirectShow backend is already used; try a different USB port.

**MediaPipe:** `module 'mediapipe' has no attribute 'solutions'` or install failure -> use Python 3.9-3.12 and `pip install mediapipe==0.10.14` with `numpy<2`. If you see `cv2` oddities after installing both `opencv-python` and `opencv-contrib-python`: `pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless` then `pip install "opencv-contrib-python>=4.8,<4.11"`. No landmarks drawn -> improve lighting, face the camera, remove the mask.

**YOLO:** "model file not found" -> follow section 7. Slow FPS -> raise `inference_every_n_frames`, lower `image_size` (e.g. 416), or set `device="cuda:0"` if you have an NVIDIA GPU with CUDA PyTorch. No phone detected -> lower `confidence` to ~0.3, hold the phone clearly in view.

**Audio:** No sound on Linux -> `sudo apt install espeak-ng` (or `speech-dispatcher`). Windows `pyttsx3` errors -> `pip install --force-reinstall pyttsx3 pywin32`. macOS uses `say`. You will hear beeps if no speech engine exists. Use `m` to mute, `--no-audio` to disable. Alerts repeat no more often than the cooldowns in `AudioConfig`.
