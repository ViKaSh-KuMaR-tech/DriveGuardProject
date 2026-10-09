# 🚗 DriveGuard — AI-Powered Driver Monitoring & Road Safety System

**Making every journey safer through real-time computer vision and intelligent driver monitoring.**

DriveGuard is an AI-powered driver monitoring system designed to detect signs of driver fatigue, distraction, and potentially unsafe driving behavior using computer vision. It analyzes visual cues such as eye closure, head position, yawning, and mobile phone usage to help identify situations that may compromise road safety.

By combining real-time video processing, modular detection components, and audio alerts, DriveGuard aims to transform an ordinary camera into an intelligent driver-safety assistant.

> **Vision:** Detect risks early. Alert intelligently. Drive safely.

---

## ✨ Key Features

### 👁️ Drowsiness Detection

* Monitors eye-related indicators to identify possible signs of fatigue.
* Analyzes facial features and eye activity.
* Helps identify prolonged eye closure and potential microsleep events.

### 📱 Mobile Phone Detection

* Includes a dedicated phone-detection module.
* Helps identify potentially distracting mobile phone usage while driving.
* Supports integration into the overall driver-safety pipeline.

### 🥱 Yawning & Fatigue Indicators

* Analyzes mouth movement and opening patterns.
* Helps identify yawning as a potential indicator of fatigue.
* Combines multiple visual indicators for a broader assessment of driver alertness.

### 🧠 Head Pose & Face Analysis

* Tracks facial orientation and head position.
* Supports detection of possible distraction caused by looking away from the road.
* Uses modular face and head-pose analysis components.

### 🔊 Intelligent Audio Alerts

* Provides audio-alert functionality for detected safety conditions.
* Supports non-blocking alert processing so that audio playback does not unnecessarily interrupt the main video-processing loop.
* Includes operating-system-based speech options and fallback alerts.

### 🛡️ Centralized Safety Engine

* Brings safety-related detection logic into a dedicated module.
* Helps organize monitoring rules and safety responses.
* Provides a foundation for extending the system with additional risk indicators.

### 🖥️ Real-Time Visual Monitoring

* Includes video-processing and overlay components.
* Organizes detection results for visual presentation.
* Uses a modular architecture to support future interface improvements.

---

## 🏗️ System Architecture

DriveGuard is organized into separate modules to improve maintainability, debugging, and future development.

```text
DriveGuard/
│
├── main.py                 # Application entry point
├── config.py               # Configuration
├── pipeline.py             # Detection and processing pipeline
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── alerts/                 # Audio and alert handling
├── detection/              # Driver and object detection
├── models/                 # Model assets
├── safety/                 # Safety monitoring logic
├── ui/                     # Visual overlays and presentation
└── utils/                  # Supporting utilities
```

**Processing flow**

```text
Camera / Video Input
        ↓
Frame Acquisition
        ↓
Face & Visual Feature Analysis
        ↓
Drowsiness / Head Pose / Phone Analysis
        ↓
Safety Evaluation
        ↓
Visual Feedback + Audio Alerts
```

*The exact execution flow and available detectors depend on the configured modules and models.*

---

## 🛠️ Technology Stack

| Technology                    | Purpose                                      |
| ----------------------------- | -------------------------------------------- |
| Python                        | Core application logic                       |
| OpenCV                        | Video capture and image processing           |
| Computer Vision               | Visual analysis of driver behavior           |
| Detection Models              | Object and phone detection, where configured |
| Threading                     | Background processing for supported tasks    |
| Operating-System Speech Tools | Audio warnings and spoken alerts             |

The exact dependencies and model requirements are defined by the implementation and `requirements.txt`.

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.10 or a compatible version for the project's dependencies
* Git
* A webcam or supported video input
* The required detection models and model weights, if applicable

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/DriveGuardProject.git
cd DriveGuardProject
```

Replace `YOUR_USERNAME` with your GitHub username.

### 3. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some detection modules may require additional model weights or system-specific dependencies.

### 5. Run DriveGuard

```bash
python main.py
```

Ensure the camera is connected and any required models are available. If startup fails, review the error message and verify the installed dependencies and model paths.

---

## 📂 Core Modules

| Module        | Responsibility                        |
| ------------- | ------------------------------------- |
| `main.py`     | Starts the application                |
| `config.py`   | Stores configuration settings         |
| `pipeline.py` | Coordinates processing components     |
| `alerts/`     | Audio and alert-related functionality |
| `detection/`  | Visual detection components           |
| `models/`     | Model files and related assets        |
| `safety/`     | Safety evaluation and rules           |
| `ui/`         | Visual feedback and overlays          |
| `utils/`      | Shared helper functions               |

---

## 🎯 Project Goals

* Detect visual indicators associated with driver fatigue.
* Identify potential distractions through computer vision.
* Provide timely, understandable warnings.
* Keep processing modules independent and maintainable.
* Build a foundation for more comprehensive driver monitoring.

---

## 🔮 Future Roadmap

Potential future improvements include:

* [ ] Driver risk scoring based on multiple indicators.
* [ ] Calibration for different drivers and lighting conditions.
* [ ] Improved low-light and nighttime performance.
* [ ] Detection confidence visualization.
* [ ] Session summaries and fatigue-event logging.
* [ ] Configurable alert thresholds.
* [ ] Performance optimization for lower-end hardware.
* [ ] Expanded testing across different camera positions and driving conditions.

---

## ⚠️ Limitations & Safety Disclaimer

DriveGuard is an experimental driver-monitoring project and is **not a certified automotive safety system**.

Computer-vision results can be affected by lighting, camera placement, occlusion, eyewear, and model accuracy. A detected behavior does not necessarily prove that a driver is fatigued or distracted.

Do not rely on this software as a replacement for attentive driving, adequate rest, or certified vehicle safety equipment. Always follow local traffic laws and keep your attention on the road.

---

## 👨‍💻 Contributing

Contributions, bug reports, testing, and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and test them.
4. Submit a pull request describing the improvement.

---

## 📜 License

No license has been specified yet. Add a `LICENSE` file before granting others explicit permission to reuse, modify, or redistribute this project.

---

## ⭐ Support the Project

If you find DriveGuard interesting, consider starring the repository and sharing ideas for improving driver safety through computer vision.

**DriveGuard — See the risk. Sound the warning. Make the journey safer.**
