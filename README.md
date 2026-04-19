# 🦾 Post-Surgery Rehab Assistant v3

A real-time AI-powered physical rehabilitation coach that uses pose estimation to guide users through post-surgery exercises, track their range of motion, and deliver live voice feedback — all from a standard webcam.

---

## 📌 Overview

Post-surgery recovery often suffers from poor adherence to exercise routines and lack of real-time feedback at home. This project addresses that by building a computer-vision-based rehab assistant that detects body pose in real time, counts reps, monitors form, and tracks recovery progress over sessions.

---

## ✨ Features

| Feature | Description |
|---|---|
| **5 Exercise Modes** | Shoulder Flexion, Bicep Curl, Shoulder Press, Torso Twist, Cross-Body Reach |
| **Sets & Reps System** | Configurable sets × reps per exercise with automatic rest countdown between sets |
| **Real-Time Angle Tracking** | Joint angles computed using Law of Cosines with 5-frame rolling average to reduce jitter |
| **Voice Feedback** | Windows SAPI-powered voice cues for rep counts, form errors, and set completion |
| **Session History** | Every session appended to `rehab_history.json` for long-term ROM tracking |
| **Live HUD** | Displays set/rep progress, joint angle, FPS, session timer, and form status |
| **Progress Bar** | Visual ROM progress bar showing angle relative to target range |
| **Pause / Resume** | Press `SPACE` to freeze processing at any time |
| **Bilateral Detection** | Automatically uses the more active (visible) side of the body |
| **Post-Session Report** | Generates a `rehab_report.png` chart summarizing exercise performance |
| **3-2-1 Countdown** | Animated countdown screen before the first rep begins |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **MediaPipe 0.10** — Pose Landmarker (Live Stream mode)
- **OpenCV** — Webcam capture, frame rendering, skeleton overlay, HUD
- **NumPy** — Numerical processing
- **Matplotlib** — Post-session report generation
- **Windows SAPI (via PowerShell)** — Voice synthesis (no pyttsx3 / COM issues)

---

## 🗂️ Project Structure

```
post-surgery-rehab-assistant/
│
├── rehab_assistant_v3.py       # Main application
├── pose_landmarker_lite.task   # MediaPipe model (download separately)
├── rehab_history.json          # Auto-generated session history
├── rehab_report.png            # Auto-generated post-session chart
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/swagat147/post-surgery-rehab-assistant.git
cd post-surgery-rehab-assistant
```

### 2. Install dependencies
```bash
pip install opencv-python mediapipe numpy matplotlib
```

### 3. Download the MediaPipe model
```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task', 'pose_landmarker_lite.task')"
```

### 4. Run the assistant
```bash
python rehab_assistant_v3.py
```

> ⚠️ Requires a webcam and Windows OS (voice feedback uses Windows SAPI).

---

## 🎮 Controls

| Key | Action |
|---|---|
| `F` | Shoulder Flexion |
| `B` | Bicep Curl |
| `P` | Shoulder Press |
| `T` | Torso Twist |
| `C` | Cross-Body Reach |
| `SPACE` | Pause / Resume |
| `Q` | Quit & generate session report |

---

## 📊 How It Works

1. Webcam feed is captured and flipped horizontally for mirror view.
2. Each frame is passed asynchronously to MediaPipe's `PoseLandmarker` in Live Stream mode.
3. 3D landmark coordinates are extracted and joint angles are computed using the **Law of Cosines**.
4. A rolling average smoother reduces noise across 5 frames.
5. Rep counting uses a **state machine** (DOWN → UP → DOWN) with configurable angle thresholds and a cooldown timer to prevent double-counting.
6. A `SetManager` handles transitions between COUNTDOWN → ACTIVE → REST → DONE states.
7. Voice cues are dispatched on a background thread via PowerShell/SAPI to keep the main loop non-blocking.
8. On quit, session data is saved to JSON and a summary chart is generated via Matplotlib.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
