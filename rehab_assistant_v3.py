"""
Post-Surgery Physical Rehab Assistant  v3
==========================================
Real-time pose-estimation rehab coach using OpenCV, MediaPipe 0.10+,
and Windows SAPI for voice feedback.

What's new in v3 (over the original)
--------------------------------------
  SETS SYSTEM      – each exercise has a configurable target (sets × reps).
                     After finishing a set the system enters REST mode with a
                     live countdown.  All sets done → celebration screen.
  PAUSE / RESUME   – press SPACE to freeze processing at any time.
  SESSION HISTORY  – every session is appended to rehab_history.json so you
                     can track ROM improvement over weeks.
  COUNTDOWN        – 3-2-1 screen before the first rep starts.
  ANGLE SMOOTHING  – 5-frame rolling average kills measurement jitter.
  VOICE COOLDOWN   – form-error cues throttled to once per 3 s.
  FPS + TIMER      – live frame-rate and session duration in the HUD.
  REP TARGET HUD   – always shows "Set 2/3 · Rep 4/10" progress.
  BILATERAL        – all exercises automatically use the more active side.
  CONFIG CLASS     – every magic number lives in one place at the top.

Controls
--------
    F  Shoulder Flexion    T  Torso Twist
    B  Bicep Curl          C  Cross-Body Reach
    P  Shoulder Press
    SPACE  Pause / Resume
    Q  Quit & generate report
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import cv2
import json
import math
import time
import queue
import datetime
import subprocess
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, asdict

import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── MediaPipe 0.10 aliases ────────────────────────────────────────────────────
BaseOptions       = mp.tasks.BaseOptions
PoseLandmarker    = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOpt = mp.tasks.vision.PoseLandmarkerOptions
RunningMode       = mp.tasks.vision.RunningMode
PoseLandmark      = mp.tasks.vision.PoseLandmark

LM_NOSE             = PoseLandmark.NOSE
LM_RIGHT_EAR        = PoseLandmark.RIGHT_EAR
LM_LEFT_EAR         = PoseLandmark.LEFT_EAR
LM_RIGHT_SHOULDER   = PoseLandmark.RIGHT_SHOULDER
LM_RIGHT_ELBOW      = PoseLandmark.RIGHT_ELBOW
LM_RIGHT_WRIST      = PoseLandmark.RIGHT_WRIST
LM_RIGHT_HIP        = PoseLandmark.RIGHT_HIP
LM_LEFT_SHOULDER    = PoseLandmark.LEFT_SHOULDER
LM_LEFT_ELBOW       = PoseLandmark.LEFT_ELBOW
LM_LEFT_WRIST       = PoseLandmark.LEFT_WRIST
LM_LEFT_HIP         = PoseLandmark.LEFT_HIP

UPPER_BODY_CONNECTIONS = [
    (LM_RIGHT_EAR,      LM_RIGHT_SHOULDER),
    (LM_LEFT_EAR,       LM_LEFT_SHOULDER),
    (LM_RIGHT_SHOULDER, LM_RIGHT_ELBOW),
    (LM_RIGHT_ELBOW,    LM_RIGHT_WRIST),
    (LM_RIGHT_SHOULDER, LM_RIGHT_HIP),
    (LM_LEFT_SHOULDER,  LM_LEFT_ELBOW),
    (LM_LEFT_ELBOW,     LM_LEFT_WRIST),
    (LM_LEFT_SHOULDER,  LM_LEFT_HIP),
    (LM_RIGHT_SHOULDER, LM_LEFT_SHOULDER),
    (LM_RIGHT_HIP,      LM_LEFT_HIP),
]

UPPER_BODY_LANDMARKS = (
    LM_NOSE, LM_RIGHT_EAR, LM_LEFT_EAR,
    LM_RIGHT_SHOULDER, LM_RIGHT_ELBOW, LM_RIGHT_WRIST, LM_RIGHT_HIP,
    LM_LEFT_SHOULDER,  LM_LEFT_ELBOW,  LM_LEFT_WRIST,  LM_LEFT_HIP,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Global Config  – change numbers here, nowhere else
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Model
    model_path:   str   = "pose_landmarker_lite.task"
    detect_conf:  float = 0.60
    track_conf:   float = 0.60

    # Camera
    cam_index:  int = 0
    cam_w:      int = 1280
    cam_h:      int = 720
    cam_fps:    int = 30

    # Reps / sets per exercise (can be overridden per-exercise)
    default_sets:    int = 3
    default_reps:    int = 10
    rest_between_s:  int = 30     # seconds to rest between sets

    # Angle processing
    smooth_window:   int   = 5    # frames in rolling average
    rep_buffer_deg:  float = 10.0 # ± tolerance at thresholds
    rep_cooldown_s:  float = 0.4  # min seconds between rep credits

    # Voice
    voice_queue_max:     int   = 2
    form_cooldown_s:     float = 3.0   # min seconds between form warnings

    # Countdown
    countdown_secs: int = 3

    # Files
    history_file:  str = "rehab_history.json"
    report_file:   str = "rehab_report.png"

    # UI
    hud_alpha:  float = 0.70
    font        = cv2.FONT_HERSHEY_SIMPLEX


CFG = Config()


# ─────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def calc_angle(a: tuple, b: tuple, c: tuple) -> float:
    """Angle at vertex b (a–b–c) via Law of Cosines.  Returns 0–180 degrees."""
    ab, bc, ac = math.dist(a, b), math.dist(b, c), math.dist(a, c)
    denom = 2 * ab * bc
    if denom == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0,
        (ab**2 + bc**2 - ac**2) / denom
    ))))


def px(lms, idx, w: int, h: int) -> tuple:
    """NormalizedLandmark → integer pixel (x, y)."""
    pt = lms[idx]
    return (int(pt.x * w), int(pt.y * h))


def visible(lms, *ids, thr: float = 0.45) -> bool:
    return all(lms[i].visibility >= thr for i in ids)


class AngleSmoother:
    """Rolling average to reduce pose-estimation jitter."""
    def __init__(self):
        self._buf = deque(maxlen=CFG.smooth_window)

    def push(self, v: float) -> float:
        self._buf.append(v)
        return sum(self._buf) / len(self._buf)

    def reset(self):
        self._buf.clear()


def fmt_time(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────────────────────
#  Voice  (Windows SAPI via PowerShell – no pyttsx3, no COM issues)
# ─────────────────────────────────────────────────────────────────────────────
class Voice:
    def __init__(self):
        self._q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            txt = self._q.get()
            if txt is None:
                return
            safe = txt.replace('"', '').replace("'", "")
            ps = (
                'Add-Type -AssemblyName System.Speech; '
                '$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                f'$s.Speak("{safe}")'
            )
            try:
                subprocess.run(
                    ["powershell", "-Command", ps],
                    capture_output=True, timeout=12,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            except Exception:
                pass

    def say(self, txt: str):
        """Non-blocking enqueue (drops if already busy)."""
        if self._q.qsize() < CFG.voice_queue_max:
            self._q.put(txt)

    def say_now(self, txt: str):
        """Flush queue then enqueue (used for rep announcements)."""
        while not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: break
        self._q.put(txt)

    def stop(self):
        self._q.put(None)


# ─────────────────────────────────────────────────────────────────────────────
#  Set / Rep state machine
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SetRecord:
    peak_angles: list = field(default_factory=list)
    timestamp:   str  = field(default_factory=lambda: datetime.datetime.now().isoformat())


class SetManager:
    """
    Tracks sets and reps for one exercise.

    States
    ------
    COUNTDOWN  – waiting for initial 3-2-1 before first rep
    ACTIVE     – user is exercising
    REST       – between sets, countdown to next
    DONE       – all sets complete for this exercise
    """

    COUNTDOWN = "COUNTDOWN"
    ACTIVE    = "ACTIVE"
    REST      = "REST"
    DONE      = "DONE"

    _PRAISE = [
        "Excellent!", "Amazing progress!",
        "You're doing great!", "Keep it up!",
        "Fantastic effort!", "Superb!",
    ]

    def __init__(self, target_sets: int, target_reps: int):
        self.target_sets = target_sets
        self.target_reps = target_reps
        self.current_set  = 1       # 1-indexed
        self.current_reps = 0
        self.state        = self.COUNTDOWN
        self.sets: list[SetRecord] = []
        self._state_start = time.monotonic()

    # ── convenience ────────────────────────────────────────────────────────

    @property
    def all_done(self) -> bool:
        return self.state == self.DONE

    @property
    def total_reps(self) -> int:
        return sum(len(r.peak_angles) for r in self.sets) + self.current_reps

    def countdown_remaining(self) -> float:
        if self.state not in (self.COUNTDOWN, self.REST):
            return 0.0
        target = (CFG.countdown_secs if self.state == self.COUNTDOWN
                  else CFG.rest_between_s)
        return max(0.0, target - (time.monotonic() - self._state_start))

    def set_label(self) -> str:
        return f"Set {self.current_set}/{self.target_sets}"

    def rep_label(self) -> str:
        return f"Rep {self.current_reps}/{self.target_reps}"

    # ── per-frame tick ──────────────────────────────────────────────────────

    def tick(self) -> bool:
        """
        Call every frame.  Returns True on the frame COUNTDOWN→ACTIVE
        or REST→ACTIVE transitions so the exercise can reset its angle state.
        """
        if self.state == self.COUNTDOWN and self.countdown_remaining() <= 0:
            self.state = self.ACTIVE
            self._state_start = time.monotonic()
            return True

        if self.state == self.REST and self.countdown_remaining() <= 0:
            self.state = self.ACTIVE
            self.current_reps = 0
            self._state_start = time.monotonic()
            self.sets.append(SetRecord())
            return True

        return False

    def add_rep(self, peak_angle: float, voice: Voice):
        """Credit one rep and handle set completion."""
        if self.state != self.ACTIVE:
            return

        self.current_reps += 1
        if self.sets:
            self.sets[-1].peak_angles.append(peak_angle)

        praise_idx = (self.current_reps // 5 - 1) % len(self._PRAISE)

        if self.current_reps >= self.target_reps:
            # ── Set complete ────────────────────────────────────────────────
            if self.current_set >= self.target_sets:
                self.state = self.DONE
                voice.say_now(
                    f"Workout complete! "
                    f"{self.target_sets} sets of {self.target_reps} reps. "
                    f"Outstanding work!"
                )
            else:
                self.current_set += 1
                self.state = self.REST
                self._state_start = time.monotonic()
                voice.say_now(
                    f"Set {self.current_set - 1} complete! "
                    f"Rest for {CFG.rest_between_s} seconds."
                )
        else:
            if self.current_reps % 5 == 0:
                voice.say_now(
                    f"Rep {self.current_reps}. {self._PRAISE[praise_idx]}"
                )
            else:
                voice.say_now(f"Rep {self.current_reps}")

    def start_first_set(self):
        """Called once when exercise is first selected."""
        self.sets = [SetRecord()]

    def to_dict(self) -> dict:
        return {
            "target_sets": self.target_sets,
            "target_reps": self.target_reps,
            "sets":        [asdict(s) for s in self.sets],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Exercise abstract base
# ─────────────────────────────────────────────────────────────────────────────
class Exercise(ABC):
    """
    All rep-counting and form logic.  Subclasses only override:
      name, target_angle, start_angle, _get_landmarks(), check_form().
    """

    def __init__(self, sets: int = CFG.default_sets, reps: int = CFG.default_reps):
        self._smoother       = AngleSmoother()
        self._hit_target     = False
        self._peak           = 0.0
        self._last_rep_t     = 0.0
        self._last_form_t    = 0.0
        self.form_ok         = True
        self.form_msg        = ""
        self.set_mgr         = SetManager(sets, reps)

    # ── abstract interface ──────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def target_angle(self) -> float: ...

    @property
    @abstractmethod
    def start_angle(self) -> float: ...

    @abstractmethod
    def _get_pts(self, lms, w: int, h: int): ...

    @abstractmethod
    def check_form(self, lms, w: int, h: int) -> tuple[bool, str]: ...

    # ── threshold helpers ───────────────────────────────────────────────────

    @property
    def _decreasing(self) -> bool:
        return self.target_angle < self.start_angle

    def _past_target(self, a: float) -> bool:
        b = CFG.rep_buffer_deg
        return a <= self.target_angle + b if self._decreasing \
               else a >= self.target_angle - b

    def _at_start(self, a: float) -> bool:
        b = CFG.rep_buffer_deg
        return a >= self.start_angle - b if self._decreasing \
               else a <= self.start_angle + b

    # ── main update ─────────────────────────────────────────────────────────

    def process_frame(self, lms, w: int, h: int, voice: Voice) -> float:
        """
        Run one video frame.  Returns smoothed joint angle (degrees).
        Returns 0.0 when landmarks are not visible.
        """
        # Handle countdown / rest transitions
        if self.set_mgr.tick():
            self._reset_rep()
            if self.set_mgr.state == SetManager.ACTIVE:
                voice.say("Go!")

        if self.set_mgr.state != SetManager.ACTIVE:
            return 0.0

        pts = self._get_pts(lms, w, h)
        if pts is None:
            return 0.0

        angle = self._smoother.push(calc_angle(*pts))
        self.form_ok, self.form_msg = self.check_form(lms, w, h)

        # Track peak during this rep
        if self._decreasing:
            self._peak = min(self._peak or angle, angle)
        else:
            self._peak = max(self._peak, angle)

        now = time.monotonic()

        # ── Step 1: reached target? ─────────────────────────────────────────
        if not self._hit_target and self._past_target(angle):
            self._hit_target = True
            voice.say("Good — now return" if self.form_ok else self.form_msg)
            if not self.form_ok:
                self._last_form_t = now

        # ── Step 2: returned to start with valid form? ──────────────────────
        elif (self._hit_target
              and self._at_start(angle)
              and now - self._last_rep_t >= CFG.rep_cooldown_s):
            if self.form_ok:
                self.set_mgr.add_rep(self._peak, voice)
                self._last_rep_t = now
            self._reset_rep()

        # ── Throttled form warning while moving ─────────────────────────────
        elif (not self.form_ok
              and now - self._last_form_t >= CFG.form_cooldown_s):
            voice.say(self.form_msg)
            self._last_form_t = now

        return angle

    def _reset_rep(self):
        self._hit_target = False
        self._peak       = 0.0
        self._smoother.reset()

    def progress(self, angle: float) -> float:
        span = abs(self.target_angle - self.start_angle)
        return max(0.0, min(1.0, abs(angle - self.start_angle) / span)) if span else 0.0

    def session_summary(self) -> dict:
        return {"exercise": self.name, **self.set_mgr.to_dict()}


# ─────────────────────────────────────────────────────────────────────────────
#  Exercise subclasses
# ─────────────────────────────────────────────────────────────────────────────

class ShoulderFlexion(Exercise):
    """Forward arm raise.  Tracks both sides, uses more active."""
    name         = "SHOULDER FLEXION"
    target_angle = 150.0
    start_angle  =  20.0

    def _get_pts(self, lms, w, h):
        r = (px(lms,LM_RIGHT_HIP,w,h), px(lms,LM_RIGHT_SHOULDER,w,h), px(lms,LM_RIGHT_WRIST,w,h))
        l = (px(lms,LM_LEFT_HIP, w,h), px(lms,LM_LEFT_SHOULDER, w,h), px(lms,LM_LEFT_WRIST, w,h))
        return r if calc_angle(*r) >= calc_angle(*l) else l

    def check_form(self, lms, w, h):
        for ear, shld in ((LM_RIGHT_EAR,LM_RIGHT_SHOULDER),(LM_LEFT_EAR,LM_LEFT_SHOULDER)):
            if math.dist(px(lms,ear,w,h), px(lms,shld,w,h)) < h * 0.07:
                return False, "Don't shrug — drop your shoulders"
        return True, ""


class TorsoTwist(Exercise):
    """Stand-straight trunk rotation.  Bilateral."""
    name         = "TORSO TWIST"
    target_angle = 50.0
    start_angle  = 90.0

    def _get_pts(self, lms, w, h):
        r = (px(lms,LM_LEFT_SHOULDER,w,h),  px(lms,LM_RIGHT_SHOULDER,w,h), px(lms,LM_RIGHT_HIP,w,h))
        l = (px(lms,LM_RIGHT_SHOULDER,w,h), px(lms,LM_LEFT_SHOULDER,w,h),  px(lms,LM_LEFT_HIP, w,h))
        return r if calc_angle(*r) <= calc_angle(*l) else l

    def check_form(self, lms, w, h):
        lh = px(lms,LM_LEFT_HIP, w,h)
        rh = px(lms,LM_RIGHT_HIP,w,h)
        if abs(lh[1]-rh[1]) > h*0.08:
            return False, "Keep hips still and level"
        return True, ""


class BicepCurl(Exercise):
    """Elbow flexion.  Tracks both arms."""
    name         = "BICEP CURL"
    target_angle = 40.0
    start_angle  = 160.0

    def _get_pts(self, lms, w, h):
        r = (px(lms,LM_RIGHT_SHOULDER,w,h), px(lms,LM_RIGHT_ELBOW,w,h), px(lms,LM_RIGHT_WRIST,w,h))
        l = (px(lms,LM_LEFT_SHOULDER, w,h), px(lms,LM_LEFT_ELBOW, w,h), px(lms,LM_LEFT_WRIST, w,h))
        return r if calc_angle(*r) <= calc_angle(*l) else l

    def check_form(self, lms, w, h):
        for sh, el, hi in (
            (LM_RIGHT_SHOULDER,LM_RIGHT_ELBOW,LM_RIGHT_HIP),
            (LM_LEFT_SHOULDER, LM_LEFT_ELBOW, LM_LEFT_HIP),
        ):
            if calc_angle(px(lms,sh,w,h), px(lms,el,w,h), px(lms,hi,w,h)) < 50:
                return False, "Pin your elbow to your side"
        return True, ""


class CrossBodyReach(Exercise):
    """Horizontal shoulder adduction.  Bilateral."""
    name         = "CROSS-BODY REACH"
    target_angle = 30.0
    start_angle  = 90.0

    def _get_pts(self, lms, w, h):
        r = (px(lms,LM_LEFT_SHOULDER, w,h), px(lms,LM_RIGHT_SHOULDER,w,h), px(lms,LM_RIGHT_WRIST,w,h))
        l = (px(lms,LM_RIGHT_SHOULDER,w,h), px(lms,LM_LEFT_SHOULDER, w,h), px(lms,LM_LEFT_WRIST, w,h))
        return r if calc_angle(*r) <= calc_angle(*l) else l

    def check_form(self, lms, w, h):
        lh = px(lms,LM_LEFT_HIP, w,h)
        rh = px(lms,LM_RIGHT_HIP,w,h)
        if abs(lh[1]-rh[1]) > h*0.08:
            return False, "Don't twist — keep hips level"
        return True, ""


class ShoulderPress(Exercise):
    """Overhead push.  Tracks both arms."""
    name         = "SHOULDER PRESS"
    target_angle = 170.0
    start_angle  =  40.0

    def _get_pts(self, lms, w, h):
        r = (px(lms,LM_RIGHT_ELBOW,w,h), px(lms,LM_RIGHT_SHOULDER,w,h), px(lms,LM_RIGHT_HIP,w,h))
        l = (px(lms,LM_LEFT_ELBOW, w,h), px(lms,LM_LEFT_SHOULDER, w,h), px(lms,LM_LEFT_HIP, w,h))
        return r if calc_angle(*r) >= calc_angle(*l) else l

    def check_form(self, lms, w, h):
        nose   = px(lms,LM_NOSE, w,h)
        r_shld = px(lms,LM_RIGHT_SHOULDER,w,h)
        l_shld = px(lms,LM_LEFT_SHOULDER, w,h)
        mid_x  = (r_shld[0]+l_shld[0])//2
        if nose[0] - mid_x > w*0.10:
            return False, "Straighten your back — don't lean forward"
        return True, ""


# ─────────────────────────────────────────────────────────────────────────────
#  Session History  (JSON append)
# ─────────────────────────────────────────────────────────────────────────────
class SessionHistory:
    """Append each session to a local JSON file for longitudinal tracking."""

    @staticmethod
    def save(exercises: dict, duration_s: float):
        record = {
            "date":       datetime.datetime.now().isoformat(),
            "duration_s": round(duration_s, 1),
            "exercises":  [ex.session_summary() for ex in exercises.values()
                           if ex.set_mgr.total_reps > 0],
        }
        path = CFG.history_file
        history = []
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    history = json.load(f)
            except (json.JSONDecodeError, IOError):
                history = []
        history.append(record)
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"[INFO] Session appended to {os.path.abspath(path)}")
        return history


# ─────────────────────────────────────────────────────────────────────────────
#  Report Generator
# ─────────────────────────────────────────────────────────────────────────────
class Report:
    """Produce a multi-panel Matplotlib session summary."""

    @staticmethod
    def generate(exercises: dict, history: list | None = None):
        active = [ex for ex in exercises.values() if ex.set_mgr.total_reps > 0]
        n_panels = len(active) + (1 if history and len(history) > 1 else 0)
        if n_panels == 0:
            print("[INFO] No reps recorded — skipping report.")
            return

        fig, axes = plt.subplots(1, max(n_panels, 1),
                                 figsize=(5 * max(n_panels, 1), 4.5))
        if n_panels == 1:
            axes = [axes]

        # ── per-exercise angle-depth charts ────────────────────────────────
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, ex in enumerate(active):
            ax = axes[i]
            all_angles = []
            for j, s in enumerate(ex.set_mgr.sets):
                if s.peak_angles:
                    xs = range(1, len(s.peak_angles)+1)
                    ax.plot(xs, s.peak_angles, marker="o", linewidth=1.8,
                            color=colors[j % len(colors)],
                            label=f"Set {j+1}")
                    all_angles.extend(s.peak_angles)

            ax.set_title(ex.name, fontweight="bold", fontsize=10)
            ax.set_xlabel("Rep")
            ax.set_ylabel("Peak Angle (°)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Draw target angle as a dashed reference line
            ax.axhline(ex.target_angle, color="red", linestyle="--",
                       linewidth=1, alpha=0.6, label=f"Target {ex.target_angle}°")
            ax.legend(fontsize=8)

        # ── session history (ROM trend) ─────────────────────────────────────
        if history and len(history) > 1:
            ax = axes[-1]
            for ex_name in {e.name for e in active}:
                dates, avg_peaks = [], []
                for session in history[-10:]:   # last 10 sessions
                    for ex_data in session.get("exercises", []):
                        if ex_data.get("exercise") == ex_name:
                            all_p = [
                                a for s in ex_data.get("sets", [])
                                for a in s.get("peak_angles", [])
                            ]
                            if all_p:
                                dates.append(session["date"][:10])
                                avg_peaks.append(sum(all_p)/len(all_p))
                if dates:
                    ax.plot(range(len(dates)), avg_peaks,
                            marker="o", label=ex_name)
                    ax.set_xticks(range(len(dates)))
                    ax.set_xticklabels(dates, rotation=30, ha="right", fontsize=7)

            ax.set_title("ROM Trend (last 10 sessions)", fontweight="bold", fontsize=10)
            ax.set_ylabel("Avg Peak Angle (°)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Post-Surgery Rehab — Session Report",
                     fontweight="bold", fontsize=13)
        fig.tight_layout()
        path = os.path.abspath(CFG.report_file)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Report saved → {path}")
        try:
            os.startfile(path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  UI drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_skeleton(frame, lms, w, h, form_ok: bool):
    col_dot  = (0, 220, 100) if form_ok else (0, 60, 255)
    col_line = (0, 180,  80) if form_ok else (0, 30, 200)
    pts = {}
    for idx in UPPER_BODY_LANDMARKS:
        if lms[idx].visibility >= 0.40:
            p = px(lms, idx, w, h)
            pts[idx] = p
            cv2.circle(frame, p, 6, col_dot, -1)
            cv2.circle(frame, p, 7, (0,0,0), 1)
    for a, b in UPPER_BODY_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], col_line, 2, cv2.LINE_AA)


def _banner(frame, h_px: int = 80, alpha: float = CFG.hud_alpha):
    """Semi-transparent dark banner at top of frame."""
    ov = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(ov, (0,0), (w, h_px), (15,15,15), -1)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)


def draw_hud_active(frame, angle: float, ex: Exercise,
                    elapsed: float, fps: float, paused: bool):
    h, w = frame.shape[:2]
    _banner(frame, 88)

    sm = ex.set_mgr
    left_col, mid_col, right_col = (15, 30), (w//2-80, 30), (w-220, 30)
    line2 = 62

    # Exercise name (top-left)
    cv2.putText(frame, ex.name, left_col, CFG.font, 0.82,
                (0,255,200), 2, cv2.LINE_AA)

    # Set · Rep progress (second line left)
    set_rep = f"{sm.set_label()}   {sm.rep_label()}"
    cv2.putText(frame, set_rep, (15, line2), CFG.font, 0.62,
                (255,255,255), 1, cv2.LINE_AA)

    # Angle + target (centre top)
    cv2.putText(frame, f"{int(angle)}°", mid_col, CFG.font, 0.70,
                (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"target {int(ex.target_angle)}°",
                (w//2-80, line2), CFG.font, 0.48, (140,140,140), 1, cv2.LINE_AA)

    # Status indicator (top-right)
    if paused:
        stxt, scol = "PAUSED", (0, 200, 255)
    elif ex._hit_target:
        stxt, scol = "RETURN ◀", (0,255,255)
    else:
        stxt, scol = "TRACKING", (130,130,130)
    cv2.putText(frame, stxt, right_col, CFG.font, 0.58, scol, 1, cv2.LINE_AA)

    # Timer (top-right second line)
    cv2.putText(frame, fmt_time(elapsed), (w-160, line2),
                CFG.font, 0.58, (180,220,255), 1, cv2.LINE_AA)

    # FPS (bottom-left)
    cv2.putText(frame, f"FPS {fps:.0f}", (10, h-10),
                CFG.font, 0.42, (90,90,90), 1, cv2.LINE_AA)

    # Form warning (bottom-centre)
    if not ex.form_ok:
        warn = f"⚠  {ex.form_msg}  ⚠"
        (tw, _), _ = cv2.getTextSize(warn, CFG.font, 0.70, 2)
        cv2.putText(frame, warn, ((w-tw)//2, h-14),
                    CFG.font, 0.70, (0,50,255), 2, cv2.LINE_AA)

    # Progress bar (right edge)
    _draw_progress_bar(frame, ex.progress(angle), ex.start_angle, ex.target_angle)


def draw_hud_countdown(frame, ex: Exercise, remaining: float):
    """Full-screen countdown overlay."""
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,h), (0,0,0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, ex.name, (w//2-200, h//2-80),
                CFG.font, 1.0, (0,255,200), 2, cv2.LINE_AA)
    n = max(1, math.ceil(remaining))
    cv2.putText(frame, str(n), (w//2-30, h//2+40),
                CFG.font, 3.5, (255,255,100), 5, cv2.LINE_AA)
    cv2.putText(frame, "Get ready!", (w//2-90, h//2+100),
                CFG.font, 0.80, (200,200,200), 1, cv2.LINE_AA)


def draw_hud_rest(frame, ex: Exercise, remaining: float):
    """REST between sets overlay."""
    h, w = frame.shape[:2]
    _banner(frame, h, alpha=0.45)
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,h), (0,0,20), -1)
    cv2.addWeighted(ov, 0.50, frame, 0.50, 0, frame)

    sm = ex.set_mgr
    cv2.putText(frame, "REST", (w//2-80, h//2-60),
                CFG.font, 2.5, (0,220,255), 4, cv2.LINE_AA)
    cv2.putText(frame, f"Next: {sm.set_label()}", (w//2-120, h//2+20),
                CFG.font, 0.90, (200,200,200), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{int(remaining)} s", (w//2-55, h//2+80),
                CFG.font, 1.60, (255,255,100), 3, cv2.LINE_AA)
    cv2.putText(frame, "Keep breathing. You're doing great.",
                (w//2-230, h//2+140), CFG.font, 0.60, (160,160,160), 1)


def draw_hud_done(frame, ex: Exercise, elapsed: float):
    """All-sets-complete celebration screen."""
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,h), (0,30,0), -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)

    cv2.putText(frame, "WORKOUT COMPLETE!", (w//2-230, h//2-60),
                CFG.font, 1.10, (0,255,120), 3, cv2.LINE_AA)
    sm = ex.set_mgr
    cv2.putText(frame,
                f"{sm.target_sets} sets × {sm.target_reps} reps — {sm.total_reps} total",
                (w//2-260, h//2+10), CFG.font, 0.72, (200,255,200), 1, cv2.LINE_AA)
    cv2.putText(frame, fmt_time(elapsed),
                (w//2-60, h//2+60), CFG.font, 0.80, (180,220,255), 2)
    cv2.putText(frame, "Press another key or Q to quit",
                (w//2-210, h//2+110), CFG.font, 0.60, (150,150,150), 1)


def draw_idle(frame, elapsed: float, fps: float):
    """Splash screen when no exercise is active."""
    h, w = frame.shape[:2]
    _banner(frame, 80)
    cv2.putText(frame, "POST-SURGERY REHAB ASSISTANT v3",
                (14, 30), CFG.font, 0.80, (0,255,200), 2, cv2.LINE_AA)
    cv2.putText(frame,
                "Select: F Flexion  T Twist  B Curl  C Reach  P Press  |  Q quit",
                (14, 60), CFG.font, 0.48, (190,190,190), 1, cv2.LINE_AA)

    entries = [
        ("F","Shoulder Flexion",  "Hip→Shoulder→Wrist", "150°"),
        ("T","Torso Twist",       "Shoulder-axis rotation","50°"),
        ("B","Bicep Curl",        "Shoulder→Elbow→Wrist","40°"),
        ("C","Cross-Body Reach",  "Horizontal adduction", "30°"),
        ("P","Shoulder Press",    "Elbow→Shoulder→Hip",  "170°"),
    ]
    y = 115
    for k, nm, detail, tgt in entries:
        cv2.putText(frame, f"[{k}] {nm}", (28,y),
                    CFG.font, 0.58, (230,230,230), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{detail}  target {tgt}", (175,y+18),
                    CFG.font, 0.40, (130,130,130), 1, cv2.LINE_AA)
        y += 52

    cv2.putText(frame, fmt_time(elapsed), (w-140,60),
                CFG.font, 0.60, (180,220,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.0f}", (10,h-10),
                CFG.font, 0.42, (90,90,90), 1, cv2.LINE_AA)


def _draw_progress_bar(frame, progress: float, start_a: float, target_a: float):
    h, w = frame.shape[:2]
    bw, mg = 28, 16
    x1, x2 = w-mg-bw, w-mg
    y1, y2 = 90, h-50
    bar_h  = y2 - y1

    cv2.rectangle(frame, (x1-2,y1-2),(x2+2,y2+2),(35,35,35),-1)
    cv2.rectangle(frame, (x1-2,y1-2),(x2+2,y2+2),(160,160,160),1)

    fill_h = int(bar_h * progress)
    col = (0,220,100) if progress>=0.80 else \
          (0,200,255) if progress>=0.40 else (0,80,220)
    if fill_h > 0:
        cv2.rectangle(frame,(x1,y2-fill_h),(x2,y2),col,-1)

    cv2.putText(frame, f"{int(progress*100)}%",
                (x1-4,y2+18), CFG.font, 0.46,(200,200,200),1)
    cv2.putText(frame, f"{int(start_a)}°",
                (x1-4,y2+34), CFG.font, 0.38,(130,130,130),1)
    cv2.putText(frame, f"{int(target_a)}°",
                (x1-4,y1-6),  CFG.font, 0.38,(130,130,130),1)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────────────────────────────────────
class RehabAssistant:
    """
    Top-level controller.
    Webcam → PoseLandmarker (async) → Exercise logic → UI → Key dispatch.
    """

    KEY_MAP = {
        ord("f"): ShoulderFlexion,
        ord("t"): TorsoTwist,
        ord("b"): BicepCurl,
        ord("c"): CrossBodyReach,
        ord("p"): ShoulderPress,
    }

    def __init__(self):
        self.voice          = Voice()
        self.exercises      = {}          # chr(key) → Exercise
        self.current_key    = None
        self._latest        = None
        self._lock          = threading.Lock()
        self._paused        = False
        self._session_start = time.monotonic()
        self._fps_buf       = deque(maxlen=30)
        self._last_t        = time.monotonic()

    # ── MediaPipe callback ──────────────────────────────────────────────────

    def _on_result(self, result, _img, _ts):
        with self._lock:
            self._latest = result

    # ── helpers ─────────────────────────────────────────────────────────────

    @property
    def current_exercise(self):
        if self.current_key is None:
            return None
        return self.exercises.get(chr(self.current_key))

    def _switch(self, key: int):
        ch = chr(key)
        if ch not in self.exercises:
            ex = self.KEY_MAP[key]()
            ex.set_mgr.start_first_set()
            self.exercises[ch] = ex
        self.current_key = key
        self._paused = False
        self.voice.say(f"{self.exercises[ch].name} mode")

    def _fps(self) -> float:
        now = time.monotonic()
        dt  = now - self._last_t
        self._last_t = now
        if dt > 0:
            self._fps_buf.append(1.0/dt)
        return sum(self._fps_buf)/max(len(self._fps_buf),1)

    # ── main loop ───────────────────────────────────────────────────────────

    def run(self):
        if not os.path.isfile(CFG.model_path):
            print(f"[ERROR] Model not found: {CFG.model_path}")
            print("Download it with:")
            print('  python -c "import urllib.request; urllib.request.urlretrieve('
                  "'https://storage.googleapis.com/mediapipe-models/"
                  "pose_landmarker/pose_landmarker_lite/float16/latest/"
                  "pose_landmarker_lite.task', 'pose_landmarker_lite.task')" '"')
            return

        opts = PoseLandmarkerOpt(
            base_options=BaseOptions(model_asset_path=CFG.model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=CFG.detect_conf,
            min_tracking_confidence=CFG.track_conf,
            result_callback=self._on_result,
        )
        landmarker = PoseLandmarker.create_from_options(opts)

        cap = cv2.VideoCapture(CFG.cam_index)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            landmarker.close()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_h)
        cap.set(cv2.CAP_PROP_FPS,          CFG.cam_fps)

        print("=" * 58)
        print("  Post-Surgery Rehab Assistant v3  —  Running")
        print(f"  Sets per exercise : {CFG.default_sets}  |  "
              f"Reps per set : {CFG.default_reps}")
        print("  Keys: F T B C P  select  |  SPACE pause  |  Q quit")
        print("=" * 58)

        self._session_start = time.monotonic()
        ts_ms = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame   = cv2.flip(frame, 1)
            h, w    = frame.shape[:2]
            fps     = self._fps()
            elapsed = time.monotonic() - self._session_start

            if not self._paused:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms += 33
                landmarker.detect_async(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
                )

            with self._lock:
                result = self._latest

            ex    = self.current_exercise
            angle = 0.0

            if result and result.pose_landmarks:
                lms = result.pose_landmarks[0]

                if ex is not None:
                    sm = ex.set_mgr

                    # ── Choose overlay based on set-manager state ──────────
                    if sm.state == SetManager.COUNTDOWN:
                        draw_skeleton(frame, lms, w, h, True)
                        draw_hud_countdown(frame, ex, sm.countdown_remaining())
                        if not self._paused:
                            ex.process_frame(lms, w, h, self.voice)

                    elif sm.state == SetManager.REST:
                        draw_skeleton(frame, lms, w, h, True)
                        draw_hud_rest(frame, ex, sm.countdown_remaining())
                        if not self._paused:
                            ex.process_frame(lms, w, h, self.voice)

                    elif sm.state == SetManager.DONE:
                        draw_skeleton(frame, lms, w, h, ex.form_ok)
                        draw_hud_done(frame, ex, elapsed)

                    else:   # ACTIVE
                        if not self._paused:
                            angle = ex.process_frame(lms, w, h, self.voice)
                        draw_skeleton(frame, lms, w, h, ex.form_ok)
                        draw_hud_active(frame, angle, ex, elapsed, fps,
                                        self._paused)
                        _draw_progress_bar(frame, ex.progress(angle),
                                           ex.start_angle, ex.target_angle)
                else:
                    draw_skeleton(frame, lms, w, h, True)
                    draw_idle(frame, elapsed, fps)
            else:
                draw_idle(frame, elapsed, fps)

            cv2.imshow("Rehab Assistant v3  |  SPACE=pause  Q=quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                self._paused = not self._paused
                self.voice.say("Paused" if self._paused else "Resumed")
            elif key in self.KEY_MAP:
                self._switch(key)

        # ── Shutdown ────────────────────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

        duration = time.monotonic() - self._session_start
        print(f"[INFO] Session: {fmt_time(duration)}")

        if self.exercises:
            history = SessionHistory.save(self.exercises, duration)
            Report.generate(self.exercises, history)

        self.voice.stop()
        print("[INFO] Done.")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RehabAssistant().run()
