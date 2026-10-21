#!/usr/bin/env python3
"""
AI Proctoring & Behaviour Analysis System  -  Hackathon Problem Statement ID 102
================================================================================

Run:   python main.py            (press  q  or  ESC  in the video window to quit,  c  to recalibrate)

Pipeline
--------
    Webcam (OpenCV) -> Face back-end (MediaPipe FaceMesh / FaceLandmarker, Haar fallback)
        -> Features (head pose via solvePnP, eye/mouth/brow geometry)
        -> BehaviourAnalyzer (timers, state machine, valence + suspicion scores)
        -> TerminalLogger (live coloured status)  +  AlertManager (cooldown + snapshot)
        -> TelegramNotifier (background thread: sendPhoto with sendMessage fallback)

Detected violations (each one triggers a Telegram alert + snapshot, with a cooldown):
    * NO FACE DETECTED
    * MULTIPLE FACES DETECTED - CHEATING ALERT
    * LOOKING AWAY FROM SCREEN (LEFT / RIGHT / UP / DOWN) for more than 2 seconds
    * (optional) SUSPICIOUS MOVEMENT DETECTED

Important: the "valence" and "suspicion" values are transparent geometric heuristics measured
against the candidate's own neutral baseline (captured during the first second after start-up).
They are NOT a trained emotion classifier and must never be used as the sole basis for a decision.
"""

from __future__ import annotations

import argparse
import html
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from colorama import Fore, Style
from colorama import init as colorama_init

import config

# ======================================================================================
# 1. Data structures
# ======================================================================================

# MediaPipe FaceMesh landmark indices (identical in the legacy and the Tasks model)
POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)  # nose tip, chin, eye corners (image-left/right), mouth corners
R_EYE = (33, 160, 158, 133, 153, 144)  # subject's right eye (appears on the image-left)
L_EYE = (362, 385, 387, 263, 373, 380)

# Generic 3-D face model in OpenCV camera convention (x right, y down, z away from camera), in mm.
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, 330.0, 65.0),  # chin
        (-225.0, -170.0, 135.0),  # eye outer corner, image-left   (landmark 33)
        (225.0, -170.0, 135.0),  # eye outer corner, image-right  (landmark 263)
        (-150.0, 150.0, 125.0),  # mouth corner, image-left       (landmark 61)
        (150.0, 150.0, 125.0),  # mouth corner, image-right      (landmark 291)
    ],
    dtype=np.float64,
)
_NO_DISTORTION = np.zeros((4, 1), dtype=np.float64)

POSE_EMA = 0.5  # smoothing of head-pose angles (1.0 = no smoothing)
VALENCE_EMA = 0.15  # smoothing of valence score


@dataclass
class FaceFeatures:
    """Everything the analyzer needs to know about one detected face."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    yaw: Optional[float] = None  # degrees, + = candidate turns toward image-left (their right)
    pitch: Optional[float] = None  # degrees, + = looking down
    roll: Optional[float] = None
    ear: Optional[float] = None  # eye aspect ratio (blink / closed eyes)
    mar: Optional[float] = None  # mouth aspect ratio (talking)
    mouth_w: Optional[float] = None  # mouth width / face width (smile)
    brow_gap: Optional[float] = None  # inner-brow distance / face width (frown)
    brow_raise: Optional[float] = None  # brow-to-eyelid distance / face height (surprise)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class Violation:
    kind: str  # key used for the alert cooldown
    message: str  # full text shown in terminal / Telegram
    label: str  # short text used as the status


@dataclass
class Analysis:
    faces: int = 0
    status: str = "STARTING"
    direction: str = "CENTER"
    yaw: float = 0.0
    pitch: float = 0.0
    valence: float = 0.0
    valence_label: str = "N/A"
    suspicion: float = 0.0
    focus: float = 100.0
    calibrating: bool = False
    notes: List[str] = field(default_factory=list)
    violations: List[Violation] = field(default_factory=list)


# ======================================================================================
# 2. Face geometry: head pose + facial features
# ======================================================================================


def estimate_head_pose(lm: np.ndarray, w: int, h: int) -> Optional[Tuple[float, float, float]]:
    """Return (yaw, pitch, roll) in degrees from 6 face landmarks via cv2.solvePnP."""
    try:
        pts = np.array([[lm[i][0] * w, lm[i][1] * h] for i in POSE_LANDMARKS], dtype=np.float64)
        focal = float(w)
        cam = np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, pts, cam, _NO_DISTORTION, flags=cv2.SOLVEPNP_ITERATIVE)
    except (cv2.error, IndexError, ValueError):
        return None
    if not ok or float(tvec[2][0]) <= 0.0:
        return None  # face "behind" the camera = degenerate / flipped solution
    rot, _ = cv2.Rodrigues(rvec)
    pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
    yaw = math.degrees(math.atan2(-rot[2, 0], math.hypot(rot[2, 1], rot[2, 2])))
    roll = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    if abs(yaw) > 90.0 or abs(pitch) > 90.0:
        return None
    return yaw, pitch, roll


def features_from_landmarks(lm: np.ndarray, w: int, h: int) -> Optional[FaceFeatures]:
    """Convert a (N>=468, 3) array of normalised FaceMesh landmarks into FaceFeatures."""
    if lm is None or len(lm) < 468:
        return None

    def point(i: int) -> np.ndarray:
        return np.array([lm[i][0] * w, lm[i][1] * h], dtype=np.float64)

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(point(a) - point(b)))

    face_w = dist(234, 454)  # cheek to cheek
    face_h = dist(10, 152)  # forehead to chin
    if face_w < 1e-3 or face_h < 1e-3:
        return None

    xs = np.asarray(lm)[:, 0] * w
    ys = np.asarray(lm)[:, 1] * h
    bbox = (
        int(max(xs.min(), 0)),
        int(max(ys.min(), 0)),
        int(min(xs.max(), w - 1)),
        int(min(ys.max(), h - 1)),
    )

    def eye_aspect_ratio(idx: Tuple[int, ...]) -> float:
        p1, p2, p3, p4, p5, p6 = idx
        return (dist(p2, p6) + dist(p3, p5)) / (2.0 * max(dist(p1, p4), 1e-6))

    mouth_width = max(dist(61, 291), 1e-6)
    pose = estimate_head_pose(lm, w, h)
    yaw = pitch = roll = None
    if pose is not None:
        yaw, pitch, roll = pose

    return FaceFeatures(
        bbox=bbox,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        ear=(eye_aspect_ratio(R_EYE) + eye_aspect_ratio(L_EYE)) / 2.0,
        mar=dist(13, 14) / mouth_width,
        mouth_w=mouth_width / face_w,
        brow_gap=dist(107, 336) / face_w,
        brow_raise=((dist(105, 159) + dist(334, 386)) / 2.0) / face_h,
    )


# ======================================================================================
# 3. Terminal logger
# ======================================================================================


class TerminalLogger:
    """Thread-safe, colourised terminal output (ASCII only so Windows consoles never choke)."""

    STATUS_COLORS = {
        "FOCUSED": Fore.GREEN,
        "CALIBRATING": Fore.CYAN,
        "DISTRACTED": Fore.YELLOW,
        "SUSPICIOUS MOVEMENT": Fore.MAGENTA,
    }

    def __init__(self, interval: float) -> None:
        self._interval = max(0.1, interval)
        self._last: Optional[float] = None
        self._lock = threading.Lock()

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _emit(self, text: str, color: str = "") -> None:
        with self._lock:
            print(f"{color}{text}{Style.RESET_ALL}", flush=True)

    def raw(self, text: str, color: str = "") -> None:
        self._emit(text, color)

    def info(self, msg: str) -> None:
        self._emit(f"[{self._ts()}] [INFO]  {msg}", Fore.CYAN)

    def ok(self, msg: str) -> None:
        self._emit(f"[{self._ts()}] [OK]    {msg}", Fore.GREEN)

    def warn(self, msg: str) -> None:
        self._emit(f"[{self._ts()}] [WARN]  {msg}", Fore.YELLOW)

    def error(self, msg: str) -> None:
        self._emit(f"[{self._ts()}] [ERROR] {msg}", Fore.RED)

    def alert(self, msg: str) -> None:
        self._emit(f"[{self._ts()}] [ALERT] {msg}", Fore.RED + Style.BRIGHT)

    def status(self, now: float, a: Analysis) -> None:
        if self._last is not None and now - self._last < self._interval:
            return
        self._last = now
        color = Fore.RED if a.violations else self.STATUS_COLORS.get(a.status, Fore.WHITE)
        line = (
            f"[{self._ts()}] {a.status:<20} | faces={a.faces} "
            f"| gaze={a.direction:<6} yaw={a.yaw:+6.1f} pitch={a.pitch:+6.1f} "
            f"| valence={a.valence:+.2f} ({a.valence_label}) "
            f"| suspicion={a.suspicion:3.0f}/100 | focus={a.focus:3.0f}/100"
        )
        if a.notes:
            line += " | " + ", ".join(a.notes)
        if a.violations:
            line += "  <<< " + "; ".join(v.message for v in a.violations)
        self._emit(line, color)


# ======================================================================================
# 4. Face detection back-ends
# ======================================================================================


class FaceBackend:
    """Interface: turn a BGR frame into a list of FaceFeatures."""

    name = "base"
    has_landmarks = True

    def detect(self, frame_bgr: np.ndarray, ts_ms: int) -> List[FaceFeatures]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LegacyFaceMeshBackend(FaceBackend):
    """mediapipe.solutions.face_mesh (available in older MediaPipe releases, no download needed)."""

    name = "MediaPipe FaceMesh (legacy solutions API)"

    def __init__(self, max_faces: int) -> None:
        import mediapipe as mp  # imported lazily so a missing/changed package only disables this back-end

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_bgr: np.ndarray, ts_ms: int) -> List[FaceFeatures]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        rgb.flags.writeable = False
        result = self._mesh.process(rgb)
        faces: List[FaceFeatures] = []
        for face in result.multi_face_landmarks or []:
            arr = np.array([[p.x, p.y, p.z] for p in face.landmark], dtype=np.float64)
            feat = features_from_landmarks(arr, w, h)
            if feat is not None:
                faces.append(feat)
        return faces

    def close(self) -> None:
        self._mesh.close()


class TasksFaceLandmarkerBackend(FaceBackend):
    """MediaPipe Tasks FaceLandmarker (the API used by current MediaPipe releases)."""

    name = "MediaPipe FaceLandmarker (Tasks API)"

    def __init__(self, model_path: str, max_faces: int) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, ts_ms: int) -> List[FaceFeatures]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(ts_ms), self._last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        faces: List[FaceFeatures] = []
        for face in result.face_landmarks or []:
            arr = np.array([[p.x, p.y, p.z] for p in face], dtype=np.float64)
            feat = features_from_landmarks(arr, w, h)
            if feat is not None:
                faces.append(feat)
        return faces

    def close(self) -> None:
        self._landmarker.close()


class HaarBackend(FaceBackend):
    """OpenCV Haar cascade fallback: face presence + count + coarse gaze from the face-box position.

    There are no landmarks in this mode, so valence cannot be estimated and head turns are only
    approximated (a fully turned head usually makes the frontal detector lose the face, which is
    then reported as NO FACE).
    """

    name = "OpenCV Haar cascade (fallback - approximate gaze, no valence)"
    has_landmarks = False

    def __init__(self) -> None:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {path}")

    def detect(self, frame_bgr: np.ndarray, ts_ms: int) -> List[FaceFeatures]:
        h, w = frame_bgr.shape[:2]
        scale = 640.0 / w if w > 640 else 1.0
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale != 1.0 else frame_bgr
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        boxes = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(60, 60))
        faces: List[FaceFeatures] = []
        for (x, y, bw, bh) in boxes:
            x1, y1 = int(x / scale), int(y / scale)
            x2, y2 = int((x + bw) / scale), int((y + bh) / scale)
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            faces.append(
                FaceFeatures(
                    bbox=(x1, y1, x2, y2),
                    yaw=-(cx - 0.5) * 100.0,  # face drifts to image-right = candidate's left = negative yaw
                    pitch=(cy - 0.5) * 100.0,  # face drifts down = looking down = positive pitch
                    roll=0.0,
                )
            )
        return faces


def ensure_model(path: Path, url: str, logger: TerminalLogger) -> bool:
    """Make sure the FaceLandmarker model file exists locally (downloads it once, ~4 MB)."""
    try:
        if path.is_file() and path.stat().st_size > 1_000_000:
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading MediaPipe FaceLandmarker model to {path} ...")
        tmp = path.with_suffix(path.suffix + ".part")
        with requests.get(url, stream=True, timeout=(10, 60)) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        fh.write(chunk)
        if tmp.stat().st_size < 1_000_000:
            tmp.unlink()
            return False
        tmp.replace(path)
        logger.ok("Model downloaded.")
        return True
    except Exception as exc:  # noqa: BLE001 - any failure just means "fall back to the next back-end"
        logger.warn(f"Could not obtain the model file: {exc}")
        return False


def build_backend(preference: str, logger: TerminalLogger) -> FaceBackend:
    """Pick the best available back-end (auto: legacy FaceMesh -> Tasks FaceLandmarker -> Haar)."""
    orders = {
        "auto": ["legacy", "tasks", "haar"],
        "legacy": ["legacy", "haar"],
        "tasks": ["tasks", "haar"],
        "haar": ["haar"],
    }
    for name in orders.get(preference, orders["auto"]):
        try:
            if name == "legacy":
                return LegacyFaceMeshBackend(config.MAX_FACES)
            if name == "tasks":
                if not ensure_model(config.FACE_LANDMARKER_MODEL_PATH, config.FACE_LANDMARKER_MODEL_URL, logger):
                    raise RuntimeError("model file not available (check your internet connection)")
                return TasksFaceLandmarkerBackend(str(config.FACE_LANDMARKER_MODEL_PATH), config.MAX_FACES)
            if name == "haar":
                backend = HaarBackend()
                if preference != "haar":
                    logger.warn("Using the Haar fallback: gaze is approximate and valence is unavailable.")
                return backend
        except Exception as exc:  # noqa: BLE001
            logger.warn(f"Back-end '{name}' is not available: {exc}")
    raise RuntimeError("No face detection back-end could be started.")


# ======================================================================================
# 5. Behaviour analysis (timers, state machine, valence / suspicion scores)
# ======================================================================================


class Baseline:
    """Median of the candidate's neutral pose / expression, used to make every metric personal."""

    def __init__(self, needed: int, clip_deg: Optional[float]) -> None:
        self.needed = max(1, needed)
        self.clip_deg = clip_deg
        self.ready = False
        self._count = 0
        self._samples: Dict[str, List[float]] = {k: [] for k in ("yaw", "pitch", "ear", "mouth_w", "brow_gap", "brow_raise")}
        self.yaw = 0.0
        self.pitch = 0.0
        self.ear: Optional[float] = None
        self.mouth_w: Optional[float] = None
        self.brow_gap: Optional[float] = None
        self.brow_raise: Optional[float] = None

    def add(self, f: FaceFeatures) -> None:
        for key in self._samples:
            value = getattr(f, key)
            if value is not None:
                self._samples[key].append(float(value))
        self._count += 1
        if self._count >= self.needed:
            self._finalize()

    def _median(self, key: str) -> Optional[float]:
        values = self._samples[key]
        return float(np.median(values)) if values else None

    def _finalize(self) -> None:
        yaw = self._median("yaw") or 0.0
        pitch = self._median("pitch") or 0.0
        if self.clip_deg is not None:  # a candidate must not be able to hide cheating by calibrating while looking away
            yaw = float(np.clip(yaw, -self.clip_deg, self.clip_deg))
            pitch = float(np.clip(pitch, -self.clip_deg, self.clip_deg))
        self.yaw, self.pitch = yaw, pitch
        self.ear = self._median("ear")
        self.mouth_w = self._median("mouth_w")
        self.brow_gap = self._median("brow_gap")
        self.brow_raise = self._median("brow_raise")
        self.ready = True


class BehaviourAnalyzer:
    """Turns per-frame FaceFeatures into a status, violations and scores."""

    def __init__(self, has_landmarks: bool = True) -> None:
        self._has_landmarks = has_landmarks
        self.recalibrate()

    # ---- state management ---------------------------------------------------------
    def recalibrate(self) -> None:
        self.baseline = Baseline(config.CALIBRATION_FRAMES, 20.0 if self._has_landmarks else None)
        self._yaw = 0.0
        self._pitch = 0.0
        self._pose_init = False
        self._direction = "CENTER"
        self._valence = 0.0
        self._no_face_since: Optional[float] = None
        self._multi_since: Optional[float] = None
        self._away_since: Optional[float] = None
        self._closed_since: Optional[float] = None
        self._susp_since: Optional[float] = None
        self._motion: Deque[Tuple[float, float, float]] = deque()
        self._mar: Deque[Tuple[float, float]] = deque()

    @staticmethod
    def _track(since: Optional[float], condition: bool, now: float) -> Optional[float]:
        """Start / keep / clear a 'since when has this been true' timer."""
        if not condition:
            return None
        return now if since is None else since

    @staticmethod
    def _duration(since: Optional[float], now: float) -> float:
        return 0.0 if since is None else max(0.0, now - since)

    def _classify(self, dyaw: float, dpitch: float) -> str:
        scale = 0.8 if self._direction != "CENTER" else 1.0  # hysteresis: easier to stay "away" than to become it
        yaw_ratio = abs(dyaw) / max(config.YAW_THRESHOLD_DEG * scale, 1e-6)
        pitch_ratio = abs(dpitch) / max(config.PITCH_THRESHOLD_DEG * scale, 1e-6)
        if max(yaw_ratio, pitch_ratio) < 1.0:
            return "CENTER"
        if yaw_ratio >= pitch_ratio:
            # yaw > 0 means the nose points toward image-left, i.e. the candidate's own RIGHT.
            right = dyaw > 0
            if config.MIRROR_DIRECTIONS:
                right = not right
            return "RIGHT" if right else "LEFT"
        return "DOWN" if dpitch > 0 else "UP"

    # ---- main entry point ---------------------------------------------------------
    def update(self, now: float, faces: List[FaceFeatures]) -> Analysis:
        n = len(faces)
        primary = max(faces, key=lambda f: f.area) if faces else None
        has_pose = primary is not None and primary.yaw is not None and primary.pitch is not None
        base = self.baseline

        self._no_face_since = self._track(self._no_face_since, n == 0, now)
        self._multi_since = self._track(self._multi_since, n > 1, now)

        # --- calibration (needs exactly one face looking at the screen) ---
        if has_pose and not base.ready and n == 1:
            base.add(primary)
        calibrating = primary is not None and not base.ready

        # --- head pose relative to the personal baseline ---
        if primary is None:
            self._pose_init = False
            self._direction = "CENTER"
            self._motion.clear()
            self._mar.clear()
        elif has_pose and base.ready:
            rel_yaw = primary.yaw - base.yaw
            rel_pitch = primary.pitch - base.pitch
            if self._pose_init:
                self._yaw = POSE_EMA * rel_yaw + (1.0 - POSE_EMA) * self._yaw
                self._pitch = POSE_EMA * rel_pitch + (1.0 - POSE_EMA) * self._pitch
            else:
                self._yaw, self._pitch, self._pose_init = rel_yaw, rel_pitch, True
            self._direction = self._classify(self._yaw, self._pitch)
            self._motion.append((now, self._yaw, self._pitch))
        elif not base.ready:
            self._direction = "CENTER"
        # (landmarks present but pose solve failed this frame: keep the previous direction)

        while self._motion and now - self._motion[0][0] > config.MOTION_WINDOW_SECONDS:
            self._motion.popleft()
        self._away_since = self._track(self._away_since, self._direction != "CENTER", now)
        away_dur = self._duration(self._away_since, now)

        # --- rapid head movement ---
        motion_range = 0.0
        if len(self._motion) >= 5:
            yaws = [m[1] for m in self._motion]
            pitches = [m[2] for m in self._motion]
            motion_range = (max(yaws) - min(yaws)) + (max(pitches) - min(pitches))
        motion_high = primary is not None and motion_range >= config.MOTION_RANGE_DEG

        # --- talking / whispering (mouth-opening variability) ---
        talking = False
        if primary is not None and primary.mar is not None:
            self._mar.append((now, primary.mar))
            while self._mar and now - self._mar[0][0] > config.MOTION_WINDOW_SECONDS:
                self._mar.popleft()
            if len(self._mar) >= 8:
                talking = float(np.std([m[1] for m in self._mar])) >= config.TALKING_MAR_STD

        # --- eyes closed for a long time (drowsy / looking down at something) ---
        eyes_closed_long = False
        if primary is not None and primary.ear is not None:
            threshold = 0.6 * base.ear if (base.ready and base.ear) else 0.15
            self._closed_since = self._track(self._closed_since, primary.ear < threshold, now)
            eyes_closed_long = self._duration(self._closed_since, now) >= config.EYES_CLOSED_SECONDS
        else:
            self._closed_since = None

        # --- emotional valence in [-1, +1] (smile up, frown / raised brows down) ---
        valence_available = (
            primary is not None
            and base.ready
            and None not in (primary.mouth_w, primary.brow_gap, primary.brow_raise)
            and bool(base.mouth_w and base.brow_gap and base.brow_raise)
        )
        if valence_available:
            smile = (primary.mouth_w - base.mouth_w) / base.mouth_w
            frown = (base.brow_gap - primary.brow_gap) / base.brow_gap
            surprise = (primary.brow_raise - base.brow_raise) / base.brow_raise
            raw = 4.0 * smile - 3.0 * frown - 1.0 * max(surprise, 0.0)
            raw = float(np.clip(raw, -1.0, 1.0))
            self._valence = (1.0 - VALENCE_EMA) * self._valence + VALENCE_EMA * raw
        valence = float(np.clip(self._valence, -1.0, 1.0)) if valence_available else 0.0
        if not valence_available:
            valence_label = "N/A"
        elif valence > 0.15:
            valence_label = "Positive"
        elif valence < -0.15:
            valence_label = "Negative"
        else:
            valence_label = "Neutral"

        # --- suspicious behaviour (sustained) ---
        self._susp_since = self._track(self._susp_since, bool(motion_high or talking), now)
        suspicious_sustained = self._duration(self._susp_since, now) >= config.SUSPICIOUS_SECONDS

        # --- violations (most severe first) ---
        violations: List[Violation] = []
        if self._duration(self._multi_since, now) >= config.MULTI_FACE_SECONDS:
            violations.append(
                Violation("MULTIPLE_FACES", "MULTIPLE FACES DETECTED - CHEATING ALERT", "MULTIPLE FACES")
            )
        if self._duration(self._no_face_since, now) >= config.NO_FACE_SECONDS:
            violations.append(Violation("NO_FACE", "NO FACE DETECTED", "NO FACE"))
        if away_dur >= config.LOOK_AWAY_SECONDS:
            violations.append(
                Violation("LOOKING_AWAY", f"LOOKING AWAY FROM SCREEN ({self._direction})", "LOOKING AWAY")
            )
        if config.ALERT_ON_SUSPICIOUS_MOVEMENT and suspicious_sustained:
            violations.append(Violation("SUSPICIOUS_MOVEMENT", "SUSPICIOUS MOVEMENT DETECTED", "SUSPICIOUS MOVEMENT"))

        # --- notes shown next to the status line ---
        notes: List[str] = []
        if motion_high:
            notes.append("rapid head movement")
        if talking:
            notes.append("mouth moving")
        if eyes_closed_long:
            notes.append("eyes closed")

        # --- suspicion score 0..100 ---
        if n == 0:
            suspicion = min(100.0, 60.0 + 10.0 * self._duration(self._no_face_since, now))
        elif n > 1:
            suspicion = 95.0
        else:
            pose_dev = 0.0
            if base.ready and has_pose:
                pose_dev = max(
                    abs(self._yaw) / max(config.YAW_THRESHOLD_DEG, 1e-6),
                    abs(self._pitch) / max(config.PITCH_THRESHOLD_DEG, 1e-6),
                )
            suspicion = 40.0 * min(pose_dev, 1.0)
            suspicion += 20.0 * min(away_dur / max(config.LOOK_AWAY_SECONDS, 1e-6), 1.0)
            suspicion += 15.0 * min(motion_range / max(config.MOTION_RANGE_DEG, 1e-6), 1.0)
            suspicion += 10.0 if talking else 0.0
            suspicion += 5.0 if eyes_closed_long else 0.0
            suspicion += 10.0 * max(-valence, 0.0)
            suspicion = float(np.clip(suspicion, 0.0, 100.0))
        focus = 0.0 if n == 0 else float(np.clip(100.0 - suspicion, 0.0, 100.0))

        # --- status ---
        if violations:
            status = violations[0].label
        elif calibrating:
            status = "CALIBRATING"
        elif motion_high or talking:
            status = "SUSPICIOUS MOVEMENT"
        elif n != 1 or self._direction != "CENTER" or eyes_closed_long:
            status = "DISTRACTED"
        else:
            status = "FOCUSED"

        show_pose = primary is not None and base.ready
        return Analysis(
            faces=n,
            status=status,
            direction=self._direction if primary is not None else "N/A",
            yaw=self._yaw if show_pose else 0.0,
            pitch=self._pitch if show_pose else 0.0,
            valence=valence,
            valence_label=valence_label,
            suspicion=suspicion,
            focus=focus,
            calibrating=calibrating,
            notes=notes,
            violations=violations,
        )


# ======================================================================================
# 6. Telegram notifier (background thread) + alert manager (cooldown + snapshots)
# ======================================================================================


@dataclass
class AlertItem:
    timestamp: str
    message: str
    valence: float
    valence_label: str
    suspicion: float
    photo_path: Optional[str]


def build_caption(item: AlertItem) -> str:
    """Telegram HTML caption (max 1024 chars for photos - this stays far below)."""
    return (
        "\U0001F6A8 <b>PROCTORING ALERT</b>\n"
        f"\U0001F552 <b>Time:</b> {html.escape(item.timestamp)}\n"
        f"\u26A0\uFE0F <b>Violation:</b> {html.escape(item.message)}\n"
        f"\U0001F4CA <b>Valence:</b> {item.valence:+.2f} ({html.escape(item.valence_label)})\n"
        f"\U0001F50D <b>Suspicion score:</b> {item.suspicion:.0f}/100\n"
        "\U0001F194 Problem ID 102 - AI Proctoring &amp; Behaviour Analysis"
    )


class TelegramNotifier:
    """Sends alerts through the Telegram Bot HTTP API without ever blocking the video loop."""

    MIN_GAP_SECONDS = 1.0  # Telegram allows roughly one message per second per chat

    def __init__(self, token: str, chat_id: str, logger: TerminalLogger) -> None:
        self._token = token
        self._chat_id = chat_id
        self._log = logger
        self.enabled = bool(token and chat_id)
        self._queue: "queue.Queue[Optional[AlertItem]]" = queue.Queue(maxsize=20)
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle ----
    def start(self) -> None:
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._worker, name="telegram-sender", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(None)  # sentinel: finish pending alerts, then exit
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        self._thread = None

    # ---- public API ----
    def send_alert(self, item: AlertItem) -> bool:
        if not self.enabled:
            return False
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self._log.warn("Telegram queue is full - alert dropped.")
            return False

    def send_test(self) -> bool:
        """Synchronous test message used by `python main.py --test-telegram`."""
        if not self.enabled:
            self._log.error("Telegram is not configured. Fill TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
            return False
        text = (
            "\u2705 <b>Proctoring bot connected</b>\n"
            f"\U0001F552 {html.escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n"
            "Problem ID 102 - AI Proctoring &amp; Behaviour Analysis"
        )
        ok, _ = self._post("sendMessage", {"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"})
        if ok:
            self._log.ok("Test message delivered to Telegram.")
        return ok

    # ---- internals ----
    def _redact(self, text: str) -> str:
        return text.replace(self._token, "<TOKEN>") if self._token else text

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._deliver(item)
                time.sleep(self.MIN_GAP_SECONDS)
            except Exception as exc:  # noqa: BLE001 - the sender thread must never die
                self._log.error(f"Telegram worker error: {self._redact(str(exc))}")
            finally:
                self._queue.task_done()

    def _deliver(self, item: AlertItem) -> None:
        caption = build_caption(item)
        ok, fatal = False, False
        if item.photo_path and os.path.isfile(item.photo_path):
            data = {"chat_id": self._chat_id, "caption": caption, "parse_mode": "HTML"}
            ok, fatal = self._post("sendPhoto", data, photo_path=item.photo_path)
        if not ok and not fatal:
            data = {"chat_id": self._chat_id, "text": caption, "parse_mode": "HTML"}
            ok, _ = self._post("sendMessage", data)
        if ok:
            self._log.ok(f"Telegram alert delivered: {item.message}")

    def _post(self, method: str, data: dict, photo_path: Optional[str] = None) -> Tuple[bool, bool]:
        """POST to the Bot API. Returns (success, fatal) - fatal means 'do not bother retrying another method'."""
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        for attempt in range(2):
            try:
                if photo_path:
                    with open(photo_path, "rb") as fh:
                        resp = requests.post(url, data=data, files={"photo": fh}, timeout=(5, 25))
                else:
                    resp = requests.post(url, data=data, timeout=(5, 15))
            except (requests.RequestException, OSError) as exc:
                self._log.error(f"Telegram {method} failed: {self._redact(str(exc))}")
                return False, True

            try:
                payload = resp.json()
            except ValueError:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}

            if resp.status_code == 429 and attempt == 0:  # rate limited: wait as instructed, retry once
                wait = 3
                try:
                    wait = min(30, int(payload.get("parameters", {}).get("retry_after", 3)))
                except (TypeError, ValueError, AttributeError):
                    pass
                time.sleep(wait + 0.5)
                continue

            if resp.ok and payload.get("ok"):
                return True, False
            description = payload.get("description") or resp.text[:120]
            self._log.error(f"Telegram {method} rejected (HTTP {resp.status_code}): {self._redact(str(description))}")
            return False, resp.status_code in (401, 403, 404)
        return False, False


class AlertManager:
    """Applies the per-violation cooldown, saves the snapshot and hands the alert to Telegram."""

    def __init__(self, notifier: TelegramNotifier, logger: TerminalLogger, cooldown: float, snapshot_dir: Path) -> None:
        self._notifier = notifier
        self._log = logger
        self._cooldown = max(0.0, cooldown)
        self._dir = Path(snapshot_dir)
        self._last: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def handle(self, now: float, analysis: Analysis, annotated_frame: np.ndarray) -> None:
        for violation in analysis.violations:
            last = self._last.get(violation.kind)
            if last is not None and (now - last) < self._cooldown:
                continue  # still cooling down for this violation type
            self._last[violation.kind] = now
            self.counts[violation.kind] = self.counts.get(violation.kind, 0) + 1

            stamp = datetime.now()
            snapshot = self._save_snapshot(annotated_frame, violation.kind, stamp)
            self._log.alert(
                f"{violation.message} | valence={analysis.valence:+.2f} ({analysis.valence_label}) "
                f"| suspicion={analysis.suspicion:.0f}/100 | snapshot={snapshot or 'not saved'}"
            )
            if self._notifier.enabled:
                self._notifier.send_alert(
                    AlertItem(
                        timestamp=stamp.strftime("%Y-%m-%d %H:%M:%S"),
                        message=violation.message,
                        valence=analysis.valence,
                        valence_label=analysis.valence_label,
                        suspicion=analysis.suspicion,
                        photo_path=snapshot,
                    )
                )

    def _save_snapshot(self, frame: np.ndarray, kind: str, stamp: datetime) -> Optional[str]:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"{kind.lower()}_{stamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
            if cv2.imwrite(str(path), frame):
                return str(path)
            self._log.warn(f"cv2.imwrite could not write {path}")
        except (OSError, cv2.error) as exc:
            self._log.warn(f"Could not save snapshot: {exc}")
        return None


# ======================================================================================
# 7. On-screen overlay (also what gets saved as the violation snapshot)
# ======================================================================================

GREEN = (80, 200, 80)
RED = (40, 40, 230)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame: np.ndarray, a: Analysis, faces: List[FaceFeatures]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    color = RED if a.violations else GREEN

    for f in faces:
        x1, y1, x2, y2 = f.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    lines = [
        f"Status: {a.status}",
        f"Faces: {a.faces}   Gaze: {a.direction}",
        f"Valence: {a.valence:+.2f} ({a.valence_label})",
        f"Suspicion: {a.suspicion:.0f}/100   Focus: {a.focus:.0f}/100",
    ]
    box_w = min(w - 10, 340)
    panel = out.copy()
    cv2.rectangle(panel, (5, 5), (5 + box_w, 12 + 22 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, out, 0.45, 0, out)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (12, 25 + 22 * i), FONT, 0.55, color if i == 0 else WHITE, 1, cv2.LINE_AA)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(out, stamp, (max(5, w - 185), 20), FONT, 0.5, WHITE, 1, cv2.LINE_AA)

    if a.violations:
        banner_h = 12 + 24 * len(a.violations)
        cv2.rectangle(out, (0, h - banner_h), (w, h), RED, -1)
        for i, v in enumerate(a.violations):
            cv2.putText(out, v.message, (10, h - banner_h + 24 + 24 * i), FONT, 0.62, WHITE, 2, cv2.LINE_AA)
    elif a.calibrating:
        cv2.putText(out, "Calibrating - look straight at the screen", (10, h - 12), FONT, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    return out


# ======================================================================================
# 8. Application
# ======================================================================================

WINDOW_NAME = "AI Proctoring - Problem ID 102  (q/ESC = quit, c = recalibrate)"


class ProctoringApp:
    def __init__(
        self,
        camera_index: int,
        backend_preference: str,
        show_window: bool,
        logger: TerminalLogger,
        notifier: TelegramNotifier,
        backend: Optional[FaceBackend] = None,
        capture=None,
        max_frames: Optional[int] = None,
    ) -> None:
        self.camera_index = camera_index
        self.backend_preference = backend_preference
        self.show_window = show_window
        self.log = logger
        self.notifier = notifier
        self.backend = backend
        self.capture = capture
        self.max_frames = max_frames
        self.alerts = AlertManager(notifier, logger, config.ALERT_COOLDOWN_SECONDS, config.SNAPSHOT_DIR)
        self.analyzer: Optional[BehaviourAnalyzer] = None

    def _open_camera(self):
        cap = None
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # faster / more reliable than MSMF
            if not cap.isOpened():
                cap.release()
                cap = None
        if cap is None:
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        return cap

    def _show(self, frame: np.ndarray) -> bool:
        """Display the frame. Returns False when the user asked to quit."""
        try:
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            self.log.warn(f"Video window unavailable ({exc}). Continuing in terminal-only mode.")
            self.show_window = False
            return True
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("c"), ord("C")) and self.analyzer is not None:
            self.analyzer.recalibrate()
            self.log.info("Recalibrating - look straight at the screen ...")
        return True

    def _banner(self) -> None:
        self.log.raw("=" * 78, Fore.CYAN)
        self.log.raw("  AI PROCTORING & BEHAVIOUR ANALYSIS SYSTEM  -  Hackathon Problem ID 102", Fore.CYAN + Style.BRIGHT)
        self.log.raw("=" * 78, Fore.CYAN)

    def run(self) -> int:
        self._banner()
        if (
            self.show_window
            and sys.platform.startswith("linux")
            and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        ):
            self.log.warn("No display detected - running in terminal-only mode.")
            self.show_window = False
        if self.backend is None:
            try:
                self.backend = build_backend(self.backend_preference, self.log)
            except RuntimeError as exc:
                self.log.error(str(exc))
                return 2
        self.log.info(f"Detection back-end : {self.backend.name}")
        self.log.info(f"Alert cooldown     : {config.ALERT_COOLDOWN_SECONDS:g}s per violation type")
        self.log.info(f"Look-away trigger  : {config.LOOK_AWAY_SECONDS:g}s")
        if self.notifier.enabled:
            self.log.ok("Telegram alerts enabled.")
        else:
            self.log.warn("Telegram NOT configured (set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env) - alerts are logged locally only.")

        self.analyzer = BehaviourAnalyzer(self.backend.has_landmarks)
        cap = self.capture if self.capture is not None else self._open_camera()
        if cap is None:
            self.log.error(
                f"Cannot open camera index {self.camera_index}. Close other apps using the webcam, "
                "try CAMERA_INDEX=1 in .env, and check the OS camera permission."
            )
            self.backend.close()
            return 3

        self.notifier.start()
        self.log.info("Calibrating: look straight at the screen for a second ...")
        failures = 0
        frames = 0
        exit_code = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 30:
                        self.log.error("Camera stopped delivering frames.")
                        exit_code = 4
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                if config.FLIP_FRAME:
                    frame = cv2.flip(frame, 1)

                now = time.monotonic()
                faces = self.backend.detect(frame, int(now * 1000))
                analysis = self.analyzer.update(now, faces)
                annotated = draw_overlay(frame, analysis, faces)

                self.log.status(now, analysis)
                self.alerts.handle(now, analysis, annotated)

                if self.show_window and not self._show(annotated):
                    break
                frames += 1
                if self.max_frames is not None and frames >= self.max_frames:
                    break
        except KeyboardInterrupt:
            self.log.info("Interrupted by user.")
        finally:
            cap.release()
            self.backend.close()
            self.log.info("Flushing pending Telegram alerts ...")
            self.notifier.stop()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            if self.alerts.counts:
                summary = ", ".join(f"{k}={v}" for k, v in sorted(self.alerts.counts.items()))
                self.log.info(f"Session summary - alerts raised: {summary}")
            else:
                self.log.info("Session summary - no violations raised.")
        return exit_code


# ======================================================================================
# 9. CLI entry point
# ======================================================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Proctoring & Behaviour Analysis System (Hackathon Problem ID 102)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="webcam index (default from .env / 0)")
    parser.add_argument("--no-window", action="store_true", help="terminal-only mode (no video window)")
    parser.add_argument(
        "--backend",
        choices=["auto", "legacy", "tasks", "haar"],
        default=config.FACE_BACKEND if config.FACE_BACKEND in {"auto", "legacy", "tasks", "haar"} else "auto",
        help="face detection back-end (default: auto)",
    )
    parser.add_argument("--test-telegram", action="store_true", help="send a test message to Telegram and exit")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    colorama_init()
    args = parse_args(argv)
    logger = TerminalLogger(config.LOG_INTERVAL_SECONDS)
    notifier = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID, logger)

    if args.test_telegram:
        return 0 if notifier.send_test() else 1

    app = ProctoringApp(
        camera_index=args.camera,
        backend_preference=args.backend,
        show_window=config.SHOW_WINDOW and not args.no_window,
        logger=logger,
        notifier=notifier,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
