"""
config.py - central configuration for the AI Proctoring & Behaviour Analysis System
(Hackathon Problem Statement ID 102).

Secrets (Telegram token / chat id) are read from a local `.env` file (see `.env.example`)
or from real environment variables. Every other setting has a sensible default and can be
overridden from `.env` as well, so you never have to edit this file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load <project>/.env (works no matter which directory you launch `python main.py` from).
load_dotenv(BASE_DIR / ".env")


# ----------------------------------------------------------------------------------------
# Small helpers (never crash on a typo in .env - fall back to the default instead)
# ----------------------------------------------------------------------------------------
def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _secret(name: str) -> str:
    """Return a secret, treating the .env.example placeholders as 'not set'."""
    value = _env_str(name)
    if value.lower().startswith(("your_", "paste_", "<")):
        return ""
    return value


# ----------------------------------------------------------------------------------------
# Telegram
# ----------------------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = _secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _secret("TELEGRAM_CHAT_ID")
ALERT_COOLDOWN_SECONDS = _env_float("ALERT_COOLDOWN_SECONDS", 5.0)  # per violation type

# ----------------------------------------------------------------------------------------
# Camera / display
# ----------------------------------------------------------------------------------------
CAMERA_INDEX = _env_int("CAMERA_INDEX", 0)
FRAME_WIDTH = _env_int("FRAME_WIDTH", 640)
FRAME_HEIGHT = _env_int("FRAME_HEIGHT", 480)
FLIP_FRAME = _env_bool("FLIP_FRAME", False)  # set True only if your camera driver mirrors the image
SHOW_WINDOW = _env_bool("SHOW_WINDOW", True)

# ----------------------------------------------------------------------------------------
# Detection back-end: auto | legacy | tasks | haar
#   legacy -> mediapipe.solutions.face_mesh (older MediaPipe releases)
#   tasks  -> MediaPipe FaceLandmarker (new API, model auto-downloaded on first run)
#   haar   -> OpenCV Haar cascade (no landmarks; coarse gaze only)
# ----------------------------------------------------------------------------------------
FACE_BACKEND = _env_str("FACE_BACKEND", "auto").lower()
MAX_FACES = _env_int("MAX_FACES", 4)
FACE_LANDMARKER_MODEL_PATH = Path(
    _env_str("FACE_LANDMARKER_MODEL_PATH", str(BASE_DIR / "models" / "face_landmarker.task"))
)
FACE_LANDMARKER_MODEL_URL = _env_str(
    "FACE_LANDMARKER_MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
)

# ----------------------------------------------------------------------------------------
# Proctoring rules
# ----------------------------------------------------------------------------------------
LOOK_AWAY_SECONDS = _env_float("LOOK_AWAY_SECONDS", 2.0)  # must look away this long to trigger
NO_FACE_SECONDS = _env_float("NO_FACE_SECONDS", 1.0)  # debounce so a blink of lost tracking is ignored
MULTI_FACE_SECONDS = _env_float("MULTI_FACE_SECONDS", 0.5)

YAW_THRESHOLD_DEG = _env_float("YAW_THRESHOLD_DEG", 25.0)  # left / right
PITCH_THRESHOLD_DEG = _env_float("PITCH_THRESHOLD_DEG", 18.0)  # up / down
MIRROR_DIRECTIONS = _env_bool("MIRROR_DIRECTIONS", False)  # swap LEFT<->RIGHT labels if they feel reversed

CALIBRATION_FRAMES = _env_int("CALIBRATION_FRAMES", 30)  # neutral-pose frames used as the baseline

# Behaviour heuristics
MOTION_WINDOW_SECONDS = _env_float("MOTION_WINDOW_SECONDS", 2.0)
MOTION_RANGE_DEG = _env_float("MOTION_RANGE_DEG", 45.0)  # yaw-range + pitch-range inside the window
TALKING_MAR_STD = _env_float("TALKING_MAR_STD", 0.03)  # mouth-opening variability => talking
EYES_CLOSED_SECONDS = _env_float("EYES_CLOSED_SECONDS", 1.5)
ALERT_ON_SUSPICIOUS_MOVEMENT = _env_bool("ALERT_ON_SUSPICIOUS_MOVEMENT", False)
SUSPICIOUS_SECONDS = _env_float("SUSPICIOUS_SECONDS", 3.0)

# ----------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------
LOG_INTERVAL_SECONDS = _env_float("LOG_INTERVAL_SECONDS", 1.0)
SNAPSHOT_DIR = Path(_env_str("SNAPSHOT_DIR", str(BASE_DIR / "snapshots")))
