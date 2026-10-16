"""Settings for the LEGO Coach app.

Same model setup as the course demos (common.py):
    default            free open-source model on your laptop with Ollama (Google Gemma)
    LEGO_COACH_API=1   OpenAI models (needs OPENAI_API_KEY in a .env file); the robot speaks with OpenAI TTS

Everything can be changed with environment variables or a .env file in the folder you run from.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


PACKAGE_DIR = Path(__file__).resolve().parent
BUILDS_DIR = PACKAGE_DIR / "builds"

# ---------------------------------------------------------------- robot
# Leave empty to let the SDK find the robot (localhost, then network scouting).
ROBOT_IP = os.getenv("ROBOT_IP", "").strip()

# ---------------------------------------------------------------- models
USE_API = _flag("LEGO_COACH_API")
if USE_API:
    BASE_URL = None  # OpenAI
    API_KEY = os.getenv("OPENAI_API_KEY")
    VLM_MODEL = os.getenv("VLM_MODEL", "gpt-6.1-sol")
    TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
else:
    BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
    API_KEY = "ollama"
    VLM_MODEL = os.getenv("VLM_MODEL", "gemma3:4b")
    TTS_MODEL = None  # local voice with pyttsx3 instead

# ---------------------------------------------------------------- experiment
# Independent variable for H2: what the VLM gets for each step.
#   "text"            the written instruction for the step only
#   "reference_image" the written instruction + a photo of the correctly finished step
CONDITIONS = ("text", "reference_image")
DEFAULT_CONDITION = os.getenv("LEGO_COACH_CONDITION", "text")

# Ground-truth labels the experimenter sets before each staged trial (RQ1).
GROUND_TRUTH_LABELS = ("correct", "missing_brick", "wrong_color", "wrong_position", "wrong_orientation")

# Add the MediaPipe brick summary to the VLM prompt.
USE_BRICK_HINTS = _flag("LEGO_COACH_BRICK_HINTS", "1")

DATA_DIR = Path(os.getenv("LEGO_COACH_DATA", "lego_coach_data")).resolve()

# ---------------------------------------------------------------- perception (MediaPipe)
MODELS_DIR = Path(os.getenv("LEGO_COACH_MODELS", Path.home() / ".cache" / "lego_coach")).resolve()
GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)
# Optional custom MediaPipe object detector trained on LEGO bricks (see README).
# When unset, bricks are found with a color detector instead.
LEGO_DETECTOR_MODEL = os.getenv("LEGO_DETECTOR_MODEL", "").strip()

THUMBS_UP_HOLD_S = 0.8  # how long a thumbs-up must be held to count as "I'm done"
HANDS_CLEAR_S = 0.6  # hands must be out of view this long before the photo is taken
HANDS_CLEAR_TIMEOUT_S = 6.0  # take the photo anyway after this long

# ---------------------------------------------------------------- robot pose
LOOK_AT_BUILD_PITCH = 22  # degrees down toward the building area
