"""Shared helpers for the three Reachy Mini demos (LLM / VLM / VLA).

Models:
    python demoX.py         free open-source models, running on your laptop with Ollama
    python demoX.py --api   OpenAI models (needs OPENAI_API_KEY in a .env file);
                            the robot also says its answers out loud
"""

import base64
import os
import sys
import time
import wave

# Hide GStreamer's harmless "generated empty RTCP messages" spam from the camera stream.
# Must be set before reachy_mini starts GStreamer.
os.environ.setdefault("GST_DEBUG", "rtpsession:0")

import cv2
from dotenv import load_dotenv
from openai import OpenAI
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
ROBOT_IP = "192.168.8.187"  # change to your robot's IP address

USE_API = "--api" in sys.argv
if USE_API:
    load_dotenv()  # reads OPENAI_API_KEY from the .env file
    client = OpenAI()
    LLM_MODEL = "gpt-6.1-sol"
    VLM_MODEL = "gpt-6.1-sol"
    TTS_MODEL = "gpt-4o-mini-tts"  # text-to-speech: the robot speaks
else:
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    LLM_MODEL = "llama3.2"  # Meta Llama, text only: demo 1
    VLM_MODEL = "gemma3:4b"  # Google Gemma, text + images: demos 2 and 3
    TTS_MODEL = None  # the free models have no voice
SPEAK = TTS_MODEL is not None

# Stay well inside the SDK safety limits (pitch/roll ±40°, head-body yaw ±65°).
YAW_LIMIT = 45
TILT_LIMIT = 25

LOOK_DOWN_PITCH = 20  # degrees; head pose that points the camera at the build. Tune for your table.


# ---------------------------------------------------------------- model

def ask(model, messages):
    """Send a chat request and return the model's text reply."""
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def image_content(frame, text):
    """Message content with one camera frame (BGR image) plus a text prompt."""
    return images_content([frame], text)


def images_content(frames, text):
    """Message content with several images (BGR) plus a text prompt, images in order."""
    content = [{"type": "text", "text": text}]
    for frame in frames:
        h, w = frame.shape[:2]
        if w > 640:  # smaller image = faster and cheaper
            frame = cv2.resize(frame, (640, h * 640 // w))
        _, jpg = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(jpg.tobytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


_voice = None


def laptop_voice():
    """Start the laptop voice on first use. Returns a function that says text out loud,
    or None (text only) if no voice can start, so the app keeps running."""
    global _voice
    if _voice is None:
        try:
            if sys.platform == "win32":
                # Windows speech (SAPI) directly: pyttsx3 on Windows goes silent
                # after its first runAndWait()
                import comtypes.client
                sapi = comtypes.client.CreateObject("SAPI.SpVoice")
                _voice = lambda text: sapi.Speak(text)  # blocks until done
            else:
                import pyttsx3
                engine = pyttsx3.init()

                def _voice(text):
                    engine.say(text)
                    engine.runAndWait()
        except Exception as error:
            print(f"(no laptop voice: {error})")
            _voice = False
    return _voice


def speak(mini, text):
    """Say text out loud on the robot's speaker (only with --api)."""
    if not SPEAK:
        # robotic-like laptop voice (cringe), on the laptop speaker
        voice = laptop_voice()
        if voice:
            voice(text)
        return
    path = os.path.abspath("last_speech.wav")
    print(f"Playing .wav file located: {path}...")
    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL, voice="alloy", input=text, response_format="wav"
    ) as response:
        response.stream_to_file(path)
    mini.media.play_sound(path)  # uploads the file to the robot and plays it
    # play_sound returns as soon as playback starts, and the next sentence would cut this
    # one off: wait until it has finished
    time.sleep(wav_seconds(path) + 0.3)


def wav_seconds(path):
    """Length of a WAV file. Uses the file size, because streamed WAVs (like OpenAI's)
    often have a placeholder length in their header."""
    with wave.open(path) as w:
        bytes_per_second = w.getframerate() * w.getnchannels() * w.getsampwidth()
    return max(0.0, (os.path.getsize(path) - 44) / bytes_per_second)  # 44 = WAV header size


# ---------------------------------------------------------------- robot

def connect_robot(media=True):
    """Connect to Reachy Mini and wake it up. Use as `with connect_robot() as mini:`.

    media=False skips the camera and speaker (faster to connect).
    """
    if ROBOT_IP == "<robot address>":
        sys.exit("Set ROBOT_IP at the top of common.py to your robot's IP address (see SETUP.md).")
    backend = "default" if media else "no_media"
    mini = ReachyMini(host=ROBOT_IP, connection_mode="network", media_backend=backend)
    mini.enable_motors()  # motors start switched off: the robot ignores moves until this
    move_head(mini)  # lift the head from the sleep pose so the camera looks forward
    move_antennas(mini, 0, 0)
    return mini


def clamp(value, limit):
    return max(-limit, min(limit, value))


def move_head(mini, yaw=0, pitch=0, roll=0, duration=0.5):
    """Move the head. Degrees: yaw + = turn left, pitch + = look down, roll + = tilt."""
    yaw, pitch, roll = clamp(yaw, YAW_LIMIT), clamp(pitch, TILT_LIMIT), clamp(roll, TILT_LIMIT)
    pose = create_head_pose(yaw=yaw, pitch=pitch, roll=roll, degrees=True)
    goto(mini, head=pose, duration=duration)


def move_antennas(mini, right, left, duration=0.3):
    """Move the antennas. Radians, 0 = straight up."""
    goto(mini, antennas=[right, left], duration=duration)


def goto(mini, **target):
    """goto_target that survives the robot occasionally rejecting a move
    (e.g. "time value is out of range [0,1]"): retry once, then skip that gesture."""
    for attempt in range(2):
        try:
            mini.goto_target(**target)
            return
        except Exception as error:
            if attempt == 1:
                print(f"  (robot skipped a move: {error})")


def nod(mini):
    move_head(mini, pitch=15, duration=0.3)
    move_head(mini, duration=0.3)


def wiggle_antennas(mini):
    move_antennas(mini, 0.6, -0.6)
    move_antennas(mini, -0.6, 0.6)
    move_antennas(mini, 0, 0)


def get_frame(mini):
    """Grab one camera frame as a BGR image (numpy array)."""
    for _ in range(80):  # the camera stream can take a few seconds to start
        frame = mini.media.get_frame()
        if frame is not None:
            return frame
        time.sleep(0.25)
    raise RuntimeError("No camera frame after 20 s")
