"""Robot outputs: head pose, antennas and speech.

Movement helpers follow the course demos (common.py), with the same safety limits.
"""

from __future__ import annotations

import os
import tempfile

from reachy_mini.utils import create_head_pose

from . import config

# Stay well inside the SDK safety limits (pitch/roll ±40°, head-body yaw ±65°).
YAW_LIMIT = 45
TILT_LIMIT = 25


def clamp(value, limit):
    return max(-limit, min(limit, value))


class Robot:
    """Thin wrapper around ReachyMini with the coach's expressive behaviors."""

    def __init__(self, mini):
        self.mini = mini
        self._tts_client = None
        self._local_tts = None
        if config.TTS_MODEL:
            from openai import OpenAI

            self._tts_client = OpenAI(api_key=config.API_KEY)
        else:
            try:
                import pyttsx3

                self._local_tts = pyttsx3.init()
            except Exception as exc:  # no audio device, e.g. running headless
                print(f"Local voice unavailable ({exc}); speech will only be printed.")

    # ---------------------------------------------------------------- motion

    def move_head(self, yaw=0, pitch=0, roll=0, duration=0.5):
        """Degrees: yaw + = turn left, pitch + = look down, roll + = tilt."""
        yaw, pitch, roll = clamp(yaw, YAW_LIMIT), clamp(pitch, TILT_LIMIT), clamp(roll, TILT_LIMIT)
        pose = create_head_pose(yaw=yaw, pitch=pitch, roll=roll, degrees=True)
        self.mini.goto_target(head=pose, duration=duration)

    def move_antennas(self, right, left, duration=0.3):
        """Radians, 0 = straight up."""
        self.mini.goto_target(antennas=[right, left], duration=duration)

    def wake(self):
        self.mini.enable_motors()
        self.look_at_student()
        self.move_antennas(0, 0)

    def look_at_student(self):
        self.move_head(pitch=0)

    def look_at_build(self):
        self.move_head(pitch=config.LOOK_AT_BUILD_PITCH)

    def thinking(self):
        """Tilt the head while the VLM checks the step (as in demo 1)."""
        self.move_head(pitch=config.LOOK_AT_BUILD_PITCH, roll=12, duration=0.4)

    def react_correct(self):
        """Look up at the student, nod, antennas up and out: 'Yes!'"""
        self.move_head(pitch=0, duration=0.4)
        self.move_antennas(0.5, -0.5, duration=0.25)
        self.move_head(pitch=12, duration=0.25)
        self.move_head(pitch=0, duration=0.25)
        self.move_antennas(0, 0)

    def react_mistake(self):
        """Lean toward the build with a curious head tilt, antennas lowered: 'Hmm, look here.'"""
        self.move_antennas(-0.4, 0.4, duration=0.4)
        self.move_head(pitch=config.LOOK_AT_BUILD_PITCH + 3, roll=-15, duration=0.5)

    def celebrate(self):
        self.look_at_student()
        for _ in range(2):
            self.move_antennas(0.6, -0.6)
            self.move_antennas(-0.6, 0.6)
        self.move_antennas(0, 0)
        self.move_head(yaw=15, duration=0.3)
        self.move_head(yaw=-15, duration=0.3)
        self.move_head(duration=0.3)

    def acknowledge(self):
        """Quick antenna twitch: 'I saw your thumbs-up.'"""
        self.move_antennas(0.3, -0.3, duration=0.15)
        self.move_antennas(0, 0, duration=0.15)

    # ---------------------------------------------------------------- speech

    def say(self, text: str):
        print(f"reachy> {text}")
        if self._tts_client is not None:
            path = os.path.join(tempfile.gettempdir(), "lego_coach_speech.wav")
            with self._tts_client.audio.speech.with_streaming_response.create(
                model=config.TTS_MODEL, voice="alloy", input=text, response_format="wav"
            ) as response:
                response.stream_to_file(path)
            self.mini.media.play_sound(path)
        elif self._local_tts is not None:
            self._local_tts.say(text)
            self._local_tts.runAndWait()
