"""LEGO Coach: a Reachy Mini app that checks a student's LEGO build step by step.

Loop for each step:
  1. The robot reads the step instruction out loud and looks down at the build area.
  2. The student builds, then holds up a thumbs-up (MediaPipe Gesture Recognizer).
     The experimenter can also press "Check now" in the control panel.
  3. The robot waits until the student's hands are out of view, then takes a photo.
  4. Bricks are found (MediaPipe / color detector) and a VLM checks the step against
     the instruction, plus a reference photo in the "reference_image" condition.
  5. The robot reacts: nod and antennas up for a correct step, a curious head tilt and
     a hint for a mistake. Every check is logged to lego_coach_data/trials.csv.

Run on your laptop:   python -m lego_coach.main   (then open http://localhost:8042)
Or install it as an app on the robot's dashboard (see README.md).
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field

import cv2
from pydantic import BaseModel
from reachy_mini import ReachyMini, ReachyMiniApp

from . import config
from .behaviors import Robot
from .perception import BrickDetector, HandWatcher, annotate, summarize
from .trial_log import TrialLog, counts_match
from .verifier import StepVerifier, list_builds, load_build, reference_path

LOOP_HZ = 10


@dataclass
class CoachState:
    build_id: str = "tower"
    step: int = 0
    condition: str = config.DEFAULT_CONDITION
    participant: str = "P01"
    ground_truth: str = ""
    auto_advance: bool = True  # turn off for staged trials so the experimenter controls the step
    use_brick_hints: bool = config.USE_BRICK_HINTS
    phase: str = "idle"  # idle | instruct | watch | wait_clear | check | done
    message: str = "Press Start in the control panel."
    n_hands: int = 0
    thumbs_up: bool = False
    last_verdict: dict = field(default_factory=dict)
    error: str = ""


class ConfigIn(BaseModel):
    build_id: str | None = None
    condition: str | None = None
    participant: str | None = None
    auto_advance: bool | None = None
    use_brick_hints: bool | None = None


class LabelIn(BaseModel):
    label: str


class LegoCoach(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8042"
    request_media_backend: str | None = None  # needs the camera

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        robot = Robot(reachy_mini)
        hands = HandWatcher()
        bricks = BrickDetector()
        verifier = StepVerifier()
        log = TrialLog(config.DATA_DIR)

        state = CoachState()
        lock = threading.Lock()
        requests: set[str] = set()  # actions requested from the control panel
        latest = {"jpeg": None, "frame": None, "hand_boxes": []}
        build = {"data": load_build(state.build_id)}

        def request(action: str):
            with lock:
                requests.add(action)

        def take(action: str) -> bool:
            with lock:
                if action in requests:
                    requests.discard(action)
                    return True
                return False

        def has_reference(step: int) -> bool:
            return reference_path(build["data"], step).exists()

        # ------------------------------------------------------------ control panel API
        app = self.settings_app

        @app.get("/api/state")
        def get_state():
            with lock:
                data = asdict(state)
            steps = build["data"]["steps"]
            data.update(
                builds=list_builds(), conditions=list(config.CONDITIONS),
                labels=list(config.GROUND_TRUTH_LABELS), n_steps=len(steps),
                instruction=steps[min(state.step, len(steps) - 1)]["instruction"],
                has_reference=has_reference(min(state.step, len(steps) - 1)),
                model=config.VLM_MODEL, detector=bricks.source,
            )
            return data

        @app.get("/api/frame.jpg")
        def get_frame():
            from fastapi import Response

            jpeg = latest["jpeg"]
            if jpeg is None:
                return Response(status_code=204)
            return Response(content=jpeg, media_type="image/jpeg")

        @app.get("/api/trials")
        def get_trials():
            return log.recent(15)

        @app.post("/api/config")
        def set_config(cfg: ConfigIn):
            with lock:
                if cfg.build_id and cfg.build_id != state.build_id:
                    build["data"] = load_build(cfg.build_id)
                    state.build_id, state.step = cfg.build_id, 0
                if cfg.condition in config.CONDITIONS:
                    state.condition = cfg.condition
                if cfg.participant is not None:
                    state.participant = cfg.participant.strip() or "P01"
                if cfg.auto_advance is not None:
                    state.auto_advance = cfg.auto_advance
                if cfg.use_brick_hints is not None:
                    state.use_brick_hints = cfg.use_brick_hints
            return {"ok": True}

        @app.post("/api/ground_truth")
        def set_ground_truth(label: LabelIn):
            with lock:
                state.ground_truth = label.label if label.label in config.GROUND_TRUTH_LABELS else ""
            return {"ok": True}

        def make_action_route(action: str):
            def handler():
                request(action)
                return {"ok": True}

            app.post(f"/api/{action}")(handler)

        for action in ("start", "stop", "check", "next", "prev", "repeat", "capture_reference"):
            make_action_route(action)

        # ------------------------------------------------------------ helpers
        def set_phase(phase: str, message: str):
            with lock:
                state.phase, state.message = phase, message

        def go_to_step(step: int):
            n = len(build["data"]["steps"])
            with lock:
                state.step = max(0, min(step, n - 1))
                state.ground_truth = ""
                requests.discard("check")  # a stale "Check now" shouldn't fire on the new step
            set_phase("instruct", f"Step {state.step + 1}")

        def grab():
            frame = reachy_mini.media.get_frame()
            if frame is None:
                return None, None
            hand_state = hands.observe(frame)
            preview = annotate(frame, [], hand_state.boxes)
            cv2.putText(preview, f"hands: {hand_state.n_hands}" + ("  THUMBS UP" if hand_state.thumbs_up else ""),
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            ok, jpg = cv2.imencode(".jpg", preview)
            with lock:
                latest.update(frame=frame, hand_boxes=hand_state.boxes, jpeg=jpg.tobytes() if ok else None)
                state.n_hands, state.thumbs_up = hand_state.n_hands, hand_state.thumbs_up
            return frame, hand_state

        def check_step(trigger: str, t_signal: float, clear_wait: float):
            frame = latest["frame"]
            with lock:
                snap = CoachState(**asdict(state))
            if snap.condition == "reference_image" and not has_reference(snap.step):
                with lock:
                    state.error = (f"No reference photo for step {snap.step + 1}. Build the step correctly, "
                                   "press 'Save reference photo', then check again.")
                set_phase("watch", "Waiting for a thumbs-up (or Check now)")
                return
            set_phase("check", "Checking the step...")
            robot.thinking()

            found = bricks.detect(frame, latest["hand_boxes"])
            summary = summarize(found, frame.shape)
            step_def = build["data"]["steps"][snap.step]

            try:
                verdict = verifier.verify(frame, build["data"], snap.step, snap.condition,
                                          summary if snap.use_brick_hints else None)
            except Exception as exc:  # model offline, missing reference photo, ...
                with lock:
                    state.error = str(exc)
                robot.say("Hmm, I had trouble checking that. Can you ask my helper?")
                set_phase("watch", f"Error: {exc}")
                return

            if verdict.correct:
                robot.react_correct()
            else:
                robot.react_mistake()
            response_latency = time.perf_counter() - t_signal  # signal -> start of spoken feedback
            robot.say(verdict.feedback)

            trial_id = log.next_trial_id()
            image = log.save_images(trial_id, frame, annotate(frame, found))
            log.write(dict(
                trial_id=trial_id, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                participant=snap.participant, build=snap.build_id, step=snap.step + 1,
                condition=snap.condition, model=config.VLM_MODEL, ground_truth=snap.ground_truth,
                predicted_correct=verdict.correct, predicted_error_type=verdict.error_type,
                detector_counts_match=counts_match(found, step_def.get("expected")),
                n_bricks_detected=len(found), brick_detector=bricks.source, trigger=trigger,
                hands_clear_wait_s=f"{clear_wait:.2f}", vlm_latency_s=f"{verdict.vlm_latency_s:.2f}",
                response_latency_s=f"{response_latency:.2f}", parse_ok=verdict.parse_ok,
                feedback=verdict.feedback, raw_reply=verdict.raw.replace("\n", " "), image=image,
            ))
            with lock:
                state.error = ""
                state.ground_truth = ""
                state.last_verdict = dict(correct=verdict.correct, error_type=verdict.error_type,
                                          feedback=verdict.feedback, response_latency_s=round(response_latency, 2),
                                          vlm_latency_s=round(verdict.vlm_latency_s, 2), trial_id=trial_id)

            last_step = snap.step >= len(build["data"]["steps"]) - 1
            if verdict.correct and snap.auto_advance:
                if last_step:
                    robot.say("You finished the whole build. Awesome work!")
                    robot.celebrate()
                    set_phase("done", "Build finished!")
                else:
                    go_to_step(snap.step + 1)
            else:
                robot.look_at_build()
                set_phase("watch", "Waiting for a thumbs-up (or Check now)")

        # ------------------------------------------------------------ main loop
        robot.wake()
        thumbs_since = None
        clear_since = None
        signal = {"t": 0.0, "trigger": ""}

        while not stop_event.is_set():
            t_loop = time.perf_counter()
            frame, hand_state = grab()

            if take("stop"):
                robot.look_at_student()
                set_phase("idle", "Stopped. Press Start to begin again.")
            if take("start"):
                go_to_step(0)
            if take("prev"):
                go_to_step(state.step - 1)
            if take("next"):
                go_to_step(state.step + 1)
            if take("repeat") and state.phase != "idle":
                set_phase("instruct", f"Step {state.step + 1}")
            if take("capture_reference") and frame is not None:
                path = reference_path(build["data"], state.step)
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), frame)
                with lock:
                    state.message = f"Saved reference photo for step {state.step + 1}"

            phase = state.phase
            if phase == "instruct":
                robot.look_at_student()
                robot.say(f"Step {state.step + 1}. {build['data']['steps'][state.step]['instruction']} "
                          "Give me a thumbs up when you're done.")
                robot.look_at_build()
                thumbs_since = None
                set_phase("watch", "Waiting for a thumbs-up (or Check now)")

            elif phase == "watch" and hand_state is not None:
                manual = take("check")
                if hand_state.thumbs_up:
                    thumbs_since = thumbs_since or time.perf_counter()
                else:
                    thumbs_since = None
                held = thumbs_since and time.perf_counter() - thumbs_since >= config.THUMBS_UP_HOLD_S
                if manual or held:
                    signal.update(t=time.perf_counter(), trigger="button" if manual else "thumbs_up")
                    robot.acknowledge()
                    if hand_state.n_hands:
                        robot.say("Let me check! Please move your hands away.")
                    clear_since = None
                    set_phase("wait_clear", "Waiting for hands to leave the view")

            elif phase == "wait_clear" and hand_state is not None:
                now = time.perf_counter()
                if hand_state.n_hands == 0:
                    clear_since = clear_since or now
                else:
                    clear_since = None
                cleared = clear_since and now - clear_since >= config.HANDS_CLEAR_S
                if cleared or now - signal["t"] >= config.HANDS_CLEAR_TIMEOUT_S:
                    check_step(signal["trigger"], signal["t"], now - signal["t"])

            time.sleep(max(0.0, 1 / LOOP_HZ - (time.perf_counter() - t_loop)))

        hands.close()
        robot.look_at_student()


if __name__ == "__main__":
    app = LegoCoach()
    try:
        app.wrapped_run(**({"host": config.ROBOT_IP} if config.ROBOT_IP else {}))
    except KeyboardInterrupt:
        app.stop()
