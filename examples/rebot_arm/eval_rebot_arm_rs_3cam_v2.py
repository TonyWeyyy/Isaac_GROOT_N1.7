#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop GR00T N1.7 evaluation client for the Seeed reBot B601-RS arm.

3-camera version (wrist Orbbec Gemini 2 + front/side USB RGB) with:
- Non-blocking background episode recording (3-camera mp4 + actions/states/latency)
- Low-lag precision trajectory tracking + arm-synchronized gripper alignment
- Optional server-side Real-Time Chunking (RTC) via ``--use-rtc``
- Live inference latency & control-jitter monitoring

Data flow:
    B601-RS joints/cameras -> GR00T observation -> GPU policy server (ZeroMQ)
    -> 16-step action chunk -> latency-aligned boundary blend -> light EMA
    -> velocity/accel-limited trajectory + phase-aligned gripper
    -> LeRobot robot.send_action()

Safety:
    - Preview mode (no --execute) prints one prediction without sending actions.
    - Ctrl+C: smooth return to startup pose; second Ctrl+C: immediate torque off.
    - Command limits validated before any motion; driver applies physical limits.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import importlib
import inspect
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from gr00t.policy.server_client import PolicyClient
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot_robot_seeed_b601 import SeeedB601RSFollower, SeeedB601RSFollowerConfig

# Allow running directly as `python examples/rebot_arm/eval_rebot_arm_rs_3cam_v2.py`
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from rebot_control_utils import (  # noqa: E402
    ACTION_HORIZON,
    ACTION_SMOOTHING_ALPHA,
    ACTION_SMOOTHING_MAX_DELTA,
    CAMERA_FOURCC,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_KEYS,
    CAMERA_WIDTH,
    CHUNK_BLEND_ARM_DISAGREEMENT_SCALE,
    CHUNK_BLEND_DECAY_STEPS,
    CHUNK_BLEND_MAX_WEIGHT,
    CONTROL_HZ,
    GRIPPER_DELAY_STEPS,
    JOINT_KEYS,
    POLICY_LATENCY_STEPS,
    POLICY_TIMEOUT_MS,
    RETURN_ARM_SPEED,
    RETURN_CONTROL_HZ,
    RETURN_GRIPPER_SPEED,
    RETURN_HOLD_S,
    TRAJECTORY_TRACKING_GAIN,
    AsyncEpisodeRecorder,
    action_dict,
    blend_overlapping_actions,
    build_policy_observation,
    decode_action_chunk,
    encode_prev_action_for_rtc,
    limit_action_trajectory,
    make_gripper_delay_buffer,
    observation_state,
    parse_camera_source,
    physical_state_to_command_state,
    smooth_action_segment,
    validate_command_state,
)


def _load_orbbec_config_class():
    """Find the Orbbec camera config class across possible fork layouts."""
    candidates = (
        ("lerobot.cameras.orbbec_color", "OrbbecColorCameraConfig"),
        ("lerobot.cameras.orbbec_color", "OrbbecCameraConfig"),
        ("lerobot.cameras.orbbec", "OrbbecCameraConfig"),
        ("lerobot.cameras.orbbec", "OrbbecColorCameraConfig"),
    )
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "Could not find an Orbbec camera config class in lerobot.cameras. "
        "Check which orbbec module/class your fork provides."
    )


def _make_wrist_camera_config(serial: str):
    """Build the wrist camera config, dropping kwargs the class does not accept."""
    cls = _load_orbbec_config_class()
    kwargs = {
        "serial_number_or_name": serial,
        "width": CAMERA_WIDTH,
        "height": CAMERA_HEIGHT,
        "fps": CAMERA_FPS,
        "color_mode": "rgb",
        "rotation": 0,
        "warmup_s": 5,
    }
    accepted = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = set(kwargs) - set(filtered)
    if dropped:
        print(f"NOTE: wrist camera config class {cls.__name__} does not accept: {sorted(dropped)}")
    return cls(**filtered)


class UnsafeReturnStateError(RuntimeError):
    """Raised after an unsafe return state causes an immediate torque-disable request."""


class AsyncPolicyWorker:
    """Compute the next action chunk in a background thread over core PolicyClient."""

    def __init__(self, host: str, port: int, timeout_ms: int, api_token: str | None = None) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gr00t-policy")
        self.client: PolicyClient | None = None

    def _infer(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
        if self.client is None:
            self.client = PolicyClient(
                host=self.host,
                port=self.port,
                timeout_ms=self.timeout_ms,
                api_token=self.api_token,
                strict=False,
            )
        t0 = time.monotonic()
        action_chunk, info = self.client.get_action(observation, options=options)
        latency_ms = (time.monotonic() - t0) * 1000.0
        return action_chunk, info, latency_ms

    def submit(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> Future[tuple[dict[str, np.ndarray], dict[str, Any], float]]:
        return self.executor.submit(self._infer, observation, options)

    def close(self) -> None:
        def close_in_worker() -> None:
            if self.client is not None:
                self.client.close()
                self.client = None

        self.executor.submit(close_in_worker).result()
        self.executor.shutdown(wait=True, cancel_futures=True)


def _grab_camera_frames_nonblocking(robot: SeeedB601RSFollower) -> dict[str, np.ndarray]:
    """Read latest buffered frames from cameras without blocking."""
    frames: dict[str, np.ndarray] = {}
    for cam_name in CAMERA_KEYS:
        cam = robot.cameras.get(cam_name)
        if cam is None:
            continue
        try:
            frame = cam.async_read()
        except Exception:
            continue
        if frame is not None:
            frames[cam_name] = np.asarray(frame)
    return frames


def return_to_start_pose(
    robot: SeeedB601RSFollower,
    start_physical_state: np.ndarray,
    *,
    control_hz: float,
    max_arm_speed: float,
    max_gripper_speed: float,
    hold_s: float,
) -> None:
    """Return smoothly to the startup pose while keeping torque enabled."""
    if not robot.is_connected:
        raise RuntimeError("Cannot return home because the robot is not connected")

    try:
        current_observation = robot.get_observation()
        current_command = validate_command_state(
            physical_state_to_command_state(observation_state(current_observation)),
            field_name="current return command",
        )
        target_command = validate_command_state(
            physical_state_to_command_state(start_physical_state),
            field_name="startup return command",
        )
    except (KeyError, TypeError, ValueError) as state_error:
        message = f"Unsafe return state; skipping return motion: {state_error}"
        try:
            robot.disable_torque()
        except Exception as disable_error:
            message += f"; immediate torque-disable request also failed: {disable_error}"
        else:
            message += "; immediate torque disable was requested"
        raise UnsafeReturnStateError(message) from state_error

    delta = target_command - current_command
    duration_arm = float(np.max(1.5 * np.abs(delta[:6]) / max_arm_speed))
    duration_gripper = float(1.5 * abs(delta[6]) / max_gripper_speed)
    duration_s = max(duration_arm, duration_gripper, 1.0)
    steps = max(2, int(np.ceil(duration_s * control_hz)))
    period = 1.0 / control_hz

    print(
        f"\nReturning safely to the startup pose: {steps} frames, "
        f"approximately {duration_s:.2f} seconds."
    )
    for index in range(1, steps + 1):
        tick = time.monotonic()
        u = index / steps
        blend = 3.0 * u * u - 2.0 * u * u * u
        command = current_command + blend * delta
        robot.send_action(action_dict(command))
        precise_sleep(max(period - (time.monotonic() - tick), 0.0))

    hold_steps = int(np.ceil(hold_s * control_hz))
    for _ in range(hold_steps):
        tick = time.monotonic()
        robot.send_action(action_dict(target_command))
        precise_sleep(max(period - (time.monotonic() - tick), 0.0))

    print("Safe return completed.")


def make_robot(args: argparse.Namespace) -> SeeedB601RSFollower:
    """Create the B601-RS and three cameras (wrist Orbbec + front/side USB)."""
    cameras = {
        "wrist": _make_wrist_camera_config(args.wrist_camera),
        "front": OpenCVCameraConfig(
            index_or_path=parse_camera_source(args.front_camera),
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            fourcc=CAMERA_FOURCC,
        ),
        "side": OpenCVCameraConfig(
            index_or_path=parse_camera_source(args.side_camera),
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            fourcc=CAMERA_FOURCC,
        ),
    }
    config = SeeedB601RSFollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        can_adapter="socketcan",
        cameras=cameras,
        max_relative_target=None,
    )
    return SeeedB601RSFollower(config)


def safe_disconnect(robot: SeeedB601RSFollower) -> None:
    """Disable torque and release fully or partially connected hardware."""
    if robot.is_connected:
        try:
            robot.disconnect()
            print("Robot disconnected; torque disabled.")
            return
        except Exception as exc:
            print(f"WARNING: normal robot disconnect failed ({exc}); forcing torque off.")

    bus = getattr(robot, "bus", None)
    if bus is not None:
        try:
            bus.disable_all()
        except Exception as exc:
            print(f"WARNING: could not disable all motors: {exc}")
        for motor in getattr(robot, "motors", {}).values():
            try:
                motor.close()
            except Exception:
                pass
        try:
            bus.close()
        except Exception:
            pass
        robot.bus = None

    for camera in getattr(robot, "cameras", {}).values():
        if camera.is_connected:
            try:
                camera.disconnect()
            except Exception:
                pass
    print("Robot resources released; torque-disable was requested.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a GR00T N1.7 policy on a Seeed reBot B601-RS arm (3 cameras)."
    )
    parser.add_argument("--robot-port", default="can0")
    parser.add_argument("--robot-id", default="follower1")
    parser.add_argument(
        "--wrist-camera",
        default="AY6B95F002P",
        help="Orbbec Gemini 2 serial number",
    )
    parser.add_argument("--front-camera", default="/dev/video0")
    parser.add_argument("--side-camera", default="/dev/video6")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5555)
    parser.add_argument("--api-token", default=None)
    parser.add_argument(
        "--instruction",
        default="Pick up the red cube and put it into the box",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=25.0,
        help="Maximum policy run time; default 25 s, 0 runs until Ctrl+C.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send commands. Without this flag, one prediction is printed only.",
    )
    parser.add_argument(
        "--record-dir",
        default="",
        help="If set, save 3-camera mp4 + actions/states/latency log asynchronously to this dir.",
    )
    # Control & precision tuning knobs (all have safe validated defaults)
    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument("--latency-steps", type=int, default=POLICY_LATENCY_STEPS)
    parser.add_argument("--ema-alpha", type=float, default=ACTION_SMOOTHING_ALPHA)
    parser.add_argument("--chunk-blend-weight", type=float, default=CHUNK_BLEND_MAX_WEIGHT)
    parser.add_argument("--tracking-gain", type=float, default=TRAJECTORY_TRACKING_GAIN)
    parser.add_argument(
        "--gripper-delay-steps",
        type=int,
        default=GRIPPER_DELAY_STEPS,
        help="Delay gripper commands by N control steps to stay in phase with arm tracking lag.",
    )
    parser.add_argument(
        "--use-rtc",
        action="store_true",
        help="Enable server-side Real-Time Chunking (RTC) inpainting using previous chunk.",
    )
    parser.add_argument("--rtc-ramp-rate", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.duration_s < 0:
        parser.error("--duration-s must be non-negative (0 means unlimited)")
    if args.action_horizon < 1:
        parser.error("--action-horizon must be >= 1")
    if not 0 <= args.latency_steps < args.action_horizon:
        parser.error("--latency-steps must satisfy 0 <= latency_steps < action_horizon")
    if not 0.0 < args.ema_alpha <= 1.0:
        parser.error("--ema-alpha must be in (0, 1]")
    if not 0.0 <= args.chunk_blend_weight < 1.0:
        parser.error("--chunk-blend-weight must be in [0, 1)")
    if not 0.0 < args.tracking_gain <= 1.0:
        parser.error("--tracking-gain must be in (0, 1]")
    if args.gripper_delay_steps < 0:
        parser.error("--gripper-delay-steps must be >= 0")
    return args


def _print_timing_summary(
    latencies_ms: list[float],
    tick_overruns: int,
    executed_actions: int,
    budget_ms: float,
) -> None:
    if not latencies_ms:
        return
    arr = np.asarray(latencies_ms, dtype=np.float64)
    over_budget = int(np.sum(arr > budget_ms))
    print(
        f"\nInference latency (n={len(arr)}): "
        f"p50={np.percentile(arr, 50):.1f} ms, "
        f"p95={np.percentile(arr, 95):.1f} ms, "
        f"max={np.max(arr):.1f} ms "
        f"(async budget={budget_ms:.1f} ms, over_budget={over_budget})"
    )
    print(f"Control ticks executed={executed_actions}, tick_overruns(>33.3ms)={tick_overruns}")


def main(argv: list[str] | None = None) -> int:
    """Connect the server and robot, then run receding-horizon control."""
    args = parse_args(argv)
    async_budget_ms = (args.action_horizon - args.latency_steps) / CONTROL_HZ * 1000.0
    effective_blend_weight = min(args.chunk_blend_weight, 0.25) if args.use_rtc else args.chunk_blend_weight

    print(
        f"Control: {CONTROL_HZ:g} Hz, horizon={args.action_horizon}, "
        f"latency_steps={args.latency_steps} (async budget={async_budget_ms:.1f} ms), "
        f"ema_alpha={args.ema_alpha}, blend_weight={effective_blend_weight}, "
        f"tracking_gain={args.tracking_gain}, gripper_delay={args.gripper_delay_steps}, "
        f"rtc={args.use_rtc}"
    )

    client = PolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=POLICY_TIMEOUT_MS,
        api_token=args.api_token,
        strict=False,
    )
    async_policy = AsyncPolicyWorker(
        args.policy_host, args.policy_port, POLICY_TIMEOUT_MS, api_token=args.api_token
    )
    robot: SeeedB601RSFollower | None = None
    startup_state: np.ndarray | None = None
    recorder: AsyncEpisodeRecorder | None = None
    inference_latencies_ms: list[float] = []
    tick_overruns = 0
    executed_actions = 0

    try:
        print(f"Checking policy server at tcp://{args.policy_host}:{args.policy_port} ...")
        if not client.ping():
            raise ConnectionError(
                f"Policy server at tcp://{args.policy_host}:{args.policy_port} did not respond to ping"
            )
        print("Policy server is ready.")

        robot = make_robot(args)
        robot.connect()
        print("Robot and cameras connected.")

        if args.record_dir:
            recorder = AsyncEpisodeRecorder(
                args.record_dir, instruction=args.instruction, control_hz=CONTROL_HZ
            )
            recorder.open(list(robot.cameras.keys()))
            print(f"Async recording enabled -> {args.record_dir}")

        observation: dict[str, Any] | None = None
        for _ in range(3):
            observation = robot.get_observation()
        assert observation is not None
        candidate_startup_state = observation_state(observation).copy()
        validate_command_state(
            physical_state_to_command_state(candidate_startup_state),
            field_name="startup command",
        )
        startup_state = candidate_startup_state

        preview_observation = observation
        if not args.execute:
            robot.disable_torque()
            print(
                "PREVIEW MODE: captured a torque-enabled observation, then disabled "
                "motor torque; no policy action will be sent."
            )

        period = 1.0 / CONTROL_HZ
        target_action_count = (
            None if args.duration_s == 0 else int(round(args.duration_s * CONTROL_HZ))
        )
        first_chunk = True
        pending_future: Future[tuple[dict[str, np.ndarray], dict[str, Any], float]] | None = None
        pending_observation_step: int | None = None
        initial_command_state = physical_state_to_command_state(startup_state)
        previous_filtered_action = initial_command_state.copy()
        previous_sent_action = initial_command_state.copy()
        previous_sent_step = np.zeros(len(JOINT_KEYS), dtype=np.float32)
        gripper_delay_buf = make_gripper_delay_buffer(
            float(initial_command_state[6]), args.gripper_delay_steps
        )
        previous_chunk_actions: np.ndarray | None = None
        previous_chunk_observation_step: int | None = None

        while target_action_count is None or executed_actions < target_action_count:
            if first_chunk:
                observation = preview_observation if not args.execute else robot.get_observation()
                policy_observation = build_policy_observation(observation, args.instruction)
                chunk_observation_step = executed_actions
                t0 = time.monotonic()
                action_chunk, _ = client.get_action(policy_observation)
                latency_ms = (time.monotonic() - t0) * 1000.0
                first_chunk = False
            else:
                if pending_future is None or pending_observation_step is None:
                    raise RuntimeError("Next policy chunk was not prefetched")
                action_chunk, _, latency_ms = pending_future.result()
                chunk_observation_step = pending_observation_step
                pending_future = None
                pending_observation_step = None

            inference_latencies_ms.append(latency_ms)
            if recorder is not None:
                recorder.record_inference_latency(latency_ms)
            if args.execute and latency_ms > async_budget_ms:
                print(
                    f"WARNING: inference took {latency_ms:.1f} ms > async budget "
                    f"{async_budget_ms:.1f} ms (consider TensorRT or fewer denoising steps)"
                )

            actions = decode_action_chunk(action_chunk)
            action_start = executed_actions - chunk_observation_step
            if action_start < 0:
                raise RuntimeError(
                    "Policy observation is newer than the current control step: "
                    f"observation_step={chunk_observation_step}, execution_step={executed_actions}"
                )
            if not args.execute:
                print(f"first action (latency={latency_ms:.1f} ms):", action_dict(actions[action_start]))
                print("Preview complete. Re-run with --execute to control the arm.")
                break

            required_actions = action_start + args.action_horizon
            if len(actions) < required_actions:
                raise ValueError(
                    f"Policy returned only {len(actions)} steps, but "
                    f"{required_actions} are required for latency-aligned execution"
                )

            raw_segment = actions[action_start : action_start + args.action_horizon].copy()
            planned_segment = blend_overlapping_actions(
                raw_segment,
                execution_start_step=executed_actions,
                previous_chunk=previous_chunk_actions,
                previous_observation_step=previous_chunk_observation_step,
                max_old_weight=effective_blend_weight,
                arm_disagreement_scale=CHUNK_BLEND_ARM_DISAGREEMENT_SCALE,
                decay_steps=CHUNK_BLEND_DECAY_STEPS,
            )

            previous_chunk_actions = actions
            previous_chunk_observation_step = chunk_observation_step

            if target_action_count is not None:
                remaining = target_action_count - executed_actions
                planned_segment = planned_segment[:remaining]

            filtered_segment, previous_filtered_action = smooth_action_segment(
                planned_segment,
                previous_filtered_action,
                arm_alpha=args.ema_alpha,
                max_delta=ACTION_SMOOTHING_MAX_DELTA,
            )
            segment, previous_sent_action, previous_sent_step = limit_action_trajectory(
                filtered_segment,
                previous_sent_action,
                previous_sent_step,
                tracking_gain=args.tracking_gain,
                gripper_delay_buffer=gripper_delay_buf,
            )
            need_next_chunk = (
                target_action_count is None
                or executed_actions + len(segment) < target_action_count
            )

            for local_step, action_to_send in enumerate(segment):
                tick = time.monotonic()
                step_frames: dict[str, np.ndarray] | None = None
                step_state_cmd: np.ndarray | None = None

                if need_next_chunk and local_step == args.latency_steps:
                    obs_now = robot.get_observation()
                    pending_observation_step = executed_actions
                    policy_observation = build_policy_observation(obs_now, args.instruction)
                    rtc_options = None
                    if (
                        args.use_rtc
                        and previous_chunk_actions is not None
                        and previous_chunk_observation_step is not None
                    ):
                        elapsed_since_prev_obs = executed_actions - previous_chunk_observation_step
                        overlap = len(previous_chunk_actions) - elapsed_since_prev_obs
                        remaining_in_segment = len(segment) - local_step
                        frozen = min(remaining_in_segment, overlap)
                        if 0 < overlap <= len(previous_chunk_actions) and frozen >= 0:
                            rtc_options = {
                                "prev_action": encode_prev_action_for_rtc(previous_chunk_actions),
                                "rtc_overlap_steps": int(overlap),
                                "rtc_frozen_steps": int(frozen),
                                "rtc_ramp_rate": float(args.rtc_ramp_rate),
                            }
                    pending_future = async_policy.submit(policy_observation, options=rtc_options)
                    # Reuse freshly read observation frames & state for recording (no double read!)
                    if recorder is not None:
                        step_frames = {
                            k: np.asarray(obs_now[k]) for k in CAMERA_KEYS if k in obs_now
                        }
                        step_state_cmd = physical_state_to_command_state(observation_state(obs_now))

                robot.send_action(action_dict(action_to_send))

                if recorder is not None:
                    if step_frames is None:
                        step_frames = _grab_camera_frames_nonblocking(robot)
                    recorder.record_step(step_frames, action_to_send, state=step_state_cmd)

                executed_actions += 1
                elapsed = time.monotonic() - tick
                if elapsed > period:
                    tick_overruns += 1
                precise_sleep(max(period - elapsed, 0.0))

            if need_next_chunk and pending_future is None:
                raise RuntimeError("Policy prefetch was not submitted during the action segment")

        if args.execute:
            assert startup_state is not None
            return_to_start_pose(
                robot,
                startup_state,
                control_hz=RETURN_CONTROL_HZ,
                max_arm_speed=RETURN_ARM_SPEED,
                max_gripper_speed=RETURN_GRIPPER_SPEED,
                hold_s=RETURN_HOLD_S,
            )

        _print_timing_summary(inference_latencies_ms, tick_overruns, executed_actions, async_budget_ms)
        print("\nEvaluation finished.")
        return 0
    except UnsafeReturnStateError as unsafe_return_error:
        print(f"\nEvaluation stopped: {unsafe_return_error}")
        return 1
    except KeyboardInterrupt:
        print("\nCtrl+C received; no new policy actions will be requested.")
        _print_timing_summary(inference_latencies_ms, tick_overruns, executed_actions, async_budget_ms)
        if args.execute and robot is not None and robot.is_connected and startup_state is not None:
            print("Returning to the startup pose. Press Ctrl+C again to disable torque immediately.")
            try:
                return_to_start_pose(
                    robot,
                    startup_state,
                    control_hz=RETURN_CONTROL_HZ,
                    max_arm_speed=RETURN_ARM_SPEED,
                    max_gripper_speed=RETURN_GRIPPER_SPEED,
                    hold_s=RETURN_HOLD_S,
                )
            except UnsafeReturnStateError as unsafe_return_error:
                print(f"WARNING: {unsafe_return_error}")
            except KeyboardInterrupt:
                print("\nSecond Ctrl+C received; aborting return and requesting torque disable.")
        return 130
    except Exception as execution_error:
        print(f"\nEvaluation failed: {execution_error}")
        if args.execute and robot is not None and robot.is_connected and startup_state is not None:
            print("Robot is still connected; attempting a safe return first.")
            try:
                return_to_start_pose(
                    robot,
                    startup_state,
                    control_hz=RETURN_CONTROL_HZ,
                    max_arm_speed=RETURN_ARM_SPEED,
                    max_gripper_speed=RETURN_GRIPPER_SPEED,
                    hold_s=RETURN_HOLD_S,
                )
            except Exception as return_error:
                print(f"WARNING: safe return after failure also failed: {return_error}")
        raise
    finally:
        if recorder is not None:
            recorder.close()
        try:
            if robot is not None:
                safe_disconnect(robot)
        finally:
            try:
                async_policy.close()
            finally:
                client.close()


if __name__ == "__main__":
    sys.exit(main())
