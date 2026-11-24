# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardware-independent control, safety, and async recording utilities for Seeed reBot B601-RS.

Keeping pure NumPy trajectory math and validation separate from LeRobot / CAN / camera
imports allows full unit-test coverage on CPU without physical hardware attached.
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np


# This order must exactly match the observation.state/action names in dataset info.json.
JOINT_KEYS: tuple[str, ...] = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_yaw.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
CAMERA_KEYS: tuple[str, ...] = ("wrist", "front", "side")

# The driver reports physical motor angles, while dataset actions are command angles.
# Must match SeeedB601RSFollowerConfig.joint_directions exactly.
JOINT_DIRECTIONS = np.array([1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 6.0], dtype=np.float32)

# Hard safety limits in dataset command space (driver enforces physical limits as secondary guard).
COMMAND_LOWER = np.array([-145.0, -5.0, -200.0, -90.0, -90.0, -90.0, -2.0], dtype=np.float32)
COMMAND_UPPER = np.array([145.0, 170.0, 1.0, 80.0, 90.0, 90.0, 45.0], dtype=np.float32)

# Validated B601-RS real-time control defaults.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"
POLICY_TIMEOUT_MS = 120_000
ACTION_HORIZON = 8
POLICY_LATENCY_STEPS = 4
CONTROL_HZ = 30.0

# Low-lag precision smoothing parameters:
# - Higher EMA alpha (0.65 vs legacy 0.4) cuts phase lag from ~1.5 steps to ~0.5 steps.
# - Lower chunk blend weight (0.50 vs legacy 0.85) with exponential decay lets fresh
#   wrist-camera alignment corrections take authority rapidly during the final 1-2 cm grasp.
# - Responsive tracking gain (0.75 vs legacy 0.35) eliminates steady-state crawl while
#   strictly preserving hardware velocity (TRAJECTORY_MAX_STEP) and acceleration
#   (TRAJECTORY_MAX_STEP_CHANGE) limits.
ACTION_SMOOTHING_ALPHA = 0.65
ACTION_SMOOTHING_MAX_DELTA = 2.5
CHUNK_BLEND_MAX_WEIGHT = 0.50
CHUNK_BLEND_ARM_DISAGREEMENT_SCALE = 4.0
CHUNK_BLEND_DECAY_STEPS = 4
TRAJECTORY_TRACKING_GAIN = 0.75
TRAJECTORY_MAX_STEP = np.array([1.2, 1.0, 1.0, 1.0, 1.2, 1.2], dtype=np.float32)
TRAJECTORY_MAX_STEP_CHANGE = np.array(
    [0.12, 0.10, 0.10, 0.10, 0.12, 0.12],
    dtype=np.float32,
)

# Gripper-to-arm phase alignment:
# Delays gripper commands by a small integer number of control ticks so the gripper
# closes when the velocity/accel-limited arm actually reaches the target pose, instead
# of closing ~150 ms prematurely in mid-air.
GRIPPER_DELAY_STEPS = 2
GRIPPER_CLOSE_GATE_DELTA = 3.0
GRIPPER_CLOSE_GATE_ARM_ERR_DEG = 1.5
GRIPPER_CLOSE_GATE_MAX_HOLD_STEPS = 3

RETURN_CONTROL_HZ = 30.0
RETURN_ARM_SPEED = 20.0
RETURN_GRIPPER_SPEED = 8.0
RETURN_HOLD_S = 2.0


def parse_camera_source(value: str) -> int | Path:
    """Parse a numeric camera index or return a /dev/videoX/file path."""
    return int(value) if value.isdecimal() else Path(value)


def observation_state(robot_observation: dict[str, Any]) -> np.ndarray:
    """Extract the 7-D physical state from a LeRobot observation in fixed joint order."""
    return np.asarray([robot_observation[key] for key in JOINT_KEYS], dtype=np.float32)


def physical_state_to_command_state(physical_state: np.ndarray) -> np.ndarray:
    """Convert physical motor angles to dataset action command space."""
    return np.asarray(physical_state, dtype=np.float32) / JOINT_DIRECTIONS


def action_dict(action: np.ndarray) -> dict[str, float]:
    """Convert a 7-D NumPy action to a LeRobot ``send_action()`` dictionary."""
    return {key: float(value) for key, value in zip(JOINT_KEYS, action, strict=True)}


def validate_command_state(command: np.ndarray, *, field_name: str) -> np.ndarray:
    """Validate a 7-D command-space state before it can drive the robot."""
    state = np.asarray(command, dtype=np.float32)
    expected_shape = (len(JOINT_KEYS),)
    if state.shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {expected_shape}, got {state.shape}")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{field_name} contains non-finite values: {state}")

    outside_limits = (state < COMMAND_LOWER) | (state > COMMAND_UPPER)
    if np.any(outside_limits):
        details = ", ".join(
            f"{key}={float(value):.3f} not in [{float(lower):.3f}, {float(upper):.3f}]"
            for key, value, lower, upper, outside in zip(
                JOINT_KEYS, state, COMMAND_LOWER, COMMAND_UPPER, outside_limits, strict=True
            )
            if outside
        )
        raise ValueError(f"{field_name} is outside B601-RS command limits: {details}")
    return state


def build_policy_observation(robot_observation: dict[str, Any], instruction: str) -> dict[str, Any]:
    """Convert a raw LeRobot observation to the GR00T Policy API format."""
    missing = [key for key in (*JOINT_KEYS, *CAMERA_KEYS) if key not in robot_observation]
    if missing:
        raise KeyError(f"Robot observation is missing keys: {missing}")

    state = np.asarray([robot_observation[key] for key in JOINT_KEYS], dtype=np.float32)
    if not np.all(np.isfinite(state)):
        raise ValueError(f"Robot returned a non-finite state: {state}")
    if np.all(np.abs(state) < 1e-6):
        raise RuntimeError("All seven joint readings are zero; no valid motor feedback received")

    videos: dict[str, np.ndarray] = {}
    for key in CAMERA_KEYS:
        frame = np.asarray(robot_observation[key])
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Camera {key!r} returned invalid shape {frame.shape}; expected HxWx3")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        videos[key] = np.ascontiguousarray(frame[None, None, ...])

    return {
        "video": videos,
        "state": {
            "single_arm": state[None, None, :6],
            "gripper": state[None, None, 6:7],
        },
        "language": {
            "annotation.human.task_description": [[instruction]],
        },
    }


def decode_action_chunk(action_chunk: dict[str, Any]) -> np.ndarray:
    """Validate, concatenate, and safely clamp a server action chunk."""
    missing = {"single_arm", "gripper"} - set(action_chunk)
    if missing:
        raise KeyError(f"Policy action is missing modalities: {sorted(missing)}")

    arm = np.asarray(action_chunk["single_arm"], dtype=np.float32)
    gripper = np.asarray(action_chunk["gripper"], dtype=np.float32)
    if arm.ndim != 3 or arm.shape[0] != 1 or arm.shape[2] != 6:
        raise ValueError(f"single_arm action must have shape (1, T, 6), got {arm.shape}")
    if gripper.ndim != 3 or gripper.shape[:2] != arm.shape[:2] or gripper.shape[2] != 1:
        raise ValueError(f"gripper action must have shape (1, T, 1), got {gripper.shape}")

    actions = np.concatenate((arm[0], gripper[0]), axis=1)
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy returned NaN or infinity; refusing to control the robot")

    clipped = np.clip(actions, COMMAND_LOWER, COMMAND_UPPER)
    if not np.array_equal(actions, clipped):
        clipped_by_joint = np.count_nonzero(actions != clipped, axis=0)
        count = int(clipped_by_joint.sum())
        details = ", ".join(
            f"{key}={int(joint_count)}"
            for key, joint_count in zip(JOINT_KEYS, clipped_by_joint, strict=True)
            if joint_count
        )
        print(f"WARNING: clipped {count} predicted values to B601-RS command limits ({details})")
    return clipped


def encode_prev_action_for_rtc(chunk_actions: np.ndarray) -> dict[str, np.ndarray]:
    """Pack a decoded (T, 7) physical command chunk into GR00T Policy RTC format."""
    arr = np.asarray(chunk_actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != len(JOINT_KEYS):
        raise ValueError(f"chunk_actions must have shape (T, 7), got {arr.shape}")
    return {
        "single_arm": np.ascontiguousarray(arr[None, :, :6]),
        "gripper": np.ascontiguousarray(arr[None, :, 6:7]),
    }


def blend_overlapping_actions(
    new_segment: np.ndarray,
    *,
    execution_start_step: int,
    previous_chunk: np.ndarray | None,
    previous_observation_step: int | None,
    max_old_weight: float = CHUNK_BLEND_MAX_WEIGHT,
    arm_disagreement_scale: float = CHUNK_BLEND_ARM_DISAGREEMENT_SCALE,
    decay_steps: int = CHUNK_BLEND_DECAY_STEPS,
) -> np.ndarray:
    """Blend overlapping action chunks by absolute control time with fast boundary decay."""
    new_segment = np.asarray(new_segment, dtype=np.float32)
    if new_segment.ndim != 2 or new_segment.shape[1] != len(JOINT_KEYS):
        raise ValueError(f"new_segment must have shape (T, 7), got {new_segment.shape}")
    if not 0.0 <= max_old_weight < 1.0:
        raise ValueError(f"max_old_weight must be in [0, 1), got {max_old_weight}")
    if arm_disagreement_scale <= 0.0:
        raise ValueError("chunk disagreement scale must be positive")
    if previous_chunk is None or previous_observation_step is None or max_old_weight == 0.0:
        return new_segment.copy()

    previous_chunk = np.asarray(previous_chunk, dtype=np.float32)
    if previous_chunk.ndim != 2 or previous_chunk.shape[1] != len(JOINT_KEYS):
        raise ValueError(f"previous_chunk must have shape (T, 7), got {previous_chunk.shape}")

    previous_start = execution_start_step - previous_observation_step
    if previous_start < 0 or previous_start >= len(previous_chunk):
        return new_segment.copy()

    overlap = min(len(new_segment), len(previous_chunk) - previous_start)
    if overlap <= 0:
        return new_segment.copy()

    old = previous_chunk[previous_start : previous_start + overlap]
    new = new_segment[:overlap]

    # Blend primarily across the first ``decay_steps`` ticks at the chunk boundary so
    # fresh visual feedback takes over quickly rather than dragging stale predictions.
    effective_blend = min(overlap, max(1, decay_steps))
    age_weight = np.zeros(overlap, dtype=np.float32)
    if effective_blend == 1:
        age_weight[0] = max_old_weight
    else:
        age_weight[:effective_blend] = max_old_weight * np.linspace(
            1.0, 0.0, effective_blend, dtype=np.float32
        )

    agreement_weight = np.exp(-np.abs(new[:, :6] - old[:, :6]) / arm_disagreement_scale)
    old_weight = age_weight[:, None] * agreement_weight

    blended = new_segment.copy()
    blended[:overlap, :6] = old_weight * old[:, :6] + (1.0 - old_weight) * new[:, :6]
    return blended


def smooth_action_segment(
    requested_segment: np.ndarray,
    previous_filtered_action: np.ndarray,
    *,
    arm_alpha: float = ACTION_SMOOTHING_ALPHA,
    max_delta: float = ACTION_SMOOTHING_MAX_DELTA,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply delta clamping and light EMA to six arm joints; pass the gripper through."""
    requested_segment = np.asarray(requested_segment, dtype=np.float32)
    previous = np.asarray(previous_filtered_action, dtype=np.float32).copy()
    if requested_segment.ndim != 2 or requested_segment.shape[1] != len(JOINT_KEYS):
        raise ValueError(f"requested_segment must have shape (T, 7), got {requested_segment.shape}")
    if previous.shape != (len(JOINT_KEYS),):
        raise ValueError(f"previous_filtered_action must have shape (7,), got {previous.shape}")
    if not 0.0 < arm_alpha <= 1.0:
        raise ValueError(f"arm_alpha must be in (0, 1], got {arm_alpha}")
    if max_delta <= 0.0:
        raise ValueError(f"max_delta must be positive, got {max_delta}")

    smoothed = np.empty_like(requested_segment)
    for index, requested in enumerate(requested_segment):
        clamped_target = previous + np.clip(requested - previous, -max_delta, max_delta)
        previous = arm_alpha * clamped_target + (1.0 - arm_alpha) * previous
        previous[6] = requested[6]
        smoothed[index] = previous
    return smoothed, previous.copy()


def limit_action_trajectory(
    filtered_segment: np.ndarray,
    previous_action: np.ndarray,
    previous_step: np.ndarray,
    *,
    tracking_gain: float = TRAJECTORY_TRACKING_GAIN,
    gripper_delay_buffer: deque[float] | None = None,
    gripper_gate_delta: float = GRIPPER_CLOSE_GATE_DELTA,
    gripper_gate_arm_err: float = GRIPPER_CLOSE_GATE_ARM_ERR_DEG,
    gripper_gate_max_hold: int = GRIPPER_CLOSE_GATE_MAX_HOLD_STEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert filtered targets into a velocity- and acceleration-continuous trajectory.

    Optionally phase-aligns the gripper stream with arm tracking lag via
    ``gripper_delay_buffer`` and briefly holds abrupt gripper transitions while the
    arm is still converging on the final 1-2 cm target pose.
    """
    targets = np.asarray(filtered_segment, dtype=np.float32)
    position = np.asarray(previous_action, dtype=np.float32).copy()
    step = np.asarray(previous_step, dtype=np.float32).copy()
    expected_shape = (len(JOINT_KEYS),)
    if targets.ndim != 2 or targets.shape[1] != len(JOINT_KEYS):
        raise ValueError(f"filtered_segment must have shape (T, 7), got {targets.shape}")
    if position.shape != expected_shape:
        raise ValueError(f"previous_action must have shape (7,), got {position.shape}")
    if step.shape != expected_shape:
        raise ValueError(f"previous_step must have shape (7,), got {step.shape}")
    if not 0.0 < tracking_gain <= 1.0:
        raise ValueError(f"tracking_gain must be in (0, 1], got {tracking_gain}")
    if not np.all(np.isfinite(targets)):
        raise ValueError("filtered_segment contains NaN or infinity")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(step)):
        raise ValueError("trajectory state contains NaN or infinity")

    commands = np.empty_like(targets)
    consecutive_gripper_holds = 0

    for index, target in enumerate(targets):
        error = target[:6] - position[:6]
        braking_step = np.sqrt(2.0 * TRAJECTORY_MAX_STEP_CHANGE * np.abs(error))
        proportional_step = tracking_gain * np.abs(error)
        desired_step = np.sign(error) * np.minimum(
            TRAJECTORY_MAX_STEP, np.minimum(braking_step, proportional_step)
        )
        step[:6] += np.clip(
            desired_step - step[:6],
            -TRAJECTORY_MAX_STEP_CHANGE,
            TRAJECTORY_MAX_STEP_CHANGE,
        )
        clipped = position.copy()
        clipped[:6] = np.clip(position[:6] + step[:6], COMMAND_LOWER[:6], COMMAND_UPPER[:6])

        raw_gripper = float(np.clip(target[6], COMMAND_LOWER[6], COMMAND_UPPER[6]))
        if gripper_delay_buffer is not None and len(gripper_delay_buffer) > 0:
            gripper_delay_buffer.append(raw_gripper)
            aligned_gripper = float(gripper_delay_buffer.popleft())
        else:
            aligned_gripper = raw_gripper

        # Pose-arrival gate: if a sharp gripper transition is requested while the arm
        # is still more than ``gripper_gate_arm_err`` degrees from the target pose,
        # hold the previous gripper value briefly (bounded by ``gripper_gate_max_hold``).
        post_step_arm_err = float(np.max(np.abs(target[:6] - clipped[:6])))
        gripper_jump = abs(aligned_gripper - float(position[6]))
        if (
            gripper_gate_max_hold > 0
            and gripper_jump >= gripper_gate_delta
            and post_step_arm_err > gripper_gate_arm_err
            and consecutive_gripper_holds < gripper_gate_max_hold
        ):
            clipped[6] = float(position[6])
            consecutive_gripper_holds += 1
        else:
            clipped[6] = aligned_gripper
            consecutive_gripper_holds = 0

        step[:6] = clipped[:6] - position[:6]
        step[6] = 0.0
        position = clipped
        commands[index] = position

    return commands, position.copy(), step.copy()


def make_gripper_delay_buffer(initial_gripper_cmd: float, delay_steps: int) -> deque[float]:
    """Create a FIFO delay buffer seeded with the robot's initial gripper command."""
    if delay_steps < 0:
        raise ValueError(f"delay_steps must be >= 0, got {delay_steps}")
    val = float(np.clip(initial_gripper_cmd, COMMAND_LOWER[6], COMMAND_UPPER[6]))
    return deque([val] * delay_steps)


class AsyncEpisodeRecorder:
    """Non-blocking episode recorder (3-camera mp4 + actions/states/latency log).

    Video encoding runs on a dedicated background thread backed by a bounded queue so
    disk I/O and MPEG-4 compression never stall the 30 Hz robot control loop.
    """

    def __init__(
        self,
        record_dir: str,
        *,
        instruction: str = "",
        control_hz: float = CONTROL_HZ,
        max_queue_size: int = 64,
    ) -> None:
        self.dir = Path(record_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stamp = time.strftime("%Y%m%d_%H%M%S")
        self.instruction = instruction
        self.control_hz = control_hz
        self.actions: list[list[float]] = []
        self.states: list[list[float]] = []
        self.inference_latencies_ms: list[float] = []
        self.dropped_frames = 0
        self._writers: dict[str, Any] = {}
        self._queue: queue.Queue[dict[str, np.ndarray] | None] = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self._closed = False

    def open(self, camera_names: list[str] | tuple[str, ...]) -> None:
        import cv2

        for cam_name in CAMERA_KEYS:
            if cam_name in camera_names:
                self._writers[cam_name] = cv2.VideoWriter(
                    str(self.dir / f"{self.stamp}_{cam_name}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    int(self.control_hz),
                    (CAMERA_WIDTH, CAMERA_HEIGHT),
                )
        self._thread = threading.Thread(
            target=self._writer_loop, name="b601-video-writer", daemon=True
        )
        self._thread.start()

    def _writer_loop(self) -> None:
        import cv2

        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            try:
                for cam_name, frame in item.items():
                    writer = self._writers.get(cam_name)
                    if writer is None:
                        continue
                    arr = np.asarray(frame)
                    if arr.ndim == 3 and arr.shape[-1] == 3:
                        writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            finally:
                self._queue.task_done()

    def record_step(
        self,
        frames: dict[str, np.ndarray],
        action: np.ndarray,
        state: np.ndarray | None = None,
    ) -> None:
        """Enqueue a control-step snapshot without blocking the real-time loop."""
        self.actions.append([float(v) for v in action])
        if state is not None:
            self.states.append([float(v) for v in state])
        if self._thread is not None and frames:
            payload = {k: np.asarray(v).copy() for k, v in frames.items() if k in self._writers}
            if payload:
                try:
                    self._queue.put_nowait(payload)
                except queue.Full:
                    self.dropped_frames += 1

    def record_inference_latency(self, latency_ms: float) -> None:
        self.inference_latencies_ms.append(float(latency_ms))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            # Drain queued frames before shutting down writer thread.
            self._queue.put(None)
            self._thread.join(timeout=15.0)
            self._thread = None
        for writer in self._writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self._writers.clear()

        if self.actions:
            np.save(self.dir / f"{self.stamp}_actions.npy", np.asarray(self.actions, dtype=np.float32))
        if self.states:
            np.save(self.dir / f"{self.stamp}_states.npy", np.asarray(self.states, dtype=np.float32))
        if self.actions or self.states:
            meta = {
                "joint_keys": list(JOINT_KEYS),
                "action_count": len(self.actions),
                "state_count": len(self.states),
                "control_hz": self.control_hz,
                "instruction": self.instruction,
                "dropped_video_frames": self.dropped_frames,
                "inference_latencies_ms": self.inference_latencies_ms,
            }
            with open(self.dir / f"{self.stamp}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)
        print(
            f"Recording saved to {self.dir} (prefix {self.stamp}, "
            f"actions={len(self.actions)}, states={len(self.states)}, "
            f"dropped_frames={self.dropped_frames})"
        )
