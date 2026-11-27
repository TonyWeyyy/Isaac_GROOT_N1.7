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

# Well-damped precision smoothing parameters (validated against B601-RS ~±0.8 deg
# diffusion chunk noise so the gripper stays rock-steady near the grasp target):
# - EMA alpha 0.40 + chunk blend weight 0.85 strongly attenuate 2-6 Hz stochastic
#   chunk-to-chunk jitter without introducing boundary steps.
# - Tracking gain 0.35 paired with hardware accel limits (TRAJECTORY_MAX_STEP_CHANGE)
#   gives critically damped convergence with zero overshoot / limit-cycle oscillation.
# - Slightly raised velocity ceiling (TRAJECTORY_MAX_STEP, <=42 deg/s = 0.73 rad/s)
#   speeds up large transit moves while keeping fine near-target motion identically damped.
ACTION_SMOOTHING_ALPHA = 0.40
ACTION_SMOOTHING_MAX_DELTA = 2.5
CHUNK_BLEND_MAX_WEIGHT = 0.85
CHUNK_BLEND_ARM_DISAGREEMENT_SCALE = 4.0
CHUNK_BLEND_DECAY_STEPS = 6
TRAJECTORY_TRACKING_GAIN = 0.35
TRAJECTORY_MAX_STEP = np.array([1.4, 1.2, 1.2, 1.2, 1.4, 1.4], dtype=np.float32)
TRAJECTORY_MAX_STEP_CHANGE = np.array(
    [0.12, 0.10, 0.10, 0.10, 0.12, 0.12],
    dtype=np.float32,
)

# Gripper phase alignment (off by default so learned grasp timing is never delayed;
# opt-in via --gripper-delay-steps if premature closing is observed).
GRIPPER_DELAY_STEPS = 0
GRIPPER_CLOSE_GATE_DELTA = 3.0
GRIPPER_CLOSE_GATE_ARM_ERR_DEG = 1.5
GRIPPER_CLOSE_GATE_MAX_HOLD_STEPS = 0

# Default state representation sent to the GR00T policy server:
# - "command": converts physical motor readings via physical_state_to_command_state()
#   to match datasets processed by align_dataset_command_space.py (true RELATIVE arm).
# - "physical": sends raw physical motor angles (legacy checkpoints).
DEFAULT_POLICY_STATE_SPACE = "command"

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


def build_policy_observation(
    robot_observation: dict[str, Any],
    instruction: str,
    *,
    allow_all_zero: bool = False,
    state_space: str = DEFAULT_POLICY_STATE_SPACE,
) -> dict[str, Any]:
    """Convert a raw LeRobot observation to the GR00T Policy API format.

    Args:
        robot_observation: Raw dict from ``SeeedB601RSFollower.get_observation()``.
        instruction: Natural-language task prompt.
        allow_all_zero: If True, permit an all-zero joint reading (only when motor
            CAN feedback has already been verified non-None at the driver level).
        state_space: ``"command"`` (converts physical motor angles into command space
            via ``physical_state_to_command_state`` so ``single_arm`` RELATIVE deltas
            share the same sign/scale as ``action``) or ``"physical"`` (raw physical
            motor angles for legacy checkpoints).
    """
    if state_space not in ("command", "physical"):
        raise ValueError(f"state_space must be 'command' or 'physical', got {state_space!r}")

    missing = [key for key in (*JOINT_KEYS, *CAMERA_KEYS) if key not in robot_observation]
    if missing:
        raise KeyError(f"Robot observation is missing keys: {missing}")

    physical_state = np.asarray([robot_observation[key] for key in JOINT_KEYS], dtype=np.float32)
    if not np.all(np.isfinite(physical_state)):
        raise ValueError(f"Robot returned a non-finite state: {physical_state}")
    if not allow_all_zero and np.all(np.abs(physical_state) < 1e-6):
        raise RuntimeError("All seven joint readings are zero; no valid motor feedback received")

    state = (
        physical_state_to_command_state(physical_state)
        if state_space == "command"
        else physical_state
    )

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


def _read_reference_frame_rgb(video_path: Path, frame_idx: int = 0) -> np.ndarray | None:
    """Read a single RGB frame from an MP4 (supports AV1 via PyAV/torchcodec and H264 via OpenCV)."""
    if not video_path.exists():
        return None
    # 1. PyAV (installed in lerobot env; reliably decodes software AV1)
    try:
        import av

        with av.open(str(video_path)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i == frame_idx:
                    return frame.to_ndarray(format="rgb24")
    except Exception:
        pass
    # 2. torchcodec (installed in Isaac-GR00T venv)
    try:
        from torchcodec.decoders import VideoDecoder

        dec = VideoDecoder(str(video_path), num_ffmpeg_threads=1)
        idx = min(max(0, frame_idx), len(dec) - 1)
        tensor = dec.get_frames_at([idx]).data[0]  # (C, H, W)
        return tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    except Exception:
        pass
    # 3. OpenCV fallback
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        cap.release()
        if ok and bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def compute_edge_alignment_score(live_rgb: np.ndarray, ref_rgb: np.ndarray) -> float:
    """Compute a [0, 1] Sobel edge-magnitude cosine similarity between two RGB frames.

    Invariant to global brightness changes while sensitive to camera bump/shift.
    """
    live = np.asarray(live_rgb, dtype=np.float32)
    ref = np.asarray(ref_rgb, dtype=np.float32)
    if live.shape[:2] != ref.shape[:2]:
        import cv2

        ref = cv2.resize(ref, (live.shape[1], live.shape[0]), interpolation=cv2.INTER_AREA)
    gray_l = 0.299 * live[..., 0] + 0.587 * live[..., 1] + 0.114 * live[..., 2]
    gray_r = 0.299 * ref[..., 0] + 0.587 * ref[..., 1] + 0.114 * ref[..., 2]

    def _edge_mag(g: np.ndarray) -> np.ndarray:
        gx = np.abs(g[:, 2:] - g[:, :-2])[1:-1, :]
        gy = np.abs(g[2:, :] - g[:-2, :])[:, 1:-1]
        return np.hypot(gx, gy)

    el = _edge_mag(gray_l).ravel()
    er = _edge_mag(gray_r).ravel()
    denom = float(np.linalg.norm(el) * np.linalg.norm(er))
    if denom < 1e-8:
        return 0.0
    return float(np.clip(np.dot(el, er) / denom, 0.0, 1.0))


def check_camera_alignment(
    live_frames_rgb: dict[str, np.ndarray],
    reference_dataset_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    episode_idx: int = 0,
    frame_idx: int = 0,
    warn_threshold: float = 0.45,
) -> dict[str, float]:
    """Compare live camera frames against training dataset reference frames at startup.

    Saves a 3-row visual diagnostic sheet (LIVE | TRAIN REF | 50/50 OVERLAY) if
    ``output_path`` is provided, and returns per-camera edge alignment scores in [0, 1].
    """
    ref_root = Path(reference_dataset_dir).expanduser()
    scores: dict[str, float] = {}
    columns: list[np.ndarray] = []

    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore[assignment]

    for cam in CAMERA_KEYS:
        if cam not in live_frames_rgb:
            continue
        video_path = (
            ref_root
            / "videos"
            / "chunk-000"
            / f"observation.images.{cam}"
            / f"episode_{episode_idx:06d}.mp4"
        )
        ref_rgb = _read_reference_frame_rgb(video_path, frame_idx=frame_idx)
        if ref_rgb is None:
            print(f"Camera alignment: skipped {cam} (could not read {video_path})")
            continue
        live_rgb = np.asarray(live_frames_rgb[cam], dtype=np.uint8)
        score = compute_edge_alignment_score(live_rgb, ref_rgb)
        scores[cam] = score

        status = "OK"
        if cam in ("front", "side") and score < warn_threshold:
            status = "WARNING: possible camera shift!"
        print(f"Camera alignment [{cam:5s}]: edge_score={score:.3f} ({status})")

        if cv2 is not None and output_path is not None:
            w, h = 320, 240
            l_bgr = cv2.resize(cv2.cvtColor(live_rgb, cv2.COLOR_RGB2BGR), (w, h))
            r_bgr = cv2.resize(cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR), (w, h))
            ov_bgr = cv2.addWeighted(l_bgr, 0.5, r_bgr, 0.5, 0.0)
            cv2.putText(l_bgr, f"LIVE: {cam}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(r_bgr, f"REF: {cam}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            color = (0, 255, 0) if (cam == "wrist" or score >= warn_threshold) else (0, 0, 255)
            cv2.putText(
                ov_bgr,
                f"OVERLAY {cam} ({score:.2f})",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
            columns.append(np.vstack([l_bgr, r_bgr, ov_bgr]))

    if cv2 is not None and output_path is not None and columns:
        out_file = Path(output_path).expanduser()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_file), np.hstack(columns))
        print(f"Camera alignment comparison saved -> {out_file}")

    return scores
