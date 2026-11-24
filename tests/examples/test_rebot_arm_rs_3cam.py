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

"""Unit tests for Seeed reBot B601-RS control, safety limits, gripper sync, and async recorder."""

from pathlib import Path
import sys

import numpy as np
import pytest


REBOT_ARM_DIR = str(Path(__file__).resolve().parents[2] / "examples" / "rebot_arm")
if REBOT_ARM_DIR not in sys.path:
    sys.path.insert(0, REBOT_ARM_DIR)

from rebot_control_utils import (  # noqa: E402
    COMMAND_LOWER,
    COMMAND_UPPER,
    JOINT_KEYS,
    TRAJECTORY_MAX_STEP,
    TRAJECTORY_MAX_STEP_CHANGE,
    AsyncEpisodeRecorder,
    blend_overlapping_actions,
    build_policy_observation,
    decode_action_chunk,
    encode_prev_action_for_rtc,
    limit_action_trajectory,
    make_gripper_delay_buffer,
    smooth_action_segment,
    validate_command_state,
)


def test_decode_action_chunk_clamps_to_hardware_limits_and_rejects_nan():
    raw = {
        "single_arm": np.array([[[200.0, -50.0, 50.0, -120.0, 120.0, -120.0]]], dtype=np.float32),
        "gripper": np.array([[[99.0]]], dtype=np.float32),
    }
    clipped = decode_action_chunk(raw)
    assert clipped.shape == (1, 7)
    assert np.all(clipped >= COMMAND_LOWER)
    assert np.all(clipped <= COMMAND_UPPER)

    bad = {
        "single_arm": np.array([[[np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
        "gripper": np.array([[[10.0]]], dtype=np.float32),
    }
    with pytest.raises(ValueError, match="NaN"):
        decode_action_chunk(bad)


def test_trajectory_limiter_enforces_velocity_and_accel_limits_and_converges_fast():
    """Hard velocity/accel bounds must never be violated, while small 1-2 deg errors converge cleanly."""
    start = np.array([0.0, 20.0, -40.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float32)
    step = np.zeros(7, dtype=np.float32)
    # Large step command to stress velocity & acceleration bounds
    large_targets = np.tile(
        np.array([30.0, 50.0, -10.0, 25.0, -25.0, 25.0, 10.0], dtype=np.float32), (20, 1)
    )
    cmds, _, _ = limit_action_trajectory(large_targets, start, step)
    deltas = np.diff(np.vstack([start[None, :], cmds]), axis=0)[:, :6]
    accels = np.diff(np.vstack([np.zeros((1, 6), dtype=np.float32), deltas]), axis=0)
    assert np.all(np.abs(deltas) <= TRAJECTORY_MAX_STEP + 1e-4)
    assert np.all(np.abs(accels) <= TRAJECTORY_MAX_STEP_CHANGE + 1e-4)
    assert np.all(cmds >= COMMAND_LOWER)
    assert np.all(cmds <= COMMAND_UPPER)

    # Fine 1.0 deg final-approach adjustment: must converge much faster than legacy gain=0.35
    fine_targets = np.tile(start + np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]), (8, 1))
    cmds_fast, _, _ = limit_action_trajectory(fine_targets, start, np.zeros(7, dtype=np.float32))
    cmds_slow, _, _ = limit_action_trajectory(
        fine_targets, start, np.zeros(7, dtype=np.float32), tracking_gain=0.35
    )
    err_fast = np.max(np.abs(fine_targets[-1, :6] - cmds_fast[-1, :6]))
    err_slow = np.max(np.abs(fine_targets[-1, :6] - cmds_slow[-1, :6]))
    assert err_fast < 0.05
    assert err_fast < err_slow


def test_gripper_delay_and_pose_arrival_gate_prevent_premature_close():
    """Gripper must stay phase-aligned with arm motion and wait while arm is far from target."""
    start = np.array([0.0, 20.0, -40.0, 0.0, 0.0, 0.0, 35.0], dtype=np.float32)  # open=35
    # Arm still needs to move 10 deg, while model already asks gripper to close to 5 at step 0
    targets = np.tile(
        np.array([10.0, 20.0, -40.0, 0.0, 0.0, 0.0, 5.0], dtype=np.float32),
        (8, 1),
    )
    delay_buf = make_gripper_delay_buffer(initial_gripper_cmd=35.0, delay_steps=2)
    cmds, _, _ = limit_action_trajectory(
        targets,
        start,
        np.zeros(7, dtype=np.float32),
        gripper_delay_buffer=delay_buf,
    )
    # During the first 2 delay steps + pose-arrival gate steps, gripper does NOT prematurely close.
    assert cmds[0, 6] == pytest.approx(35.0)
    assert cmds[1, 6] == pytest.approx(35.0)
    assert cmds[2, 6] == pytest.approx(35.0)  # held by pose-arrival gate (arm error > 1.5 deg)
    # After max_hold_steps (3), gripper is released so it never deadlocks.
    assert cmds[-1, 6] == pytest.approx(5.0)


def test_chunk_blending_and_rtc_encoding():
    prev = np.zeros((16, 7), dtype=np.float32)
    new_seg = np.ones((8, 7), dtype=np.float32) * 2.0
    blended = blend_overlapping_actions(
        new_seg,
        execution_start_step=8,
        previous_chunk=prev,
        previous_observation_step=0,
    )
    # Step 0 is blended toward prev, while tail steps (>= decay_steps) fully trust new chunk.
    assert blended[0, 0] < new_seg[0, 0]
    assert blended[-1, 0] == pytest.approx(new_seg[-1, 0])

    rtc_payload = encode_prev_action_for_rtc(prev)
    assert rtc_payload["single_arm"].shape == (1, 16, 6)
    assert rtc_payload["gripper"].shape == (1, 16, 1)


def test_build_policy_observation_and_validate_command_state():
    obs = {k: 1.0 for k in JOINT_KEYS}
    for cam in ("wrist", "front", "side"):
        obs[cam] = np.zeros((48, 64, 3), dtype=np.uint8)
    policy_obs = build_policy_observation(obs, "pick up red cube")
    assert policy_obs["video"]["wrist"].shape == (1, 1, 48, 64, 3)
    assert policy_obs["state"]["single_arm"].shape == (1, 1, 6)
    assert policy_obs["state"]["gripper"].shape == (1, 1, 1)

    with pytest.raises(ValueError, match="outside B601-RS command limits"):
        validate_command_state(np.array([999.0, 0, 0, 0, 0, 0, 0]), field_name="test")


def test_async_episode_recorder_saves_actions_states_and_meta(tmp_path):
    rec = AsyncEpisodeRecorder(str(tmp_path), instruction="pick cube", control_hz=30.0)
    # Test non-video metadata/state/action persistence without requiring cv2 codecs
    action = np.array([1.0, 2.0, -3.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float32)
    state = np.array([0.9, 1.9, -2.9, 0.0, 0.0, 0.0, 10.0], dtype=np.float32)
    rec.record_step({}, action, state=state)
    rec.record_inference_latency(42.5)
    rec.close()

    actions_files = list(tmp_path.glob("*_actions.npy"))
    states_files = list(tmp_path.glob("*_states.npy"))
    meta_files = list(tmp_path.glob("*_meta.json"))
    assert len(actions_files) == 1
    assert len(states_files) == 1
    assert len(meta_files) == 1
    assert np.load(actions_files[0]).shape == (1, 7)
    assert np.load(states_files[0]).shape == (1, 7)


def test_smooth_action_segment_keeps_gripper_unsmoothed():
    prev = np.zeros(7, dtype=np.float32)
    req = np.array([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0]], dtype=np.float32)
    smoothed, last = smooth_action_segment(req, prev, arm_alpha=0.5, max_delta=5.0)
    assert smoothed[0, 0] == pytest.approx(1.0)
    assert smoothed[0, 6] == pytest.approx(25.0)
    assert last[6] == pytest.approx(25.0)
