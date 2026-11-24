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

"""Regression tests for Gr00tPolicy RTC/warmup and PolicyClient socket cleanup."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.types import ModalityConfig
from gr00t.policy.server_client import PolicyClient


def _build_mock_policy():
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.device = torch.device("cpu")
    mock_model.dtype = torch.bfloat16
    mock_model.get_action = MagicMock(
        return_value=BatchFeature(data={"action_pred": torch.zeros(1, 16, 132)})
    )

    modality_configs = {
        "new_embodiment": {
            "video": ModalityConfig(delta_indices=[0], modality_keys=["wrist", "front", "side"]),
            "state": ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"]),
            "action": ModalityConfig(
                delta_indices=list(range(16)), modality_keys=["single_arm", "gripper"]
            ),
            "language": ModalityConfig(
                delta_indices=[0], modality_keys=["annotation.human.task_description"]
            ),
        }
    }

    mock_processor = MagicMock()
    mock_processor.get_modality_configs.return_value = modality_configs
    mock_processor.max_action_dim = 132
    mock_processor.max_action_horizon = 40
    mock_processor.state_action_processor = MagicMock()
    mock_processor.state_action_processor.norm_params = {
        "new_embodiment": {
            "state": {
                "single_arm": {"dim": np.array(6)},
                "gripper": {"dim": np.array(1)},
            }
        }
    }
    mock_processor.state_action_processor.apply_action.side_effect = lambda prev_b, tag, state: {
        "single_arm": prev_b["single_arm"],
        "gripper": prev_b["gripper"],
    }
    mock_processor.side_effect = lambda msgs: {"state": torch.zeros(1, 132)}
    mock_processor.collator = MagicMock(
        side_effect=lambda items: {"inputs": {"state": torch.zeros(1, 1, 132)}}
    )
    mock_processor.decode_action = MagicMock(
        return_value={
            "single_arm": np.zeros((1, 16, 6), dtype=np.float32),
            "gripper": np.zeros((1, 16, 1), dtype=np.float32),
        }
    )

    with (
        patch("gr00t.policy.gr00t_policy.AutoModel") as MockAutoModel,
        patch("gr00t.policy.gr00t_policy.AutoProcessor") as MockAutoProcessor,
        patch("pathlib.Path.is_dir", return_value=False),
        patch("pathlib.Path.exists", return_value=True),
    ):
        MockAutoModel.from_pretrained.return_value = mock_model
        MockAutoProcessor.from_pretrained.return_value = mock_processor
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        policy = Gr00tPolicy(
            embodiment_tag="new_embodiment",
            model_path="/fake/ckpt",
            device="cpu",
        )
    return policy, mock_model, mock_processor


def test_gr00t_policy_forwards_rtc_prev_action_and_options():
    policy, mock_model, _ = _build_mock_policy()
    obs = {
        "video": {
            k: np.zeros((1, 1, 32, 32, 3), dtype=np.uint8) for k in ("wrist", "front", "side")
        },
        "state": {
            "single_arm": np.zeros((1, 1, 6), dtype=np.float32),
            "gripper": np.zeros((1, 1, 1), dtype=np.float32),
        },
        "language": {"annotation.human.task_description": [["pick cube"]]},
    }
    prev_action = {
        "single_arm": np.ones((1, 16, 6), dtype=np.float32),
        "gripper": np.ones((1, 16, 1), dtype=np.float32),
    }
    rtc_options = {
        "prev_action": prev_action,
        "rtc_overlap_steps": 8,
        "rtc_frozen_steps": 4,
        "rtc_ramp_rate": 5.0,
    }

    action, _ = policy.get_action(obs, options=rtc_options)
    assert "single_arm" in action

    call_kwargs = mock_model.get_action.call_args.kwargs
    assert "action" in call_kwargs["inputs"]
    assert call_kwargs["inputs"]["action"].shape == (1, 40, 132)
    assert call_kwargs["options"] == {
        "action_horizon": 16,
        "rtc_overlap_steps": 8,
        "rtc_frozen_steps": 4,
        "rtc_ramp_rate": 5.0,
    }


def test_gr00t_policy_warmup_runs_single_dummy_inference():
    policy, mock_model, _ = _build_mock_policy()
    policy.warmup(image_size=(64, 64))
    assert mock_model.get_action.call_count == 1


def test_policy_client_reinit_closes_previous_socket():
    """Calling _init_socket() on reconnect must close the previous ZMQ socket (no FD leak)."""
    client = PolicyClient(host="127.0.0.1", port=59999, timeout_ms=50)
    try:
        first_sock = client.socket
        assert first_sock is not None and not first_sock.closed
        client._init_socket()
        assert first_sock.closed
        assert client.socket is not first_sock
        assert not client.socket.closed
    finally:
        client.close()
