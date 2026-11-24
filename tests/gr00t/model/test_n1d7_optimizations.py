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

"""Regression tests for GR00T N1.7 model, backbone, processor, and data optimizations."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset, merge_statistics
from gr00t.data.dataset.sharded_single_step_dataset import (
    ShardedSingleStepDataset,
    extract_step_data,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    Gr00tN1d7Processor,
)
from gr00t.model.modules.qwen3_backbone import Qwen3Backbone


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "processor_config"


def test_processor_from_pretrained_honors_all_finetune_overrides():
    """from_pretrained must apply extra_augmentation_config, use_percentiles, fractional crop, etc."""
    mock_vlm = MagicMock()
    mock_vlm.tokenizer.padding_side = "left"
    extra_aug = {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}]}

    with patch(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor",
        return_value=mock_vlm,
    ) as mock_build:
        proc = Gr00tN1d7Processor.from_pretrained(
            FIXTURE_DIR,
            use_percentiles=False,
            extra_augmentation_config=extra_aug,
            letter_box_transform=True,
            shortest_image_edge=320,
            crop_fraction=0.9,
            image_crop_size=None,
            image_target_size=None,
            state_dropout_prob=0.0,
        )

    # Collator shares the processor's VLM processor instead of calling build_processor twice.
    assert mock_build.call_count == 1
    assert proc.collator.processor is proc.processor
    assert proc.use_percentiles is False
    assert proc.extra_augmentation_config == extra_aug
    assert proc.letter_box_transform is True
    assert proc.shortest_image_edge == 320
    assert proc.crop_fraction == pytest.approx(0.9)
    assert proc.image_crop_size is None
    assert proc.image_target_size is None
    assert proc.train_image_transform.mask_transforms is not None
    assert len(proc.train_image_transform.mask_transforms) == 1


def test_processor_does_not_mutate_global_embodiment_mapping():
    """Constructing Gr00tN1d7Processor with custom mapping must not mutate caller or global dict."""
    mock_vlm = MagicMock()
    mock_vlm.tokenizer.padding_side = "left"
    custom = {"custom_robot": 31}
    before_global = dict(EMBODIMENT_TAG_TO_PROJECTOR_INDEX)

    with patch(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor",
        return_value=mock_vlm,
    ):
        proc = Gr00tN1d7Processor.from_pretrained(FIXTURE_DIR)
        proc.embodiment_id_mapping["ephemeral_tag"] = 30

    assert "ephemeral_tag" not in EMBODIMENT_TAG_TO_PROJECTOR_INDEX
    assert EMBODIMENT_TAG_TO_PROJECTOR_INDEX == before_global
    assert custom == {"custom_robot": 31}


def test_relative_action_honors_use_percentiles():
    """When use_percentiles=True, relative_action normalization must use q01/q99, not raw min/max."""
    modality_configs = {
        "new_embodiment": {
            "state": ModalityConfig(delta_indices=[0], modality_keys=["arm"]),
            "action": ModalityConfig(
                delta_indices=[0, 1],
                modality_keys=["arm"],
                action_configs=[
                    ActionConfig(
                        rep=ActionRepresentation.RELATIVE,
                        type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT,
                    )
                ],
            ),
        }
    }
    stats = {
        "new_embodiment": {
            "state": {
                "arm": {
                    "min": [-10.0, -10.0],
                    "max": [10.0, 10.0],
                    "q01": [-1.0, -1.0],
                    "q99": [1.0, 1.0],
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                }
            },
            "action": {
                "arm": {
                    "min": [-10.0, -10.0],
                    "max": [10.0, 10.0],
                    "q01": [-1.0, -1.0],
                    "q99": [1.0, 1.0],
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                }
            },
            "relative_action": {
                "arm": {
                    "min": [[-50.0, -50.0], [-50.0, -50.0]],
                    "max": [[50.0, 50.0], [50.0, 50.0]],
                    "q01": [[-2.0, -2.0], [-2.0, -2.0]],
                    "q99": [[2.0, 2.0], [2.0, 2.0]],
                    "mean": [[0.0, 0.0], [0.0, 0.0]],
                    "std": [[1.0, 1.0], [1.0, 1.0]],
                }
            },
        }
    }

    proc_pct = StateActionProcessor(
        modality_configs=modality_configs,
        statistics=stats,
        use_percentiles=True,
        use_relative_action=True,
    )
    np.testing.assert_allclose(
        proc_pct.norm_params["new_embodiment"]["action"]["arm"]["min"],
        np.array([[-2.0, -2.0], [-2.0, -2.0]]),
    )
    np.testing.assert_allclose(
        proc_pct.norm_params["new_embodiment"]["action"]["arm"]["max"],
        np.array([[2.0, 2.0], [2.0, 2.0]]),
    )

    proc_raw = StateActionProcessor(
        modality_configs=modality_configs,
        statistics=stats,
        use_percentiles=False,
        use_relative_action=True,
    )
    np.testing.assert_allclose(
        proc_raw.norm_params["new_embodiment"]["action"]["arm"]["min"],
        np.array([[-50.0, -50.0], [-50.0, -50.0]]),
    )


def test_mean_std_action_not_clipped_to_unit_range():
    """Z-score normalized actions must not be clipped to [-1, 1] when clip_outliers=True."""
    modality_configs = {
        "new_embodiment": {
            "state": ModalityConfig(delta_indices=[0], modality_keys=["arm"]),
            "action": ModalityConfig(
                delta_indices=[0],
                modality_keys=["arm"],
                mean_std_embedding_keys=["arm"],
            ),
        }
    }
    stats = {
        "new_embodiment": {
            "state": {
                "arm": {
                    "min": [-1.0],
                    "max": [1.0],
                    "q01": [-1.0],
                    "q99": [1.0],
                    "mean": [0.0],
                    "std": [1.0],
                }
            },
            "action": {
                "arm": {
                    "min": [-1.0],
                    "max": [1.0],
                    "q01": [-1.0],
                    "q99": [1.0],
                    "mean": [0.0],
                    "std": [1.0],
                }
            },
        }
    }
    proc = StateActionProcessor(
        modality_configs=modality_configs, statistics=stats, clip_outliers=True
    )
    norm = proc.apply_action({"arm": np.array([[2.5]], dtype=np.float32)}, "new_embodiment")
    np.testing.assert_allclose(norm["arm"], np.array([[2.5]], dtype=np.float32))


def test_qwen3_backbone_forward_skips_lm_head_and_preserves_top_layer_train_mode():
    """Qwen3Backbone.forward must pass use_cache=False & logits_to_keep=slice(0, 0) and keep top layers in train()."""
    backbone = Qwen3Backbone.__new__(Qwen3Backbone)
    torch.nn.Module.__init__(backbone)

    hidden = torch.randn(1, 5, 16)
    layer0 = torch.nn.Linear(4, 4)
    layer1 = torch.nn.Linear(4, 4)
    lang_model = torch.nn.Module()
    lang_model.layers = torch.nn.ModuleList([layer0, layer1])
    visual = torch.nn.Linear(4, 4)
    lm_head = torch.nn.Linear(4, 16)

    call_log: list[dict] = []

    class _FakeQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = lang_model
            self.visual = visual
            self.lm_head = lm_head
            self.config = SimpleNamespace(image_token_id=99)

        def forward(self, **kwargs):
            call_log.append(kwargs)
            return SimpleNamespace(hidden_states=(hidden,))

    backbone.model = _FakeQwen()
    backbone.set_trainable_parameters(tune_llm=False, tune_visual=False, tune_top_llm_layers=1)

    backbone.train()
    vl_in = BatchFeature(
        data={
            "input_ids": torch.tensor([[1, 99, 99, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "pixel_values": torch.randn(2, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
    )
    out = backbone.forward(vl_in)

    assert len(call_log) == 1
    assert call_log[0]["output_hidden_states"] is True
    assert call_log[0]["use_cache"] is False
    assert call_log[0]["logits_to_keep"] == slice(0, 0)
    assert torch.equal(out["backbone_features"], hidden)
    # lm_head frozen, frozen lower layer in eval(), trainable top layer stays in train().
    assert not any(p.requires_grad for p in lm_head.parameters())
    assert not layer0.training
    assert layer1.training
    assert not visual.training


def test_extract_step_data_rejects_negative_index_without_padding():
    """Negative delta_indices must raise instead of silently wrapping via pandas iloc."""
    df = pd.DataFrame(
        {
            "video.cam": [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(10)],
            "state.arm": [np.array([float(i)], dtype=np.float32) for i in range(10)],
            "action.arm": [np.array([float(i)], dtype=np.float32) for i in range(10)],
            "language.task": ["pick"] * 10,
        }
    )
    modality_configs = {
        "video": ModalityConfig(delta_indices=[-5, 0], modality_keys=["cam"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["arm"]),
        "action": ModalityConfig(delta_indices=[0, 1], modality_keys=["arm"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }

    with pytest.raises(IndexError, match="out of bounds"):
        extract_step_data(df, 0, modality_configs, EmbodimentTag.NEW_EMBODIMENT, allow_padding=False)

    padded = extract_step_data(
        df, 0, modality_configs, EmbodimentTag.NEW_EMBODIMENT, allow_padding=True
    )
    assert len(padded.images["cam"]) == 2


def test_sharded_dataset_uses_subset_video_loading():
    """ShardedSingleStepDataset.get_shard must pass only required frame indices to load_episode."""
    modality_configs = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["cam"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["x"]),
        "action": ModalityConfig(delta_indices=[0, 1], modality_keys=["x"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }
    full_df = pd.DataFrame(
        {
            "video.cam": [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(40)],
            "state.x": [np.array([float(i)], dtype=np.float32) for i in range(40)],
            "action.x": [np.array([float(i)], dtype=np.float32) for i in range(40)],
            "language.task": ["task"] * 40,
        }
    )
    requested_indices: list[list[int]] = []

    def fake_load_episode(ep_idx, video_indices=None):
        requested_indices.append(list(video_indices or []))
        return full_df

    with patch("gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoader") as MockLoader:
        mock_loader = MagicMock()
        mock_loader.episode_lengths = [40]
        mock_loader.get_episode_length = lambda idx: 40
        mock_loader.load_episode.side_effect = fake_load_episode
        MockLoader.return_value = mock_loader

        ds = ShardedSingleStepDataset(
            dataset_path="/fake",
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_configs=modality_configs,
            shard_size=10,
            episode_sampling_rate=0.25,
            seed=0,
        )
        ds.set_processor(lambda msgs: {"ok": True})
        shard0 = ds.get_shard(0)

    assert len(shard0) > 0
    assert len(requested_indices) == 1
    assert len(requested_indices[0]) == len(shard0)
    assert len(requested_indices[0]) < 40


def test_merge_statistics_constant_feature_has_zero_std_not_nan():
    """Floating-point cancellation on constant features must yield std=0.0, not NaN."""
    stats = [
        {
            "arm": {
                "min": [0.3],
                "max": [0.3],
                "mean": [0.3],
                "std": [0.0],
                "q01": [0.3],
                "q99": [0.3],
            }
        },
        {
            "arm": {
                "min": [0.3],
                "max": [0.3],
                "mean": [0.3],
                "std": [0.0],
                "q01": [0.3],
                "q99": [0.3],
            }
        },
    ]
    merged = merge_statistics(stats, [0.3, 0.7])
    assert np.isfinite(merged["arm"]["std"][0])
    assert merged["arm"]["std"][0] == pytest.approx(0.0)


def test_sharded_mixture_eval_mode_terminates_and_closes_executor():
    """Eval-mode ShardedMixtureDataset must finish after one pass and clean up its executor."""
    from tests.gr00t.data.test_sharded_datasets import MockShardedDataset

    datasets = [MockShardedDataset("/fake/0", num_shards=2, shard_length=3)]
    proc = MagicMock()
    with patch("torch.distributed.is_initialized", return_value=False):
        mixture = ShardedMixtureDataset(
            datasets=datasets,
            weights=[1.0],
            processor=proc,
            seed=42,
            training=False,
        )
    items = list(mixture)
    assert len(items) == 6
    assert mixture._executor is None


def test_action_head_rtc_inpainting_freezes_prefix_steps():
    """RTC inpainting must keep rtc_frozen_steps exactly fixed to previous action tail."""
    cfg = Gr00tN1d7Config(
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=8,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        use_alternate_vl_dit=False,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 32,
            "interleave_self_attention": True,
        },
    )
    head = Gr00tN1d7ActionHead(cfg).eval()
    backbone_out = BatchFeature(
        data={
            "backbone_features": torch.randn(1, 6, 32),
            "backbone_attention_mask": torch.ones(1, 6, dtype=torch.bool),
            "image_mask": torch.ones(1, 6, dtype=torch.bool),
        }
    )
    prev_action = torch.arange(8 * 7, dtype=torch.float32).reshape(1, 8, 7)
    action_in = BatchFeature(
        data={
            "state": torch.zeros(1, 1, 7),
            "embodiment_id": torch.zeros(1, dtype=torch.long),
            "action": prev_action,
        }
    )
    options = {
        "action_horizon": 8,
        "rtc_overlap_steps": 4,
        "rtc_frozen_steps": 2,
        "rtc_ramp_rate": 5.0,
    }
    out = head.get_action(backbone_out, action_in, options=options)
    # First 2 steps must equal prev_action[:, 4:6, :] exactly (vel_strength=0).
    torch.testing.assert_close(out["action_pred"][:, :2, :], prev_action[:, 4:6, :])
