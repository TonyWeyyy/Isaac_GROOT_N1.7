#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a command-space-aligned copy of a Seeed reBot B601-RS LeRobot v2 dataset.

In raw B601-RS LeRobot recordings, ``observation.state`` stores physical motor angles
while ``action`` stores command angles (``command = physical / JOINT_DIRECTIONS`` where
``JOINT_DIRECTIONS = [1, 1, -1, -1, -1, 1, 6]``). Because ``elbow_flex``, ``wrist_flex``,
and ``wrist_yaw`` have opposite signs between state and action, GR00T's ``RELATIVE``
action representation (``action - state``) would otherwise compute ``~ -2 * state``.

This script writes a non-destructive converted copy where:
  - ``observation.state`` in every parquet file is converted to command space
    (``state / JOINT_DIRECTIONS``), matching ``action`` in sign and scale.
  - ``videos/`` is symlinked to avoid duplicating gigabytes of video files.
  - Stale ``meta/stats.json`` and ``meta/relative_stats.json`` are omitted so GR00T
    recomputes exact statistics in command space automatically.

Usage:
    python examples/rebot_arm/align_dataset_command_space.py \
        --src ~/.cache/huggingface/lerobot/seeed_rebot_b601_rs/grab_red_cube_3cam_v3_for_groot \
        --dst ~/.cache/huggingface/lerobot/seeed_rebot_b601_rs/grab_red_cube_3cam_v3_cmd_rel_for_groot
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from rebot_control_utils import JOINT_DIRECTIONS, JOINT_KEYS  # noqa: E402


def convert_state_column(
    states_np: np.ndarray,
    directions: np.ndarray = JOINT_DIRECTIONS,
) -> np.ndarray:
    """Convert an (N, 7) array of physical states into command space."""
    arr = np.asarray(states_np, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != len(directions):
        raise ValueError(f"Expected observation.state shape (N, {len(directions)}), got {arr.shape}")
    return (arr / np.asarray(directions, dtype=np.float32)).astype(np.float32)


def align_dataset(
    src_dir: Path,
    dst_dir: Path,
    *,
    overwrite: bool = False,
    symlink_videos: bool = True,
) -> dict[str, np.ndarray]:
    """Create a command-space-aligned dataset copy and return diagnostic delta stats."""
    src_dir = src_dir.expanduser().resolve()
    dst_dir = dst_dir.expanduser().resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {src_dir}")
    if src_dir == dst_dir:
        raise ValueError("--dst must be a different directory from --src")

    if dst_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_dir}. Pass --overwrite to replace it."
            )
        if dst_dir.is_symlink():
            dst_dir.unlink()
        else:
            shutil.rmtree(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy required metadata files (omit stale stats.json / relative_stats.json so GR00T
    #    recomputes exact command-space stats on first load).
    src_meta = src_dir / "meta"
    dst_meta = dst_dir / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    for meta_name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"):
        src_file = src_meta / meta_name
        if src_file.exists():
            shutil.copy2(src_file, dst_meta / meta_name)
        elif meta_name != "modality.json":
            raise FileNotFoundError(f"Missing required metadata file: {src_file}")

    # Ensure modality.json exists (fall back to examples/rebot_arm/modality.json if absent)
    if not (dst_meta / "modality.json").exists():
        template = Path(__file__).resolve().parent / "modality.json"
        if template.exists():
            shutil.copy2(template, dst_meta / "modality.json")

    # 2. Link or copy videos directory
    src_videos = src_dir / "videos"
    dst_videos = dst_dir / "videos"
    if src_videos.exists():
        if symlink_videos:
            dst_videos.symlink_to(src_videos, target_is_directory=True)
        else:
            shutil.copytree(src_videos, dst_videos)

    # 3. Convert every parquet episode file
    src_data = src_dir / "data"
    parquet_files = sorted(src_data.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {src_data}")

    raw_step0_deltas = []
    aligned_step0_deltas = []
    aligned_step15_deltas = []
    total_rows = 0

    for src_pq in parquet_files:
        rel_path = src_pq.relative_to(src_dir)
        dst_pq = dst_dir / rel_path
        dst_pq.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(src_pq)
        if "observation.state" not in table.column_names:
            raise KeyError(f"Missing 'observation.state' column in {src_pq}")

        raw_state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(
            np.float32
        )
        action = np.stack(table["action"].to_numpy(zero_copy_only=False)).astype(np.float32)
        aligned_state = convert_state_column(raw_state)

        raw_step0_deltas.append(action[:, :6] - raw_state[:, :6])
        aligned_step0_deltas.append(action[:, :6] - aligned_state[:, :6])
        if len(action) > 15:
            aligned_step15_deltas.append(action[15:, :6] - aligned_state[:-15, :6])

        # Preserve exact column type (fixed_size_list or list of float32)
        col_idx = table.schema.get_field_index("observation.state")
        orig_field = table.schema.field(col_idx)
        new_col = pa.array([row.tolist() for row in aligned_state], type=orig_field.type)
        new_table = table.set_column(col_idx, orig_field, new_col)
        pq.write_table(new_table, dst_pq)
        total_rows += len(raw_state)

    raw_d0 = np.concatenate(raw_step0_deltas, axis=0)
    aln_d0 = np.concatenate(aligned_step0_deltas, axis=0)
    aln_d15 = np.concatenate(aligned_step15_deltas, axis=0) if aligned_step15_deltas else aln_d0

    return {
        "episodes": np.array(len(parquet_files)),
        "frames": np.array(total_rows),
        "raw_step0_span": raw_d0.max(axis=0) - raw_d0.min(axis=0),
        "aligned_step0_span": aln_d0.max(axis=0) - aln_d0.min(axis=0),
        "aligned_step0_std": aln_d0.std(axis=0),
        "aligned_step15_span": aln_d15.max(axis=0) - aln_d15.min(axis=0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert B601-RS LeRobot v2 observation.state to command space for true RELATIVE actions."
    )
    parser.add_argument("--src", required=True, type=Path, help="Source LeRobot v2 dataset dir")
    parser.add_argument("--dst", required=True, type=Path, help="Output aligned dataset dir")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing --dst")
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy videos instead of symlinking (uses extra disk space)",
    )
    args = parser.parse_args(argv)

    summary = align_dataset(
        args.src,
        args.dst,
        overwrite=args.overwrite,
        symlink_videos=not args.copy_videos,
    )
    print(
        f"Aligned {int(summary['episodes'])} episodes ({int(summary['frames'])} frames) -> {args.dst}"
    )
    print("Joints:", ", ".join(JOINT_KEYS[:6]))
    print("  BEFORE (raw physical state) step-0 (action-state) span :", np.round(summary["raw_step0_span"], 2))
    print("  AFTER  (cmd-aligned state)  step-0 (action-state) span :", np.round(summary["aligned_step0_span"], 2))
    print("  AFTER  (cmd-aligned state)  step-0 (action-state) std  :", np.round(summary["aligned_step0_std"], 2))
    print("  AFTER  (cmd-aligned state)  step-15 (action-state) span:", np.round(summary["aligned_step15_span"], 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
