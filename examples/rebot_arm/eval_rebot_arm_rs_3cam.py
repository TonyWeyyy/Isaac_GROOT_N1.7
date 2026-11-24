#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entrypoint forwarding to ``eval_rebot_arm_rs_3cam_v2.py``.

Both ``eval_rebot_arm_rs_3cam.py`` and ``eval_rebot_arm_rs_3cam_v2.py`` now share
the single unified implementation in ``eval_rebot_arm_rs_3cam_v2.py`` +
``rebot_control_utils.py`` to eliminate duplicate control/safety logic.
"""

from __future__ import annotations

from pathlib import Path
import sys


_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from eval_rebot_arm_rs_3cam_v2 import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
