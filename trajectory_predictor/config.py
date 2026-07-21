#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared default paths and runtime helpers for the predictor project."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


# =============================================================================
# 默认绝对路径
#
# 这些只是默认值，运行 train_predictor.py 或 build_tube.py 时，
# 仍然可以通过终端参数覆盖。
# =============================================================================

# 外部输入：真实轨迹数据
DEFAULT_REAL_PATH = Path(
    # "/home/tealab_shared/trajectories/mountain_car/starv_state/real_trajectories.npz"
    #"/home/tealab_shared/trajectories/pendulum/starv_state/real_trajectories.npz"
    "/home/tealab_shared/trajectories/cartpole/starv_state/real_trajectories.npz"
)

# 外部输入：DWM verification grid 和 cell 信息
DEFAULT_GRID_RESULT_PATH = Path(
#    "/home/tealab_shared/dwm_reachable_tube/mountain_car/safety_result.json"
#    "/home/UFAD/xinyangwang/projects/verifiable_wm/results/mountain_car/safety_result.json"
#    "/home/UFAD/xinyangwang/projects/verifiable_wm/results/pendulum/safety_result.json"
    "/home/UFAD/xinyangwang/projects/verifiable_wm/results/cartpole/safety_result.json"
)

# Predictor 模型输出
DEFAULT_CHECKPOINT_PATH = Path(
    "/home/UFAD/xinyangwang/projects/verifiable_wm/trajectory_predictor/predictor_transformer.pth"
)

# Predictor Tube 输出
DEFAULT_TUBE_OUTPUT_PATH = Path(
    "/home/UFAD/xinyangwang/projects/verifiable_wm/trajectory_predictor/predictor_tube.json"
)


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """
    Resolve the requested computation device.

    auto:
        Use CUDA when available, otherwise use CPU.
    """

    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return torch.device(name)


def absolute_path(path: Path) -> Path:
    """
    Expand '~' and convert a terminal path to an absolute path.

    Default paths are already absolute, but this function is still needed
    when a path is overridden from the terminal.
    """

    return path.expanduser().resolve()


def ensure_parent(path: Path) -> None:
    """
    Create the parent directory for an output file when necessary.

    This is mainly used when --checkpoint or --tube-output is overridden.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )