#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Default paths and shared runtime helpers for the predictor project.

The paths below are fixed defaults, not search locations.  Every value can
still be overridden by the corresponding command-line argument.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


# Predictor runtime defaults.  The trajectory horizon is inferred from the
# selected dataset/checkpoint unless the user explicitly supplies --horizon.
DEFAULT_SAMPLES_PER_CELL = 3


# =============================================================================
# Default experiment and paths
#
# These values form one internally consistent default experiment.  No file or
# directory is searched automatically.  Edit these constants for another
# permanent default, or override any of them from the command line.
# =============================================================================

#DEFAULT_ENVIRONMENT = "brake_system"
#DEFAULT_ENVIRONMENT = "cartpole"
#DEFAULT_ENVIRONMENT = "mountain_car"
DEFAULT_ENVIRONMENT = "pendulum"

DEFAULT_DATA_ROOT = Path("/home/tealab_shared/safety_results")
DEFAULT_PROJECT_ROOT = Path(
    "/home/UFAD/xinyangwang/projects/verifiable_wm/trajectory_predictor"
)

DEFAULT_REAL_PATH = (
    DEFAULT_DATA_ROOT / DEFAULT_ENVIRONMENT / "real_trajectories.npz"
)
DEFAULT_GRID_RESULT_PATH = (
#    "/home/tealab_shared/safety_results/brake_system/safety_result.json"
#    "/home/tealab_shared/safety_results/cartpole/safety_result_big_cell_a8_lamda01.json"
#   "/home/tealab_shared/safety_results/mountain_car/safety_result_big_cell_best.json" 
    "/home/tealab_shared/safety_results/pendulum/safety_result_big_cell_a16_lambda05.json"   
)

DEFAULT_MODEL_DIR = (
    DEFAULT_PROJECT_ROOT / "models" / DEFAULT_ENVIRONMENT
)
DEFAULT_CHECKPOINT_PATH = (
    DEFAULT_MODEL_DIR / "predictor_transformer.pth"
)
DEFAULT_TRAJECTORY_OUTPUT_PATH = (
    DEFAULT_MODEL_DIR / "predictor_trajectories.npz"
)
DEFAULT_TUBE_OUTPUT_PATH = (
    DEFAULT_MODEL_DIR / "predictor_tube.json"
)


def validate_environment(name: str) -> str:
    """Validate and return a user-supplied environment label."""

    environment = name.strip()
    if not environment:
        raise ValueError("--env must not be empty")
    return environment


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
    """Expand ``~`` and resolve a user-supplied path."""

    return path.expanduser().resolve()


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a user-supplied output file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
