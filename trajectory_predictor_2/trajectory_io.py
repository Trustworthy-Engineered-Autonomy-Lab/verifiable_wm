#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NPZ compatibility helpers for predictor trajectory outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


TRAJECTORY_SPLIT_KEYS = ("train_traj", "val_traj", "test_traj")


def load_reference_trajectory_npz(
    path: Path,
    horizon: int,
    state_dim: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load a real-trajectory NPZ and extract initial states for prediction.

    The returned payload is a copy of the reference file.  Standard trajectory
    arrays are not copied because they will be replaced by predictor outputs.
    Action arrays and scalar metadata are preserved so the generated NPZ keeps
    the same public format as its reference file.
    """
    if not path.exists():
        raise FileNotFoundError(f"reference trajectory file does not exist: {path}")

    payload: Dict[str, np.ndarray] = {}
    initial_states: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as source:
        present_splits = [
            key for key in TRAJECTORY_SPLIT_KEYS if key in source.files
        ]
        if not present_splits:
            raise KeyError(
                f"{path} has none of {TRAJECTORY_SPLIT_KEYS}; "
                f"available keys: {list(source.files)}"
            )

        for key in source.files:
            if key not in TRAJECTORY_SPLIT_KEYS:
                payload[key] = np.asarray(source[key]).copy()

        for key in present_splits:
            trajectories = np.asarray(source[key])
            if trajectories.ndim != 3:
                raise ValueError(
                    f"{key} must have shape (N, T+1, state_dim), "
                    f"got {trajectories.shape}"
                )
            if trajectories.shape[0] == 0:
                raise ValueError(f"{key} is empty")
            if trajectories.shape[1] < horizon + 1:
                raise ValueError(
                    f"{key} contains {trajectories.shape[1] - 1} transition "
                    f"steps, but horizon={horizon} was requested"
                )
            if trajectories.shape[2] != state_dim:
                raise ValueError(
                    f"{key} state_dim={trajectories.shape[2]} does not match "
                    f"predictor state_dim={state_dim}"
                )
            if not np.isfinite(trajectories).all():
                raise ValueError(f"{key} contains NaN or Inf")
            initial_states[key] = np.asarray(
                trajectories[:, 0, :], dtype=np.float32
            )

    # If a shorter predictor horizon was requested, matching action arrays must
    # be shortened as well.  Their values come from the reference rollouts; the
    # state-only predictor does not synthesize controller actions.
    for split_key, states in initial_states.items():
        action_key = split_key.replace("_traj", "_actions")
        if action_key not in payload:
            continue
        actions = np.asarray(payload[action_key])
        if actions.ndim < 2:
            raise ValueError(
                f"{action_key} must have at least 2 dimensions, got {actions.shape}"
            )
        if actions.shape[0] != states.shape[0]:
            raise ValueError(
                f"{action_key} count={actions.shape[0]} does not match "
                f"{split_key} count={states.shape[0]}"
            )
        if actions.shape[1] < horizon:
            raise ValueError(
                f"{action_key} contains {actions.shape[1]} steps, "
                f"but horizon={horizon} was requested"
            )
        payload[action_key] = actions[:, :horizon, ...].copy()

    if "rollout_steps" in payload:
        payload["rollout_steps"] = np.asarray(horizon, dtype=np.int64)

    return payload, initial_states


def add_predicted_splits(
    payload: Dict[str, np.ndarray],
    predicted_splits: Dict[str, np.ndarray],
    horizon: int,
    state_dim: int,
) -> Dict[str, np.ndarray]:
    """Insert predicted train/val/test arrays after strict shape validation."""
    result = dict(payload)
    for key, trajectories in predicted_splits.items():
        if key not in TRAJECTORY_SPLIT_KEYS:
            raise KeyError(f"unsupported trajectory split: {key}")
        trajectories = np.asarray(trajectories, dtype=np.float32)
        expected_tail = (horizon + 1, state_dim)
        if trajectories.ndim != 3 or trajectories.shape[1:] != expected_tail:
            raise ValueError(
                f"{key} must have shape (N, {horizon + 1}, {state_dim}), "
                f"got {trajectories.shape}"
            )
        if not np.isfinite(trajectories).all():
            raise ValueError(f"predicted {key} contains NaN or Inf")
        result[key] = trajectories
    return result
