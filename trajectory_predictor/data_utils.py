#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trajectory loading, splitting, normalization, and PyTorch dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_SPLITS = ("train_traj", "val_traj", "test_traj")


def load_real_trajectories(
    path: Path,
    missing_train_policy: str = "error",
    derived_train_ratio: float = 0.8,
    seed: int = 2025,
) -> Dict[str, np.ndarray]:
    """Load train/validation/test trajectories.

    ``split-val`` is an explicit compatibility mode for legacy Brake System
    data containing only ``val_traj`` and ``test_traj``. It deterministically
    derives train/validation splits from the original validation data.
    ``test_traj`` is never used for training.
    """
    if not path.exists():
        raise FileNotFoundError(f"real trajectory file does not exist: {path}")
    if missing_train_policy not in {"error", "split-val"}:
        raise ValueError(
            "missing_train_policy must be either 'error' or 'split-val'"
        )
    if not 0.0 < derived_train_ratio < 1.0:
        raise ValueError("derived_train_ratio must be between 0 and 1")

    with np.load(path, allow_pickle=False) as data:
        available = set(data.files)
        missing_non_train = [
            key for key in ("val_traj", "test_traj") if key not in available
        ]
        if missing_non_train:
            raise KeyError(
                f"missing keys {missing_non_train} in {path}; "
                f"available keys: {list(data.files)}"
            )

        if "train_traj" in available:
            splits = {
                key: np.asarray(data[key], dtype=np.float32)
                for key in REQUIRED_SPLITS
            }
        elif missing_train_policy == "error":
            raise KeyError(
                f"missing key 'train_traj' in {path}; available keys: "
                f"{list(data.files)}. For a legacy Brake System file, use "
                "--missing-train-policy split-val."
            )
        else:
            original_val = np.asarray(data["val_traj"], dtype=np.float32)
            test_traj = np.asarray(data["test_traj"], dtype=np.float32)
            if len(original_val) < 2:
                raise ValueError(
                    "val_traj needs at least two trajectories for split-val mode"
                )
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(original_val))
            train_count = int(len(original_val) * derived_train_ratio)
            train_count = max(1, min(len(original_val) - 1, train_count))
            splits = {
                "train_traj": original_val[indices[:train_count]].copy(),
                "val_traj": original_val[indices[train_count:]].copy(),
                "test_traj": test_traj,
            }

    reference_shape = splits["train_traj"].shape[1:]
    for key, trajectories in splits.items():
        if trajectories.ndim != 3:
            raise ValueError(
                f"{key} must have shape (N, T+1, state_dim), got {trajectories.shape}"
            )
        if trajectories.shape[1:] != reference_shape:
            raise ValueError(
                "all splits must share (T+1, state_dim); "
                f"train={reference_shape}, {key}={trajectories.shape[1:]}"
            )
        if len(trajectories) == 0:
            raise ValueError(f"{key} is empty")
        if not np.isfinite(trajectories).all():
            raise ValueError(f"{key} contains NaN or Inf")

    return splits


def truncate_trajectory_splits(
    splits: Dict[str, np.ndarray],
    horizon: int,
) -> Dict[str, np.ndarray]:
    """Keep states s_0...s_horizon in every trajectory split."""
    if horizon < 1:
        raise ValueError("horizon must be positive")

    required_states = horizon + 1
    truncated = {}
    for key in REQUIRED_SPLITS:
        trajectories = splits[key]
        if trajectories.shape[1] < required_states:
            raise ValueError(
                f"{key} only contains {trajectories.shape[1] - 1} transition "
                f"steps, but horizon={horizon} was requested"
            )
        truncated[key] = trajectories[:, :required_states, :].copy()
    return truncated


def split_fit_selection(
    train_traj: np.ndarray,
    fit_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split train_traj into parameter-fit and checkpoint-selection sets."""
    if not 0.0 < fit_ratio < 1.0:
        raise ValueError("fit_ratio must be between 0 and 1")
    if len(train_traj) < 2:
        raise ValueError("train_traj needs at least two trajectories")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_traj))
    fit_count = int(len(train_traj) * fit_ratio)
    fit_count = max(1, min(len(train_traj) - 1, fit_count))

    fit_traj = train_traj[indices[:fit_count]].copy()
    selection_traj = train_traj[indices[fit_count:]].copy()
    return fit_traj, selection_traj


def compute_normalization(trajectories: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean/std from fit trajectories only."""
    flat = trajectories.reshape(-1, trajectories.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    return mean, std


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        trajectories: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        normalized = (
            trajectories - mean[None, None, :]
        ) / std[None, None, :]
        self.trajectories = torch.from_numpy(normalized.astype(np.float32))

    def __len__(self) -> int:
        return int(self.trajectories.shape[0])

    def __getitem__(self, index: int):
        trajectory = self.trajectories[index]
        return trajectory[0], trajectory
