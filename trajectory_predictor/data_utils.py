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


def load_real_trajectories(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"real trajectory file does not exist: {path}")

    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_SPLITS if key not in data.files]
        if missing:
            raise KeyError(
                f"missing keys {missing} in {path}; available keys: {list(data.files)}"
            )
        splits = {
            key: np.asarray(data[key], dtype=np.float32)
            for key in REQUIRED_SPLITS
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
