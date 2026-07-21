#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid handling, sampled tube construction, conformal calibration, and saving."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from predictor_model import TrajectoryTransformer, predict_trajectories


EPS = 1e-10


@dataclass
class Grid:
    names: List[str]
    starts: np.ndarray
    stops: np.ndarray
    nums: np.ndarray
    steps: np.ndarray

    @property
    def ndim(self) -> int:
        return len(self.names)

    @property
    def total_cells(self) -> int:
        return int(np.prod(self.nums))

    def point_to_cell_index(self, point: np.ndarray) -> int:
        point = np.asarray(point, dtype=float).reshape(-1)
        if point.size < self.ndim:
            raise ValueError(f"point dim={point.size}, grid dim={self.ndim}")

        indices = []
        for dim in range(self.ndim):
            value = float(point[dim])
            start = float(self.starts[dim])
            stop = float(self.stops[dim])
            step = float(self.steps[dim])
            num = int(self.nums[dim])

            if value < start - EPS or value > stop + EPS:
                raise ValueError("initial state outside grid")
            if abs(value - stop) <= EPS:
                index = num - 1
            else:
                index = int(math.floor((value - start) / step))
                index = max(0, min(num - 1, index))
            indices.append(index)

        linear_index = 0
        for index, num in zip(indices, self.nums):
            linear_index = linear_index * int(num) + int(index)
        return int(linear_index)


def load_grid(path: Path) -> Tuple[Dict[str, Any], Grid, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"grid result does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        source = json.load(file)

    if "grid" not in source or "dims" not in source["grid"]:
        raise ValueError(f"{path} is missing grid.dims")

    dims = source["grid"]["dims"]
    names = [str(dim.get("name", f"dim{i}")) for i, dim in enumerate(dims)]
    starts = np.asarray([float(dim["start"]) for dim in dims], dtype=float)
    stops = np.asarray([float(dim["stop"]) for dim in dims], dtype=float)
    nums = np.asarray([int(dim["num"]) for dim in dims], dtype=int)
    steps = np.asarray(
        [
            float(
                dim.get(
                    "step",
                    (float(dim["stop"]) - float(dim["start"])) / int(dim["num"]),
                )
            )
            for dim in dims
        ],
        dtype=float,
    )
    grid = Grid(names, starts, stops, nums, steps)

    cell_bounds = []
    source_cells = source.get("cells", [])
    if len(source_cells) == grid.total_cells:
        for cell in source_cells:
            bounds_history = cell.get("bounds", [])
            if not bounds_history:
                cell_bounds = []
                break
            initial_bounds = np.asarray(bounds_history[0], dtype=float)
            if initial_bounds.shape != (grid.ndim, 2):
                cell_bounds = []
                break
            cell_bounds.append(initial_bounds)

    if not cell_bounds:
        for multi_index in np.ndindex(*(int(num) for num in grid.nums)):
            bounds = []
            for dim, index in enumerate(multi_index):
                lower = grid.starts[dim] + index * grid.steps[dim]
                upper = grid.starts[dim] + (index + 1) * grid.steps[dim]
                if index == int(grid.nums[dim]) - 1:
                    upper = grid.stops[dim]
                bounds.append([float(lower), float(upper)])
            cell_bounds.append(np.asarray(bounds, dtype=float))

    return source, grid, np.stack(cell_bounds, axis=0)


def sample_cell(bounds: np.ndarray, samples_per_dim: int) -> np.ndarray:
    axes = [
        np.linspace(bounds[d, 0], bounds[d, 1], samples_per_dim)
        for d in range(bounds.shape[0])
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([axis.reshape(-1) for axis in mesh], axis=1).astype(np.float32)


def build_raw_tubes(
    model: TrajectoryTransformer,
    mean: np.ndarray,
    std: np.ndarray,
    cell_bounds: np.ndarray,
    samples_per_dim: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    raw_tubes = np.empty(
        (
            len(cell_bounds),
            model.horizon + 1,
            model.state_dim,
            2,
        ),
        dtype=np.float32,
    )
    samples_per_cell = samples_per_dim ** cell_bounds.shape[1]

    print("========== Raw predictor tube ==========")
    print(f"cells             : {len(cell_bounds)}")
    print(f"samples per dim   : {samples_per_dim}")
    print(f"samples per cell  : {samples_per_cell}")
    print(f"total predictions : {len(cell_bounds) * samples_per_cell}")

    progress_interval = max(1, len(cell_bounds) // 10)
    for cell_index, bounds in enumerate(cell_bounds):
        initial_states = sample_cell(bounds, samples_per_dim)
        predictions = predict_trajectories(
            model, initial_states, mean, std, batch_size, device
        )
        lower = predictions.min(axis=0)
        upper = predictions.max(axis=0)

        # Exact initial cell, rather than sampled min/max.
        lower[0] = bounds[:, 0]
        upper[0] = bounds[:, 1]
        raw_tubes[cell_index, :, :, 0] = lower
        raw_tubes[cell_index, :, :, 1] = upper

        completed = cell_index + 1
        if completed % progress_interval == 0 or completed == len(cell_bounds):
            print(f"built cells       : {completed}/{len(cell_bounds)}")

    return raw_tubes


def trajectory_tube_violation(
    trajectory: np.ndarray,
    tube: np.ndarray,
) -> Tuple[float, np.ndarray]:
    trajectory = np.asarray(trajectory, dtype=float)
    lower = np.asarray(tube[:, :, 0], dtype=float)
    upper = np.asarray(tube[:, :, 1], dtype=float)
    if trajectory.shape != lower.shape:
        raise ValueError(
            f"trajectory/tube shape mismatch: {trajectory.shape} vs {lower.shape}"
        )

    violation = np.maximum(np.maximum(lower - trajectory, trajectory - upper), 0.0)
    state_inside = np.all(violation <= EPS, axis=1)
    return float(np.max(violation)), state_inside


def calibration_scores(
    trajectories: np.ndarray,
    raw_tubes: np.ndarray,
    grid: Grid,
) -> Tuple[np.ndarray, int]:
    scores = []
    outside = 0
    for trajectory in trajectories:
        try:
            cell_index = grid.point_to_cell_index(trajectory[0])
        except ValueError:
            outside += 1
            continue
        score, _ = trajectory_tube_violation(trajectory, raw_tubes[cell_index])
        scores.append(score)

    if not scores:
        raise ValueError("no calibration trajectories start inside the grid")
    return np.asarray(scores, dtype=float), outside


def conformal_delta(scores: np.ndarray, alpha: float) -> Tuple[float, int]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    scores = np.asarray(scores, dtype=float).reshape(-1)
    rank = int(math.ceil((len(scores) + 1) * (1.0 - alpha)))
    augmented = np.concatenate([scores, np.asarray([np.inf])])
    augmented.sort()
    delta = float(augmented[rank - 1])

    if not math.isfinite(delta):
        raise ValueError(
            f"conformal delta is infinite; calibration set is too small for alpha={alpha}"
        )
    return delta, rank


def inflate_tubes(raw_tubes: np.ndarray, delta: float) -> np.ndarray:
    certified = raw_tubes.copy()
    certified[:, 1:, :, 0] -= np.float32(delta)
    certified[:, 1:, :, 1] += np.float32(delta)
    return certified


def evaluate_tubes(
    trajectories: np.ndarray,
    tubes: np.ndarray,
    grid: Grid,
) -> Dict[str, Any]:
    in_grid = 0
    outside = 0
    fully_contained = 0
    step_rates = []
    max_violations = []

    for trajectory in trajectories:
        try:
            cell_index = grid.point_to_cell_index(trajectory[0])
        except ValueError:
            outside += 1
            continue

        in_grid += 1
        maximum_violation, state_inside = trajectory_tube_violation(
            trajectory, tubes[cell_index]
        )
        fully_contained += int(np.all(state_inside))
        step_rates.append(float(np.mean(state_inside)))
        max_violations.append(maximum_violation)

    return {
        "total_trajectories": int(len(trajectories)),
        "in_grid_trajectories": int(in_grid),
        "outside_grid_trajectories": int(outside),
        "fully_contained": int(fully_contained),
        "not_fully_contained": int(in_grid - fully_contained),
        "containment_rate": float(fully_contained / in_grid) if in_grid else 0.0,
        "mean_step_containment": float(np.mean(step_rates)) if step_rates else 0.0,
        "mean_max_violation": (
            float(np.mean(max_violations)) if max_violations else 0.0
        ),
        "worst_max_violation": (
            float(np.max(max_violations)) if max_violations else 0.0
        ),
    }


def save_tube_json(
    path: Path,
    source_grid: Dict[str, Any],
    grid: Grid,
    cell_bounds: np.ndarray,
    raw_tubes: np.ndarray,
    certified_tubes: np.ndarray,
    scores: np.ndarray,
    calibration_outside: int,
    delta: float,
    rank: int,
    raw_evaluation: Dict[str, Any],
    certified_evaluation: Dict[str, Any],
    checkpoint: Dict[str, Any],
    real_path: Path,
    grid_path: Path,
    checkpoint_path: Path,
    samples_per_dim: int,
    alpha: float,
    environment: str,
) -> None:
    grid_json = dict(source_grid["grid"])
    grid_json["dims"] = [
        {
            **dict(source_grid["grid"]["dims"][i]),
            "name": grid.names[i],
            "start": float(grid.starts[i]),
            "stop": float(grid.stops[i]),
            "num": int(grid.nums[i]),
            "step": float(grid.steps[i]),
        }
        for i in range(grid.ndim)
    ]

    cells = [
        {
            "bounds": certified_tubes[i].astype(float).tolist(),
            "raw_bounds": raw_tubes[i].astype(float).tolist(),
            "initial_bounds": cell_bounds[i].astype(float).tolist(),
        }
        for i in range(grid.total_cells)
    ]

    payload = {
        "method": "transformer_sampled_envelope_conformal",
        "environment": environment,
        "guarantee_type": "marginal probabilistic trajectory containment",
        "alpha": float(alpha),
        "coverage": float(1.0 - alpha),
        "conformal_delta": float(delta),
        "conformal_rank": int(rank),
        "samples_per_dim": int(samples_per_dim),
        "samples_per_cell": int(samples_per_dim ** grid.ndim),
        "horizon": int(raw_tubes.shape[1] - 1),
        "state_dim": int(raw_tubes.shape[2]),
        "source_real_trajectories": str(real_path),
        "source_grid_result": str(grid_path),
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "best_selection_loss": float(
            checkpoint.get("best_selection_loss", math.nan)
        ),
        "training_protocol": checkpoint.get("training_protocol", {}),
        "grid": grid_json,
        "calibration": {
            "source_split": "val_traj",
            "score_count": int(len(scores)),
            "outside_grid_count": int(calibration_outside),
            "score_min": float(scores.min()),
            "score_mean": float(scores.mean()),
            "score_max": float(scores.max()),
        },
        "evaluation": {
            "source_split": "test_traj",
            "raw_tube": raw_evaluation,
            "certified_tube": certified_evaluation,
        },
        "cells": cells,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
