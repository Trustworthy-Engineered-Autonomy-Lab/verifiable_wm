#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid handling, three-point trajectory prediction, and tube saving."""

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


def sample_cell(bounds: np.ndarray, samples_per_cell: int = 3) -> np.ndarray:
    """Return exactly ``samples_per_cell`` deterministic points in a cell.

    Points are evenly spaced on the diagonal from the lower corner to the
    upper corner.  With the required default of three this gives the lower
    corner, cell center, and upper corner.  Including both corners also makes
    the min/max envelope at step zero equal to the full initial cell.
    """
    bounds = np.asarray(bounds, dtype=np.float32)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must have shape (state_dim, 2), got {bounds.shape}")
    if samples_per_cell < 2:
        raise ValueError("samples_per_cell must be at least 2")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("cell lower bounds must not exceed upper bounds")

    weights = np.linspace(0.0, 1.0, samples_per_cell, dtype=np.float32)[:, None]
    lower = bounds[:, 0][None, :]
    upper = bounds[:, 1][None, :]
    return lower + weights * (upper - lower)


def build_raw_tubes(
    model: TrajectoryTransformer,
    mean: np.ndarray,
    std: np.ndarray,
    cell_bounds: np.ndarray,
    samples_per_cell: int,
    horizon: int,
    batch_size: int,
    device: torch.device,
    trajectory_output_path: Path,
) -> np.ndarray:
    """Predict every cell, save one aggregate NPZ, and form min/max tubes."""
    if horizon < 1 or horizon > model.horizon:
        raise ValueError(
            f"horizon must be in [1, {model.horizon}], got {horizon}"
        )
    trajectory_output_path.parent.mkdir(parents=True, exist_ok=True)

    num_cells = len(cell_bounds)
    raw_tubes = np.empty(
        (
            num_cells,
            horizon + 1,
            model.state_dim,
            2,
        ),
        dtype=np.float32,
    )
    all_initial_states = np.empty(
        (num_cells, samples_per_cell, model.state_dim),
        dtype=np.float32,
    )
    all_trajectories = np.empty(
        (num_cells, samples_per_cell, horizon + 1, model.state_dim),
        dtype=np.float32,
    )

    print("========== Raw predictor tube ==========")
    print(f"cells             : {num_cells}")
    print(f"samples per cell  : {samples_per_cell}")
    print(f"transition steps  : {horizon}")
    print(f"total predictions : {num_cells * samples_per_cell}")
    print(f"trajectory output : {trajectory_output_path}")

    progress_interval = max(1, num_cells // 10)
    for cell_index, bounds in enumerate(cell_bounds):
        initial_states = sample_cell(bounds, samples_per_cell)
        predictions = predict_trajectories(
            model, initial_states, mean, std, batch_size, device
        )[:, : horizon + 1, :]
        lower = predictions.min(axis=0)
        upper = predictions.max(axis=0)

        all_initial_states[cell_index] = initial_states
        all_trajectories[cell_index] = predictions
        raw_tubes[cell_index, :, :, 0] = lower
        raw_tubes[cell_index, :, :, 1] = upper

        completed = cell_index + 1
        if completed % progress_interval == 0 or completed == num_cells:
            print(f"built cells       : {completed}/{num_cells}")

    np.savez_compressed(
        trajectory_output_path,
        cell_indices=np.arange(num_cells, dtype=np.int64),
        initial_bounds=np.asarray(cell_bounds, dtype=np.float32),
        initial_states=all_initial_states,
        trajectories=all_trajectories,
        lower=raw_tubes[..., 0],
        upper=raw_tubes[..., 1],
        horizon=np.asarray(horizon, dtype=np.int64),
        samples_per_cell=np.asarray(samples_per_cell, dtype=np.int64),
    )
    return raw_tubes


def save_tube_json(
    path: Path,
    source_grid: Dict[str, Any],
    grid: Grid,
    cell_bounds: np.ndarray,
    raw_tubes: np.ndarray,
    trajectory_path: Path,
    checkpoint: Dict[str, Any],
    grid_path: Path,
    checkpoint_path: Path,
    samples_per_cell: int,
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

    if raw_tubes.shape[0] != grid.total_cells:
        raise ValueError(
            f"expected {grid.total_cells} cell tubes, got {raw_tubes.shape[0]}"
        )

    try:
        trajectory_file = str(trajectory_path.relative_to(path.parent))
    except ValueError:
        trajectory_file = str(trajectory_path)

    cells = []
    for i in range(grid.total_cells):
        cells.append(
            {
                "bounds": raw_tubes[i].astype(float).tolist(),
                "raw_bounds": raw_tubes[i].astype(float).tolist(),
                "initial_bounds": cell_bounds[i].astype(float).tolist(),
                "trajectory_file": trajectory_file,
                "trajectory_index": i,
            }
        )

    best_selection_loss = checkpoint.get("best_selection_loss")
    payload = {
        "method": "transformer_three_sample_minmax_envelope",
        "environment": environment,
        "guarantee_type": "sampled envelope; no formal coverage guarantee",
        "sampling_strategy": "lower_corner_center_upper_corner",
        "samples_per_cell": int(samples_per_cell),
        "horizon": int(raw_tubes.shape[1] - 1),
        "state_dim": int(raw_tubes.shape[2]),
        "source_grid_result": str(grid_path),
        "checkpoint": str(checkpoint_path),
        "trajectory_file": trajectory_file,
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "best_selection_loss": (
            None
            if best_selection_loss is None
            else float(best_selection_loss)
        ),
        "training_protocol": checkpoint.get("training_protocol", {}),
        "grid": grid_json,
        "cells": cells,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
