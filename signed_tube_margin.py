#!/usr/bin/env python3
"""Evaluate signed trajectory margins against StarV reachable tubes.

For every checked state, the margin is the smallest signed perpendicular
distance to the tube boundaries in the selected two dimensions.  A positive
margin is inside the tube, zero is on a boundary, and a negative margin is
outside.  Each trajectory is assigned its smallest state margin.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_DIMS = {
    "cartpole": (0, 2),
    "mountain_car": (0, 1),
    "pendulum": (0, 1),
}
EPS = 1e-10

# Edit these paths directly when running this script without terminal options.
# DEFAULT_ENV = "cartpole"
# DEFAULT_SAFETY_PATH = PROJECT_ROOT / "results/cartpole/safety_result_cell_100_a8_lamda01.json"
# DEFAULT_REAL_TRAJ_PATH = PROJECT_ROOT / "datasets/cartpole/data_cell_100/real_trajectories.npz"
# DEFAULT_DWM_TRAJ_PATH = PROJECT_ROOT / "datasets/cartpole/data_cell_100/dwm_trajectories_saliency.npz"
# DEFAULT_OUT_DIR = PROJECT_ROOT / "results/cartpole/signed_tube_margin"

DEFAULT_ENV = "mountain_car"
DEFAULT_SAFETY_PATH = PROJECT_ROOT / "results/mountain_car/safety_result_cell_100_a16_lambda05.json"
DEFAULT_REAL_TRAJ_PATH = PROJECT_ROOT / "datasets/mountain_car/data_cell_100/real_trajectories.npz"
DEFAULT_DWM_TRAJ_PATH = PROJECT_ROOT / "datasets/mountain_car/data_cell_100/dwm_trajectories_saliency.npz"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results/mountain_car/signed_tube_margin"

# DEFAULT_ENV = "pendulum"
# DEFAULT_SAFETY_PATH = PROJECT_ROOT / "results/pendulum/safety_result_cell_100_a16_lambda05.json"
# DEFAULT_REAL_TRAJ_PATH = PROJECT_ROOT / "datasets/pendulum/data_cell_100/real_trajectories.npz"
# DEFAULT_DWM_TRAJ_PATH = PROJECT_ROOT / "datasets/pendulum/data_cell_100/dwm_trajectories_saliency.npz"
# DEFAULT_OUT_DIR = PROJECT_ROOT / "results/pendulum/signed_tube_margin"


@dataclass
class GridInfo:
    names: list[str]
    starts: np.ndarray
    stops: np.ndarray
    nums: np.ndarray
    steps: np.ndarray

    @property
    def ndim(self) -> int:
        return len(self.names)

    def point_to_linear_index(self, point: np.ndarray) -> int | None:
        point = np.asarray(point, dtype=float)
        if point.size < self.ndim:
            return None

        index = 0
        for dim in range(self.ndim):
            value = float(point[dim])
            start, stop = float(self.starts[dim]), float(self.stops[dim])
            step, count = float(self.steps[dim]), int(self.nums[dim])
            if value < start - EPS or value > stop + EPS:
                return None
            local_index = count - 1 if abs(value - stop) <= EPS else int(math.floor((value - start) / step))
            index = index * count + max(0, min(count - 1, local_index))
        return index


def interval_pairs(bounds: Sequence[float]) -> list[tuple[float, float]]:
    values = np.asarray(bounds, dtype=float).reshape(-1)
    if values.size < 2 or values.size % 2:
        raise ValueError(f"bounds must contain low/high pairs, got {values.tolist()}")

    pairs = []
    for start in range(0, values.size, 2):
        low, high = float(values[start]), float(values[start + 1])
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("bounds must be finite")
        pairs.append((min(low, high), max(low, high)))
    return pairs


def signed_interval_margin(value: float, bounds: Sequence[float], delta: float = 0.0) -> float:
    """Return signed distance to a one-dimensional interval union boundary."""
    if delta < 0.0:
        raise ValueError("tube inflation delta must be nonnegative")
    return float(max(min(value - (low - delta), (high + delta) - value) for low, high in interval_pairs(bounds)))


def state_signed_margin(
    state: np.ndarray, bounds: Sequence[Sequence[float]], dims: Sequence[int], delta: float = 0.0
) -> float:
    """Return the smallest signed boundary margin among the selected dimensions."""
    if len(dims) != 2:
        raise ValueError(f"exactly two check dimensions are required, got {tuple(dims)}")
    if any(dim < 0 or dim >= len(state) or dim >= len(bounds) for dim in dims):
        raise ValueError("selected dimension is missing from state or tube bounds")
    return float(min(signed_interval_margin(float(state[dim]), bounds[dim], delta) for dim in dims))


def trajectory_signed_margin(
    trajectory: np.ndarray,
    bounds_history: Sequence[Sequence[Sequence[float]]],
    dims: Sequence[int],
    delta: float = 0.0,
) -> float:
    if len(trajectory) != len(bounds_history):
        raise ValueError(
            f"trajectory has {len(trajectory)} states but tube has {len(bounds_history)} time steps"
        )
    if not len(trajectory):
        raise ValueError("trajectory is empty")
    return float(min(
        state_signed_margin(state, bounds, dims, delta) for state, bounds in zip(trajectory, bounds_history)
    ))


def descending_p95(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=float)
    if values.size != 400:
        raise ValueError(f"expected exactly 400 valid trajectory scores, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectory scores must be finite")
    return float(np.sort(values)[::-1][379])


def load_safety_result(path: Path) -> tuple[GridInfo, list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if "grid" not in data or "cells" not in data:
        raise ValueError(f"{path} is missing grid or cells")
    dimensions = data["grid"].get("dims", [])
    if not dimensions:
        raise ValueError(f"{path} has no grid dimensions")
    grid = GridInfo(
        names=[dimension.get("name", f"state_{index}") for index, dimension in enumerate(dimensions)],
        starts=np.array([float(dimension["start"]) for dimension in dimensions]),
        stops=np.array([float(dimension["stop"]) for dimension in dimensions]),
        nums=np.array([int(dimension["num"]) for dimension in dimensions]),
        steps=np.array([
            float(dimension.get("step", (float(dimension["stop"]) - float(dimension["start"])) / int(dimension["num"])))
            for dimension in dimensions
        ]),
    )
    return grid, data["cells"]


def _state_in_initial_bounds(state: np.ndarray, bounds: Sequence[Sequence[float]], ndim: int) -> bool:
    if len(state) < ndim or len(bounds) < ndim:
        return False
    return all(signed_interval_margin(float(state[dim]), bounds[dim]) >= -EPS for dim in range(ndim))


def find_cell(initial_state: np.ndarray, grid: GridInfo, cells: Sequence[dict[str, Any]]) -> tuple[int | None, dict[str, Any] | None]:
    candidate = grid.point_to_linear_index(initial_state)
    if candidate is not None and candidate < len(cells):
        cell = cells[candidate]
        if cell.get("bounds") and _state_in_initial_bounds(initial_state, cell["bounds"][0], grid.ndim):
            return candidate, cell
    for index, cell in enumerate(cells):
        if cell.get("bounds") and _state_in_initial_bounds(initial_state, cell["bounds"][0], grid.ndim):
            return index, cell
    return None, None


def evaluate_set(
    trajectories: np.ndarray,
    grid: GridInfo,
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int],
    delta: float = 0.0,
) -> list[dict[str, Any]]:
    if trajectories.ndim != 3:
        raise ValueError(f"trajectories must have shape (N, T+1, state_dim), got {trajectories.shape}")
    rows = []
    for trajectory_index, trajectory in enumerate(trajectories):
        cell_index, cell = find_cell(trajectory[0], grid, cells)
        row: dict[str, Any] = {
            "traj_index": trajectory_index,
            "cell_index": cell_index,
            "status": "invalid",
            "signed_margin": None,
            "error": "",
        }
        if cell is None:
            row["error"] = "initial state does not match a StarV cell"
        elif "error_msg" in cell:
            row["error"] = f"StarV cell error: {cell['error_msg']}"
        else:
            try:
                row["signed_margin"] = trajectory_signed_margin(trajectory, cell["bounds"], dims, delta)
                row["status"] = "valid"
            except (KeyError, TypeError, ValueError) as error:
                row["error"] = str(error)
        rows.append(row)
    return rows


def interval_union_length(bounds: Sequence[float], delta: float = 0.0) -> float:
    """Return the length of an interval union after symmetric inflation."""
    if delta < 0.0:
        raise ValueError("tube inflation delta must be nonnegative")
    merged: list[list[float]] = []
    for low, high in sorted((low - delta, high + delta) for low, high in interval_pairs(bounds)):
        if merged and low <= merged[-1][1] + EPS:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return float(sum(high - low for low, high in merged))


def normalized_cell_tube_areas(
    cells: Sequence[dict[str, Any]], grid: GridInfo, dims: Sequence[int], delta: float = 0.0
) -> list[float]:
    """Return one time-averaged normalized 2-D tube area for every valid cell."""
    if len(dims) != 2 or any(dim < 0 or dim >= grid.ndim for dim in dims):
        raise ValueError("exactly two valid check dimensions are required")
    grid_area = float(np.prod(grid.stops[list(dims)] - grid.starts[list(dims)]))
    if not np.isfinite(grid_area) or grid_area <= 0.0:
        raise ValueError("verification grid area must be positive and finite")

    values = []
    for cell in cells:
        if "error_msg" in cell:
            continue
        history = cell.get("bounds", [])
        if len(history) < 2:
            raise ValueError("tube history must include an initial state and at least one future state")
        time_areas = []
        for bounds in history[1:]:
            if any(dim >= len(bounds) for dim in dims):
                raise ValueError("tube bounds do not contain selected dimensions")
            time_areas.append(float(np.prod([interval_union_length(bounds[dim], delta) for dim in dims])))
        values.append(float(np.mean(time_areas) / grid_area))
    return values


def table_metrics(
    trajectories: np.ndarray,
    grid: GridInfo,
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int],
    delta: float = 0.0,
) -> dict[str, Any]:
    """Summarize Table II robustness and normalized tube-area metrics."""
    rows = evaluate_set(trajectories, grid, cells, dims, delta)
    scores = [float(row["signed_margin"]) for row in rows if row["status"] == "valid"]
    areas = normalized_cell_tube_areas(cells, grid, dims, delta)
    return {
        "delta": float(delta),
        "robustness": {
            "mean": float(np.mean(scores)) if scores else None,
            "minimum": float(min(scores)) if scores else None,
            "maximum": float(max(scores)) if scores else None,
        },
        "robustness_valid_trajectories": len(scores),
        "robustness_total_trajectories": len(rows),
        "tube_area": {
            "mean": float(np.mean(areas)) if areas else None,
            "std": float(np.std(areas)) if areas else None,
        },
        "tube_area_valid_cells": len(areas),
        "tube_area_total_cells": len(cells),
    }


def write_table_metrics(metrics_by_method: dict[str, dict[str, Any]], output_dir: Path) -> Path:
    """Write the Table II metric fields in one row per tube construction method."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "tube_table_metrics.csv"
    fields = [
        "method", "delta", "robustness_mean", "robustness_minimum", "robustness_maximum",
        "robustness_valid_trajectories", "robustness_total_trajectories",
        "tube_area_mean", "tube_area_std", "tube_area_valid_cells", "tube_area_total_cells",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for method, metrics in metrics_by_method.items():
            robustness = metrics["robustness"]
            tube_area = metrics["tube_area"]
            writer.writerow({
                "method": method,
                "delta": metrics["delta"],
                "robustness_mean": robustness["mean"],
                "robustness_minimum": robustness["minimum"],
                "robustness_maximum": robustness["maximum"],
                "robustness_valid_trajectories": metrics["robustness_valid_trajectories"],
                "robustness_total_trajectories": metrics["robustness_total_trajectories"],
                "tube_area_mean": tube_area["mean"],
                "tube_area_std": tube_area["std"],
                "tube_area_valid_cells": metrics["tube_area_valid_cells"],
                "tube_area_total_cells": metrics["tube_area_total_cells"],
            })
    return path


def write_results(rows: Sequence[dict[str, Any]], label: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{label}_signed_tube_margins.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["traj_index", "cell_index", "status", "signed_margin", "error"])
        writer.writeheader()
        writer.writerows(rows)

    valid_scores = [float(row["signed_margin"]) for row in rows if row["status"] == "valid"]
    p95_desc = descending_p95(valid_scores) if len(valid_scores) == 400 else None
    return {
        "total_trajectories": len(rows),
        "valid_trajectories": len(valid_scores),
        "invalid_trajectories": len(rows) - len(valid_scores),
        "minimum": float(min(valid_scores)) if valid_scores else None,
        "maximum": float(max(valid_scores)) if valid_scores else None,
        "p95_desc": p95_desc,
        "csv": str(path),
    }


def load_trajectory(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(f"key {key!r} is absent from {path}; available keys: {list(data.files)}")
        return np.asarray(data[key], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENV_DIMS, default=DEFAULT_ENV)
    parser.add_argument("--safety", type=Path, default=DEFAULT_SAFETY_PATH, help="StarV safety_result JSON")
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL_TRAJ_PATH, help="real trajectory NPZ")
    parser.add_argument("--dwm", type=Path, default=DEFAULT_DWM_TRAJ_PATH, help="DWM trajectory NPZ")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT_DIR, help="directory for CSV and JSON results")
    parser.add_argument("--real-key", default="test_traj")
    parser.add_argument("--dwm-key", default="test_traj")
    parser.add_argument("--check-dims", type=int, nargs=2, default=None)
    parser.add_argument(
        "--delta", type=float, default=None,
        help="B1 symmetric tube inflation Gamma_(1-alpha); omit to report A1 only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dims = tuple(args.check_dims) if args.check_dims is not None else ENV_DIMS[args.env]
    grid, cells = load_safety_result(args.safety)
    real_trajectories = load_trajectory(args.real, args.real_key)
    summaries = {}
    for label, trajectories in (("real", real_trajectories), ("dwm", load_trajectory(args.dwm, args.dwm_key))):
        rows = evaluate_set(trajectories, grid, cells, dims)
        summary = write_results(rows, label, args.outdir)
        summaries[label] = summary
        print(
            f"{label}: valid={summary['valid_trajectories']}/{summary['total_trajectories']}, "
            f"min={summary['minimum']}, max={summary['maximum']}, p95_desc={summary['p95_desc']}"
        )
    metrics_by_method = {"A1 (ours)": table_metrics(real_trajectories, grid, cells, dims)}
    if args.delta is not None:
        metrics_by_method["B1 (inflated)"] = table_metrics(
            real_trajectories, grid, cells, dims, args.delta
        )
    table_path = write_table_metrics(metrics_by_method, args.outdir)
    for method, metrics in metrics_by_method.items():
        robustness = metrics["robustness"]
        tube_area = metrics["tube_area"]
        print(
            f"{method}: gamma mean/min/max={robustness['mean']}/{robustness['minimum']}/{robustness['maximum']}; "
            f"normalized tube area mean+std={tube_area['mean']}+{tube_area['std']}"
        )
    summary_path = args.outdir / "signed_tube_margin_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"env": args.env, "check_dims": dims, "datasets": summaries, "table_metrics": metrics_by_method},
            file, indent=2,
        )
    print(f"table metrics: {table_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
