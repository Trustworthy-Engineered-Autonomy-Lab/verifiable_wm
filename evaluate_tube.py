#!/usr/bin/env python3
"""Evaluate real trajectories against one StarV-format reachable tube."""

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
EPS = 1e-10

EVALUATION_CONFIGS = {
    "cartpole": {
        "tube": PROJECT_ROOT / "safety_results/cartpole/3600cell_dwm_safety_result.json",
        "real": PROJECT_ROOT / "safety_results/cartpole/3600cell_real_trajectories.npz",
        "real_key": "test_traj",
        "check_dims": (0, 2),
        "outdir": PROJECT_ROOT / "results/cartpole/tube_evaluation",
    },
    "mountain_car": {
        "tube": PROJECT_ROOT / "safety_results/mountain_car/6400cell_dwm_safety_result.json",
        "real": PROJECT_ROOT / "safety_results/mountain_car/6400cell_real_trajectories.npz",
        "real_key": "test_traj",
        "check_dims": (0, 1),
        "outdir": PROJECT_ROOT / "results/mountain_car/tube_evaluation",
    },
    "pendulum": {
        "tube": PROJECT_ROOT / "safety_results/pendulum/5000cell_dwm_safety_result.json",
        "real": PROJECT_ROOT / "safety_results/pendulum/5000cell_real_trajectories.npz",
        "real_key": "test_traj",
        "check_dims": (0, 1),
        "outdir": PROJECT_ROOT / "results/pendulum/tube_evaluation",
    },
    "brake_system": {
        "tube": PROJECT_ROOT / "safety_results/brake_system/1600cell_dwm_safety_result.json",
        "real": PROJECT_ROOT / "safety_results/brake_system/1600cell_real_trajectories.npz",
        "real_key": "test_traj",
        "check_dims": (0, 1),
        "outdir": PROJECT_ROOT / "results/brake_system/tube_evaluation",
    },
}

# Select one default environment. Command-line arguments may override every field.
ACTIVE_ENV = "cartpole"
# ACTIVE_ENV = "mountain_car"
# ACTIVE_ENV = "pendulum"
# ACTIVE_ENV = "brake_system"


@dataclass(frozen=True)
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
            start = float(self.starts[dim])
            stop = float(self.stops[dim])
            count = int(self.nums[dim])
            step = float(self.steps[dim])
            if value < start - EPS or value > stop + EPS:
                return None
            if abs(value - stop) <= EPS:
                local_index = count - 1
            else:
                local_index = int(math.floor((value - start) / step))
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


def signed_interval_margin(value: float, bounds: Sequence[float]) -> float:
    return float(max(
        min(value - low, high - value) for low, high in interval_pairs(bounds)
    ))


def state_signed_margin(
    state: np.ndarray,
    bounds: Sequence[Sequence[float]],
    dims: Sequence[int],
) -> float:
    if len(dims) != 2:
        raise ValueError(f"exactly two check dimensions are required, got {tuple(dims)}")
    if any(dim < 0 or dim >= len(state) or dim >= len(bounds) for dim in dims):
        raise ValueError("selected dimension is missing from state or tube bounds")
    return float(min(
        signed_interval_margin(float(state[dim]), bounds[dim]) for dim in dims
    ))


def trajectory_signed_margin(
    trajectory: np.ndarray,
    bounds_history: Sequence[Sequence[Sequence[float]]],
    dims: Sequence[int],
) -> tuple[float, list[float]]:
    if len(trajectory) != len(bounds_history):
        raise ValueError(
            f"trajectory has {len(trajectory)} states but tube has "
            f"{len(bounds_history)} time steps"
        )
    if not len(trajectory):
        raise ValueError("trajectory is empty")
    state_margins = [
        state_signed_margin(state, bounds, dims)
        for state, bounds in zip(trajectory, bounds_history)
    ]
    return float(min(state_margins)), state_margins


def _grid_from_payload(payload: dict[str, Any], path: Path) -> GridInfo:
    dimensions = payload.get("grid", {}).get("dims", [])
    if not dimensions:
        raise ValueError(f"{path} has no grid dimensions")
    starts = np.array([float(dim["start"]) for dim in dimensions])
    stops = np.array([float(dim["stop"]) for dim in dimensions])
    nums = np.array([int(dim["num"]) for dim in dimensions])
    steps = np.array([
        float(dim.get("step", (stop - start) / count))
        for dim, start, stop, count in zip(dimensions, starts, stops, nums)
    ])
    return GridInfo(
        names=[dim.get("name", f"state_{index}") for index, dim in enumerate(dimensions)],
        starts=starts,
        stops=stops,
        nums=nums,
        steps=steps,
    )


def load_tube(path: Path) -> tuple[dict[str, Any], GridInfo, list[dict[str, Any]]]:
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if "grid" not in payload or "cells" not in payload:
        raise ValueError(f"{path} is missing grid or cells")
    return payload, _grid_from_payload(payload, path), payload["cells"]


def load_trajectories(path: Path, key: str) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as data:
        if key not in data:
            raise KeyError(
                f"key {key!r} is absent from {path}; available keys: {list(data.files)}"
            )
        trajectories = np.asarray(data[key], dtype=float)
    if trajectories.ndim != 3:
        raise ValueError(
            f"trajectories must have shape (N, T+1, state_dim), got {trajectories.shape}"
        )
    return trajectories


def _initial_bounds_contain(
    state: np.ndarray,
    bounds: Sequence[Sequence[float]],
    ndim: int,
) -> bool:
    if len(state) < ndim or len(bounds) < ndim:
        return False
    return all(
        signed_interval_margin(float(state[dim]), bounds[dim]) >= -EPS
        for dim in range(ndim)
    )


def find_cell(
    initial_state: np.ndarray,
    grid: GridInfo,
    cells: Sequence[dict[str, Any]],
) -> tuple[int | None, dict[str, Any] | None]:
    candidate = grid.point_to_linear_index(initial_state)
    if candidate is not None and candidate < len(cells):
        cell = cells[candidate]
        if cell.get("bounds") and _initial_bounds_contain(
            initial_state, cell["bounds"][0], grid.ndim
        ):
            return candidate, cell
    for index, cell in enumerate(cells):
        if cell.get("bounds") and _initial_bounds_contain(
            initial_state, cell["bounds"][0], grid.ndim
        ):
            return index, cell
    return None, None


def evaluate_trajectories(
    trajectories: np.ndarray,
    grid: GridInfo,
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int],
) -> list[dict[str, Any]]:
    if trajectories.ndim != 3:
        raise ValueError(
            f"trajectories must have shape (N, T+1, state_dim), got {trajectories.shape}"
        )
    rows = []
    for trajectory_index, trajectory in enumerate(trajectories):
        cell_index, cell = find_cell(trajectory[0], grid, cells)
        if cell is None:
            raise ValueError(
                f"trajectory {trajectory_index}: initial state does not match a tube cell"
            )
        if "error_msg" in cell:
            raise ValueError(
                f"trajectory {trajectory_index}: tube cell error: {cell['error_msg']}"
            )
        try:
            margin, state_margins = trajectory_signed_margin(
                trajectory, cell["bounds"], dims
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"trajectory {trajectory_index}: {error}") from error
        inside_flags = [value >= -EPS for value in state_margins]
        first_out = next(
            (index for index, inside in enumerate(inside_flags) if not inside),
            None,
        )
        rows.append({
            "traj_index": int(trajectory_index),
            "cell_index": int(cell_index),
            "signed_margin": float(margin),
            "fully_inside": bool(all(inside_flags)),
            "inside_states": int(sum(inside_flags)),
            "compared_states": int(len(inside_flags)),
            "inside_ratio": float(np.mean(inside_flags)),
            "first_out_step": first_out,
            "max_violation": float(max(0.0, -margin)),
        })
    if not rows:
        raise ValueError("at least one trajectory is required")
    return rows


def interval_union_length(bounds: Sequence[float]) -> float:
    merged: list[list[float]] = []
    for low, high in sorted(interval_pairs(bounds)):
        if merged and low <= merged[-1][1] + EPS:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return float(sum(high - low for low, high in merged))


def average_tube_area(
    cells: Sequence[dict[str, Any]], dims: Sequence[int]
) -> float:
    if len(dims) != 2 or any(dim < 0 for dim in dims):
        raise ValueError("exactly two nonnegative check dimensions are required")
    cell_areas = []
    for cell_index, cell in enumerate(cells):
        if "error_msg" in cell:
            raise ValueError(f"tube cell {cell_index} has error: {cell['error_msg']}")
        history = cell.get("bounds", [])
        if len(history) < 2:
            raise ValueError(
                f"tube cell {cell_index} must contain an initial cell and a future state"
            )
        time_areas = []
        for bounds in history[1:]:
            if any(dim >= len(bounds) for dim in dims):
                raise ValueError(
                    f"tube cell {cell_index} does not contain selected dimensions"
                )
            time_areas.append(float(np.prod([
                interval_union_length(bounds[dim]) for dim in dims
            ])))
        cell_areas.append(float(np.mean(time_areas)))
    if not cell_areas:
        raise ValueError("tube must contain at least one cell")
    return float(np.mean(cell_areas))


def calculate_metrics(
    trajectories: np.ndarray,
    grid: GridInfo,
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = evaluate_trajectories(trajectories, grid, cells, dims)
    coverage = float(np.mean([row["fully_inside"] for row in rows]))
    area = average_tube_area(cells, dims)
    return {"coverage": coverage, "area": area}, rows


def write_metrics_table(path: Path, metrics: dict[str, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Coverage", "Area"])
        writer.writeheader()
        writer.writerow({
            "Coverage": f"{100.0 * metrics['coverage']:.2f}%",
            "Area": f"{metrics['area']:.6f}",
        })


def _require_writable_outputs(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing output(s): " + ", ".join(existing)
        )


def run_evaluation(
    *,
    tube_path: Path,
    real_path: Path,
    real_key: str,
    dims: Sequence[int],
    outdir: Path,
    overwrite: bool,
) -> dict[str, float]:
    from tools.plot_real_tube import plot_real_tube

    outdir = Path(outdir)
    table_path = outdir / "tube_metrics.csv"
    plot_path = outdir / "real_vs_tube.png"
    _require_writable_outputs((table_path, plot_path), overwrite)
    payload, grid, cells = load_tube(tube_path)
    trajectories = load_trajectories(real_path, real_key)
    metrics, rows = calculate_metrics(trajectories, grid, cells, dims)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_real_tube(
        plot_path,
        trajectories=trajectories,
        rows=rows,
        cells=cells,
        grid_names=grid.names,
        dims=dims,
        tube_payload=payload,
        coverage=metrics["coverage"],
    )
    write_metrics_table(table_path, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=tuple(EVALUATION_CONFIGS), default=ACTIVE_ENV)
    parser.add_argument("--tube", type=Path, default=None)
    parser.add_argument("--real", type=Path, default=None)
    parser.add_argument("--real-key", default=None)
    parser.add_argument("--check-dims", type=int, nargs=2, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = EVALUATION_CONFIGS[args.env]
    metrics = run_evaluation(
        tube_path=args.tube or config["tube"],
        real_path=args.real or config["real"],
        real_key=args.real_key or config["real_key"],
        dims=tuple(args.check_dims or config["check_dims"]),
        outdir=args.outdir or config["outdir"],
        overwrite=args.overwrite,
    )
    print(f"Coverage: {100.0 * metrics['coverage']:.2f}%")
    print(f"Area: {metrics['area']:.6f}")


if __name__ == "__main__":
    main()
