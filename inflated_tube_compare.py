#!/usr/bin/env python3
"""Inflate reachable tubes from real-trajectory violations and plot comparisons."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import compare
import signed_tube_margin as stm


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_DIMS = stm.ENV_DIMS

# Edit these paths directly when running this script without terminal options.
# DEFAULT_ENV = "mountain_car"
# DEFAULT_SAFETY_PATH = PROJECT_ROOT / "results/mountain_car/safety_result_cell_100_a16_lambda05.json"
# DEFAULT_REAL_TRAJ_PATH = PROJECT_ROOT / "datasets/mountain_car/data_cell_100/real_trajectories.npz"
# DEFAULT_DWM_TRAJ_PATH = PROJECT_ROOT / "datasets/mountain_car/data_cell_100/dwm_trajectories_saliency.npz"
# DEFAULT_OUT_DIR = PROJECT_ROOT / "results/mountain_car/inflated_tube_compare"

# DEFAULT_ENV = "mountain_car"
# DEFAULT_SAFETY_PATH = PROJECT_ROOT / "results/mountain_car/safety_result_cell_100_a16_lambda05.json"
# DEFAULT_REAL_TRAJ_PATH = PROJECT_ROOT / "datasets/mountain_car/data_cell_100/real_trajectories.npz"
# DEFAULT_DWM_TRAJ_PATH = PROJECT_ROOT / "datasets/mountain_car/data_cell_100/dwm_trajectories_saliency.npz"
# DEFAULT_OUT_DIR = PROJECT_ROOT / "results/mountain_car/inflated_tube_compare"

DEFAULT_ENV = "mountain_car"
DEFAULT_SAFETY_PATH = PROJECT_ROOT / "results/mountain_car/safety_result_cell_100_a16_lambda05.json"
DEFAULT_REAL_TRAJ_PATH = PROJECT_ROOT / "datasets/mountain_car/data_cell_100/real_trajectories.npz"
DEFAULT_DWM_TRAJ_PATH = PROJECT_ROOT / "datasets/mountain_car/data_cell_100/dwm_trajectories_saliency.npz"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results/mountain_car/inflated_tube_compare"


def trajectory_dimension_margins(
    trajectory: np.ndarray,
    bounds_history: Sequence[Sequence[Sequence[float]]],
    dims: Sequence[int],
) -> np.ndarray:
    """Return each selected dimension's smallest signed boundary margin over time."""
    if len(trajectory) != len(bounds_history):
        raise ValueError(
            f"trajectory has {len(trajectory)} states but tube has {len(bounds_history)} time steps"
        )
    if not len(trajectory):
        raise ValueError("trajectory is empty")
    return np.asarray([
        min(
            stm.signed_interval_margin(float(state[dim]), bounds[dim])
            for state, bounds in zip(trajectory, bounds_history)
        )
        for dim in dims
    ], dtype=float)


def calibration_epsilons(
    real_trajectories: np.ndarray,
    grid: stm.GridInfo,
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int],
) -> np.ndarray:
    """Compute one mean-plus-std violation inflation amount per selected dimension."""
    if real_trajectories.ndim != 3:
        raise ValueError(f"real trajectories must have shape (N, T+1, state_dim), got {real_trajectories.shape}")

    violations = []
    for trajectory_index, trajectory in enumerate(real_trajectories):
        _, cell = stm.find_cell(trajectory[0], grid, cells)
        if cell is None:
            raise ValueError(f"real trajectory {trajectory_index} does not match a StarV cell")
        if "error_msg" in cell:
            raise ValueError(f"real trajectory {trajectory_index} has StarV cell error: {cell['error_msg']}")
        margins = trajectory_dimension_margins(trajectory, cell.get("bounds", []), dims)
        violations.append(np.maximum(0.0, -margins))
    if not violations:
        raise ValueError("no real trajectories were supplied")

    values = np.asarray(violations, dtype=float)
    return np.mean(values, axis=0) + np.std(values, axis=0)


def inflate_cells(
    cells: Sequence[dict[str, Any]], dims: Sequence[int], epsilons: Sequence[float]
) -> list[dict[str, Any]]:
    """Copy cells and symmetrically expand every history bound, including t=0."""
    if len(dims) != len(epsilons):
        raise ValueError("each selected dimension needs one inflation epsilon")
    inflated = copy.deepcopy(cells)
    for cell in inflated:
        for state_bounds in cell.get("bounds", []):
            for dim, epsilon in zip(dims, epsilons):
                if dim < 0 or dim >= len(state_bounds):
                    raise ValueError(f"tube bounds do not contain selected dimension {dim}")
                values = np.asarray(state_bounds[dim], dtype=float).reshape(-1)
                if values.size < 2 or values.size % 2:
                    raise ValueError("tube bounds must contain low/high pairs")
                values[0::2] -= float(epsilon)
                values[1::2] += float(epsilon)
                state_bounds[dim] = values.tolist()
    return inflated


def make_summary(epsilons: Sequence[float], dims: Sequence[int], names: Sequence[str]) -> dict[str, Any]:
    return {
        "method": "per-dimension mean(max(0, -trajectory_min_signed_margin)) + std",
        "calibration_dataset": "real",
        "check_dims": [int(dim) for dim in dims],
        "epsilons": {
            str(names[dim] if dim < len(names) else f"state_{dim}"): float(epsilon)
            for dim, epsilon in zip(dims, epsilons)
        },
        "inflates_initial_cell": True,
    }


def load_safety_payload(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENV_DIMS, default=DEFAULT_ENV)
    parser.add_argument("--safety", type=Path, default=DEFAULT_SAFETY_PATH)
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL_TRAJ_PATH)
    parser.add_argument("--dwm", type=Path, default=DEFAULT_DWM_TRAJ_PATH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--real-key", default="test_traj")
    parser.add_argument("--dwm-key", default="test_traj")
    parser.add_argument("--check-dims", type=int, nargs=2, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dims = tuple(args.check_dims) if args.check_dims is not None else ENV_DIMS[args.env]
    safety = load_safety_payload(args.safety)
    grid, cells = stm.load_safety_result(args.safety)
    real_trajectories = stm.load_trajectory(args.real, args.real_key)
    dwm_trajectories = stm.load_trajectory(args.dwm, args.dwm_key)

    epsilons = calibration_epsilons(real_trajectories, grid, cells, dims)
    inflated_cells = inflate_cells(cells, dims, epsilons)
    summary = make_summary(epsilons, dims, grid.names)

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / "inflated_tube_calibration.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    compare.PLOT_DIMS = dims
    compare.CHECK_DIMS = dims
    compare.DELTA = 0.0
    compare.MAX_STEPS = None
    for title, trajectories, cmap_name, output_name in (
        ("Real trajectory", real_trajectories, "Oranges", "real_vs_inflated_reachable_tube.png"),
        ("DWM trajectory", dwm_trajectories, "Blues", "dwm_vs_inflated_reachable_tube.png"),
    ):
        rows = compare.compare_set(trajectories, grid, inflated_cells)
        compare.plot_set(args.outdir / output_name, title, trajectories, rows, grid, inflated_cells, safety, cmap_name)

    print("inflation epsilons:", summary["epsilons"])
    print(f"summary: {summary_path}")
    print(f"plots: {args.outdir / 'real_vs_inflated_reachable_tube.png'}")
    print(f"       {args.outdir / 'dwm_vs_inflated_reachable_tube.png'}")


if __name__ == "__main__":
    main()
