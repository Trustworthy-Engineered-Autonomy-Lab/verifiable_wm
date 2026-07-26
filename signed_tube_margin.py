#!/usr/bin/env python3
"""Evaluate signed trajectory margins against StarV reachable tubes.

For every checked state, the margin is the smallest signed perpendicular
distance to the tube boundaries in the selected two dimensions.  A positive
margin is inside the tube, zero is on a boundary, and a negative margin is
outside.  Each trajectory is assigned its smallest state margin.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import compare
import conformal


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_DIMS = {
    "cartpole": (0, 2),
    "mountain_car": (0, 1),
    "pendulum": (0, 1),
    "brake_system": (0, 1),
}
EPS = 1e-10

# Edit these three values when running this script directly without arguments.
DEFAULT_ENV = "cartpole"
DEFAULT_DECODER = "dwm"
DEFAULT_CONSTRUCTION = "symbolic"

CASE_DIRS = {
    ("dwm", "symbolic"): "a_dwm_symbolic",
    ("cgan", "symbolic"): "a_cgan_symbolic",
    ("dwm", "sampled"): "b_dwm_sampled",
    ("cgan", "sampled"): "b_cgan_sampled",
}

DWM_INPUTS = {
    "cartpole": (
        "results/cartpole/safety_result_big_cell_a8_lamda01.json",
        "datasets/cartpole/big_cell/real_trajectories.npz",
        "datasets/cartpole/big_cell/dwm_trajectories_saliency.npz",
    ),
    "mountain_car": (
        "results/mountain_car/safety_result_big_cell_best.json",
        "datasets/mountain_car/big_cell_best/real_trajectories.npz",
        "datasets/mountain_car/big_cell_best/dwm_trajectories_saliency.npz",
    ),
    "pendulum": (
        "results/pendulum/safety_result_big_cell_a16_lambda05.json",
        "datasets/pendulum/big_cell/real_trajectories.npz",
        "datasets/pendulum/big_cell/dwm_trajectories_saliency.npz",
    ),
    "brake_system": (
        "safety_results/brake_system/safety_result.json",
        "safety_results/brake_system/real_trajectories.npz",
        "safety_results/brake_system/dwm_trajectories_saliency.npz",
    ),
}

SAMPLED_TUBES = {
    ("cartpole", "dwm"): "results/cartpole/sampled_tube/sampled_reachable_tube_saliency_seed_2025.json",
    ("mountain_car", "dwm"): "results/mountain_car/sampled_tube/sampled_reachable_tube_saliency_seed_2025.json",
    ("pendulum", "dwm"): "results/pendulum/sampled_tube/sampled_reachable_tube_saliency_seed_2025.json",
    ("brake_system", "dwm"): "results/brake_system/sampled_tube/sampled_reachable_tube_old_seed_2025.json",
    ("cartpole", "cgan"): "results/cartpole/sampled_tube/sampled_reachable_tube_g_mlp_seed_2025.json",
    ("mountain_car", "cgan"): "results/mountain_car/sampled_tube/sampled_reachable_tube_g_mlp_seed_2025.json",
    ("pendulum", "cgan"): "results/pendulum/sampled_tube/sampled_reachable_tube_g_mlp_seed_2025.json",
    ("brake_system", "cgan"): "results/brake_system/sampled_tube/sampled_reachable_tube_g_mlp_seed_2025.json",
}


@dataclass(frozen=True)
class Experiment:
    env: str
    decoder: str
    construction: str
    safety: Path
    real: Path
    model: Path
    outdir: Path


def resolve_experiment(
    env: str,
    decoder: str,
    construction: str,
    project_root: Path = PROJECT_ROOT,
) -> Experiment:
    if env not in ENV_DIMS:
        raise ValueError(f"unknown environment: {env}")
    if (decoder, construction) not in CASE_DIRS:
        raise ValueError(
            f"unknown Table III case: decoder={decoder}, construction={construction}"
        )

    if decoder == "dwm":
        symbolic_safety, real, model = DWM_INPUTS[env]
    else:
        symbolic_safety = f"safety_results/{env}/safety_result_g_mlp.json"
        real = f"safety_results/{env}/real_trajectories.npz"
        model = f"safety_results/{env}/dwm_trajectories_g_mlp.npz"

    safety = (
        symbolic_safety
        if construction == "symbolic"
        else SAMPLED_TUBES[(env, decoder)]
    )
    return Experiment(
        env=env,
        decoder=decoder,
        construction=construction,
        safety=project_root / safety,
        real=project_root / real,
        model=project_root / model,
        outdir=project_root / "results" / env / CASE_DIRS[(decoder, construction)],
    )

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


def calibration_gamma(
    calibration_trajectories: np.ndarray,
    grid: GridInfo,
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int],
) -> tuple[float, float]:
    """Calibrate one scalar inflation from the rank-380 signed margin."""
    if calibration_trajectories.ndim != 3:
        raise ValueError(
            "calibration trajectories must have shape (N, T+1, state_dim), "
            f"got {calibration_trajectories.shape}"
        )
    rows = evaluate_set(calibration_trajectories, grid, cells, dims)
    scores = [float(row["signed_margin"]) for row in rows if row["status"] == "valid"]
    if len(rows) != 400 or len(scores) != 400:
        raise ValueError(
            "rank-380 calibration requires exactly 400 valid trajectories, "
            f"got valid={len(scores)}, total={len(rows)}"
        )
    rank_margin = descending_p95(scores)
    return max(0.0, -rank_margin), rank_margin


def inflate_cells(
    cells: Sequence[dict[str, Any]], dims: Sequence[int], epsilons: Sequence[float]
) -> list[dict[str, Any]]:
    """Return copied cells with selected bounds symmetrically expanded."""
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


def build_tube_variants(
    *,
    cells: Sequence[dict[str, Any]],
    grid: GridInfo,
    dims: Sequence[int],
    calibration_trajectories: np.ndarray,
    real_path: Path,
    model_path: Path,
    env: str,
    alpha: float,
    require_matching_starv_config: bool = True,
) -> tuple[dict[str, Sequence[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Build the raw tube and its CP-D and CP-R scalar inflations."""
    cp_d = conformal.calibrate_gamma(
        real_path=real_path,
        dwm_path=model_path,
        dims=dims,
        alpha=alpha,
        split="val",
        circular_dims=conformal.ENV_CIRCULAR_DIMS.get(env, ()),
        horizon=conformal.ENV_HORIZON[env],
        require_matching_starv_config=require_matching_starv_config,
    )
    cp_r_gamma, cp_r_rank_margin = calibration_gamma(
        calibration_trajectories, grid, cells, dims
    )
    variants = {
        "Raw tube": cells,
        "CP-D (inflated)": inflate_cells(
            cells, dims, [float(cp_d["gamma"])] * len(dims)
        ),
        "CP-R (inflated)": inflate_cells(
            cells, dims, [cp_r_gamma] * len(dims)
        ),
    }
    cp_r = {
        "method": "descending rank-380 signed trajectory margin",
        "n": 400,
        "sort": "descending",
        "rank": 380,
        "signed_margin_at_rank": cp_r_rank_margin,
        "gamma": cp_r_gamma,
        "check_dims": [int(dim) for dim in dims],
    }
    return variants, {"cp_d": cp_d, "cp_r": cp_r}


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


def cell_tube_areas(
    cells: Sequence[dict[str, Any]], dims: Sequence[int], delta: float = 0.0
) -> list[float]:
    """Return one time-averaged absolute 2-D tube area for every valid cell."""
    if len(dims) != 2 or any(dim < 0 for dim in dims):
        raise ValueError("exactly two nonnegative check dimensions are required")

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
        values.append(float(np.mean(time_areas)))
    return values


def table_metrics(
    trajectories: np.ndarray,
    grid: GridInfo,
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int],
    delta: float = 0.0,
) -> dict[str, Any]:
    """Summarize Table II robustness and absolute tube-area metrics."""
    rows = evaluate_set(trajectories, grid, cells, dims, delta)
    scores = [float(row["signed_margin"]) for row in rows if row["status"] == "valid"]
    areas = cell_tube_areas(cells, dims, delta)
    covered = sum(score >= -EPS for score in scores)
    return {
        "coverage_rate": float(covered / len(scores)) if scores else None,
        "covered_trajectories": covered,
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
    """Write a readable paper-style CSV plus a full-precision companion CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "tube_table_metrics.csv"
    readable_fields = [
        "Method", "CP score", "CP bound",
        "Coverage rate", "Robustness γ mean", "Robustness γ min",
        "Robustness γ max", "Avg. tube area (mean ± std)",
        "Valid real trajectories", "Valid tube cells",
    ]
    raw_fields = [
        "method", "cp_score", "cp_bound",
        "coverage_rate", "covered_trajectories",
        "robustness_mean", "robustness_minimum", "robustness_maximum",
        "robustness_valid_trajectories", "robustness_total_trajectories",
        "tube_area_mean", "tube_area_std", "tube_area_valid_cells", "tube_area_total_cells",
    ]
    raw_path = output_dir / "tube_table_metrics_raw.csv"
    raw_rows = []
    for method, metrics in metrics_by_method.items():
        robustness = metrics["robustness"]
        tube_area = metrics["tube_area"]
        raw_rows.append({
            "method": method,
            "cp_score": metrics.get("cp_score", "-"),
            "cp_bound": metrics.get("cp_bound"),
            "coverage_rate": metrics["coverage_rate"],
            "covered_trajectories": metrics["covered_trajectories"],
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
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=readable_fields)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({
                "Method": row["method"],
                "CP score": row["cp_score"],
                "CP bound": (
                    "-"
                    if row["cp_bound"] is None
                    else f"{float(row['cp_bound']):.12g}"
                ),
                "Coverage rate": f"{100.0 * row['coverage_rate']:.2f}%",
                "Robustness γ mean": f"{row['robustness_mean']:.6f}",
                "Robustness γ min": f"{row['robustness_minimum']:.6f}",
                "Robustness γ max": f"{row['robustness_maximum']:.6f}",
                "Avg. tube area (mean ± std)": (
                    f"{row['tube_area_mean']:.6f} ± {row['tube_area_std']:.6f}"
                ),
                "Valid real trajectories": (
                    f"{row['robustness_valid_trajectories']}/{row['robustness_total_trajectories']}"
                ),
                "Valid tube cells": f"{row['tube_area_valid_cells']}/{row['tube_area_total_cells']}",
            })
    with raw_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)
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


def write_comparison_plots(
    output_dir: Path,
    safety: dict[str, Any],
    grid: GridInfo,
    tube_variants: dict[str, Sequence[dict[str, Any]]],
    real_trajectories: np.ndarray,
    model_trajectories: np.ndarray,
    dims: Sequence[int],
) -> dict[str, list[str]]:
    """Write real and model comparisons for raw, CP-D, and CP-R tubes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    compare.PLOT_DIMS = tuple(dims)
    compare.CHECK_DIMS = tuple(dims)
    compare.DELTA = 0.0
    compare.MAX_STEPS = None
    suffixes = {
        "Raw tube": "raw",
        "CP-D (inflated)": "cp_d",
        "CP-R (inflated)": "cp_r",
        "raw": "raw",
        "cp_d": "cp_d",
        "cp_r": "cp_r",
    }
    plots: dict[str, list[str]] = {}
    for label, tube_cells in tube_variants.items():
        suffix = suffixes[label]
        plots[suffix] = []
        for title, trajectories, cmap, prefix in (
            ("Real trajectory", real_trajectories, "Oranges", "real"),
            ("Model trajectory", model_trajectories, "Blues", "model"),
        ):
            rows = compare.compare_set(trajectories, grid, tube_cells)
            path = output_dir / f"{prefix}_vs_{suffix}_tube.png"
            compare.plot_set(path, title, trajectories, rows, grid, tube_cells, safety, cmap)
            plots[suffix].append(str(path))
    return plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENV_DIMS, default=DEFAULT_ENV)
    parser.add_argument("--decoder", choices=("dwm", "cgan"), default=DEFAULT_DECODER)
    parser.add_argument(
        "--construction",
        choices=("symbolic", "sampled"),
        default=DEFAULT_CONSTRUCTION,
    )
    parser.add_argument("--safety", type=Path, default=None, help="override tube JSON")
    parser.add_argument("--real", type=Path, default=None, help="override real trajectory NPZ")
    parser.add_argument(
        "--model", "--dwm", dest="model", type=Path, default=None,
        help="override paired DWM/cGAN trajectory NPZ",
    )
    parser.add_argument(
        "--calibration-real", type=Path, default=None,
        help="override real trajectory NPZ used for both calibrations",
    )
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="override the new Table III case directory",
    )
    parser.add_argument("--real-key", default="test_traj")
    parser.add_argument("--model-key", "--dwm-key", dest="model_key", default="test_traj")
    parser.add_argument("--calibration-key", default="val_traj")
    parser.add_argument("--check-dims", type=int, nargs=2, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="allow replacing files in an existing standard result directory",
    )
    return parser.parse_args()


def _require_inputs(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing required input file(s):\n  " + "\n  ".join(missing)
        )


def _require_fresh_output_dir(path: Path, *, overwrite: bool = False) -> None:
    if not overwrite and path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite existing results: {path}"
        )


def main() -> None:
    args = parse_args()
    profile = resolve_experiment(args.env, args.decoder, args.construction)
    safety_path = args.safety or profile.safety
    real_path = args.real or profile.real
    model_path = args.model or profile.model
    calibration_path = args.calibration_real or real_path
    outdir = args.outdir or profile.outdir
    _require_inputs([safety_path, real_path, model_path, calibration_path])
    _require_fresh_output_dir(outdir, overwrite=args.overwrite)

    dims = tuple(args.check_dims) if args.check_dims is not None else ENV_DIMS[args.env]
    grid, cells = load_safety_result(safety_path)
    real_trajectories = load_trajectory(real_path, args.real_key)
    model_trajectories = load_trajectory(model_path, args.model_key)
    calibration_trajectories = load_trajectory(calibration_path, args.calibration_key)
    variants, calibrations = build_tube_variants(
        cells=cells,
        grid=grid,
        dims=dims,
        calibration_trajectories=calibration_trajectories,
        real_path=calibration_path,
        model_path=model_path,
        env=args.env,
        alpha=args.alpha,
        require_matching_starv_config=args.decoder != "cgan",
    )
    metrics_by_method = {
        method: table_metrics(real_trajectories, grid, tube_cells, dims)
        for method, tube_cells in variants.items()
    }
    method_cp = {
        "Raw tube": ("-", None),
        "CP-D (inflated)": ("CP-D", float(calibrations["cp_d"]["gamma"])),
        "CP-R (inflated)": ("CP-R", float(calibrations["cp_r"]["gamma"])),
    }
    for method, (cp_score, cp_bound) in method_cp.items():
        metrics_by_method[method]["cp_score"] = cp_score
        metrics_by_method[method]["cp_bound"] = cp_bound
    table_path = write_table_metrics(metrics_by_method, outdir)
    safety = json.loads(safety_path.read_text(encoding="utf-8"))
    plots = write_comparison_plots(
        outdir, safety, grid, variants,
        real_trajectories, model_trajectories, dims,
    )
    calibrations["cp_r"].update({
        "calibration_path": str(calibration_path),
        "calibration_key": args.calibration_key,
        "evaluation_path": str(real_path),
        "evaluation_key": args.real_key,
        "inflates_both_dimensions_equally": True,
        "inflates_initial_cell": True,
    })
    calibration_path_out = outdir / "tube_calibrations.json"
    outdir.mkdir(parents=True, exist_ok=True)
    calibration_path_out.write_text(
        json.dumps(calibrations, indent=2), encoding="utf-8"
    )
    summary_path = outdir / "signed_tube_margin_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "env": args.env,
                "decoder": args.decoder,
                "construction": args.construction,
                "check_dims": dims,
                "inputs": {
                    "safety": str(safety_path),
                    "real": str(real_path),
                    "model": str(model_path),
                },
                "calibrations": calibrations,
                "table_metrics": metrics_by_method,
                "plots": plots,
            },
            file, indent=2,
        )
    print("CP-D gamma:", calibrations["cp_d"]["gamma"])
    print("CP-R gamma:", calibrations["cp_r"]["gamma"])
    print(f"table metrics: {table_path}")
    print(f"calibrations: {calibration_path_out}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
