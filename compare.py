#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory vs reachable tube plot only.

This version has editable default paths, so it can run without
entering paths every time. Terminal arguments can still override
any default path.

It automatically plots the worst-containment and best-containment
trajectories. Manual trajectory index selection is intentionally
removed.

Environment defaults:
- cartpole     : plot/check dimensions (0, 2) = x and theta
- mountain_car : plot/check dimensions (0, 1) = position and velocity
- pendulum     : plot/check dimensions (0, 1) = theta and omega

Outputs:
- real_vs_reachable_tube.png
- dwm_vs_reachable_tube.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# ============================================================
# Default paths.
#
# If you run `python compare.py` without terminal
# path arguments, these defaults are used. You can still override
# any path from the terminal with --safety / --real / --dwm / --outdir.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# DEFAULT_SAFETY_PATH = PROJECT_ROOT / "results/cartpole/safety_result_cell_100_a8_lamda01.json"
# DEFAULT_REAL_TRAJ_PATH = PROJECT_ROOT / "datasets/cartpole/data_cell_100/real_trajectories.npz"
# DEFAULT_DWM_TRAJ_PATH = PROJECT_ROOT / "datasets/cartpole/data_cell_100/dwm_trajectories_saliency.npz"
# DEFAULT_OUT_DIR = PROJECT_ROOT / "results/cartpole/compare_plot"

# DEFAULT_SAFETY_PATH = PROJECT_ROOT / "results/mountain_car/safety_result_cell_100_a16_lambda05.json"
# DEFAULT_REAL_TRAJ_PATH = PROJECT_ROOT / "datasets/mountain_car/data_cell_100/real_trajectories.npz"
# DEFAULT_DWM_TRAJ_PATH = PROJECT_ROOT / "datasets/mountain_car/data_cell_100/dwm_trajectories_saliency.npz"
# DEFAULT_OUT_DIR = PROJECT_ROOT / "results/mountain_car/compare_plot"

DEFAULT_SAFETY_PATH = PROJECT_ROOT / "results/pendulum/safety_result_cell_100_a16_lambda05.json"
DEFAULT_REAL_TRAJ_PATH = PROJECT_ROOT / "datasets/pendulum/data_cell_100/real_trajectories.npz"
DEFAULT_DWM_TRAJ_PATH = PROJECT_ROOT / "datasets/pendulum/data_cell_100/dwm_trajectories_saliency.npz"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results/pendulum/compare_plot"

SAFETY_PATH: Path = DEFAULT_SAFETY_PATH
REAL_TRAJ_PATH: Path = DEFAULT_REAL_TRAJ_PATH
DWM_TRAJ_PATH: Path = DEFAULT_DWM_TRAJ_PATH
OUT_DIR: Path = DEFAULT_OUT_DIR

REAL_KEY = "test_traj"
DWM_KEY = "test_traj"

# Environment-specific dimensions used by default.
# CartPole compares cart position x (dim 0) and pole angle theta (dim 2).
# MountainCar and Pendulum continue to compare dimensions (0, 1).
# DEFAULT_ENV = "cartpole"
# DEFAULT_ENV = "mountain_car"
DEFAULT_ENV = "pendulum"
ENV_DEFAULT_DIMS = {
    "cartpole": {
        "plot_dims": (0, 2),
        "check_dims": (0, 2),
        "description": "cart position x vs pole angle theta",
    },
    "mountain_car": {
        "plot_dims": (0, 1),
        "check_dims": (0, 1),
        "description": "position vs velocity",
    },
    "pendulum": {
        "plot_dims": (0, 1),
        "check_dims": (0, 1),
        "description": "theta vs omega",
    },
}

ENV_NAME = DEFAULT_ENV
PLOT_DIMS = ENV_DEFAULT_DIMS[DEFAULT_ENV]["plot_dims"]
CHECK_DIMS = ENV_DEFAULT_DIMS[DEFAULT_ENV]["check_dims"]
MAX_STEPS: Optional[int] = None
DELTA = 0.0
PRINT_KEYS_ONLY = False
DPI = 230


# ============================================================
# Implementation below. Usually no need to edit.
# ============================================================

EPS = 1e-10


@dataclass
class GridInfo:
    names: List[str]
    starts: np.ndarray
    stops: np.ndarray
    nums: np.ndarray
    steps: np.ndarray

    @property
    def ndim(self) -> int:
        return len(self.names)

    def linear_index(self, indices: Sequence[int]) -> int:
        idx = 0
        for i, n in zip(indices, self.nums):
            idx = idx * int(n) + int(i)
        return int(idx)

    def point_to_indices(self, point: np.ndarray) -> Optional[List[int]]:
        point = np.asarray(point, dtype=float)
        if point.shape[0] < self.ndim:
            return None

        out = []
        for d in range(self.ndim):
            x = float(point[d])
            start = float(self.starts[d])
            stop = float(self.stops[d])
            step = float(self.steps[d])
            num = int(self.nums[d])

            if x < start - EPS or x > stop + EPS:
                return None

            if abs(x - stop) <= EPS:
                i = num - 1
            else:
                i = int(math.floor((x - start) / step))
                i = max(0, min(num - 1, i))
            out.append(i)

        return out

    def point_to_linear_index(self, point: np.ndarray) -> Optional[int]:
        idxs = self.point_to_indices(point)
        if idxs is None:
            return None
        return self.linear_index(idxs)


def pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def npz_keys(path: Path) -> List[str]:
    with np.load(path, allow_pickle=False) as z:
        return list(z.files)


def load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(z.files)}")
        return np.asarray(z[key], dtype=float)


def load_safety_result(path: Path) -> Tuple[Dict[str, Any], GridInfo, List[Dict[str, Any]]]:
    data = load_json(path)
    if "grid" not in data or "cells" not in data:
        raise ValueError(f"{path} does not look like safety_result.json: missing grid/cells")

    dims = data["grid"]["dims"]
    names = [d.get("name", f"state_{i}") for i, d in enumerate(dims)]
    starts = np.array([float(d["start"]) for d in dims], dtype=float)
    stops = np.array([float(d["stop"]) for d in dims], dtype=float)
    nums = np.array([int(d["num"]) for d in dims], dtype=int)
    steps = np.array([
        float(d.get("step", (float(d["stop"]) - float(d["start"])) / int(d["num"])))
        for d in dims
    ], dtype=float)

    return data, GridInfo(names, starts, stops, nums, steps), data["cells"]


def ensure_traj_shape(arr: np.ndarray, label: str) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"{label} should have shape (N, T+1, dim), got {arr.shape}")
    return arr


def dim_intervals(dim_bounds: Sequence[float]) -> List[Tuple[float, float]]:
    """
    Normal bound: [low, high]
    Wrapped bound, often pendulum theta: [low1, high1, low2, high2]
    """
    arr = np.asarray(dim_bounds, dtype=float).reshape(-1)
    if arr.size < 2:
        return []
    if arr.size % 2 == 1:
        arr = arr[:2]

    intervals = []
    for i in range(0, arr.size, 2):
        lo, hi = float(arr[i]), float(arr[i + 1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        intervals.append((min(lo, hi), max(lo, hi)))
    return intervals


def one_dim_inside(x: float, dim_bounds: Sequence[float], delta: float) -> bool:
    for lo, hi in dim_intervals(dim_bounds):
        if lo - delta - EPS <= x <= hi + delta + EPS:
            return True
    return False


def one_dim_violation(x: float, dim_bounds: Sequence[float], delta: float) -> float:
    intervals = dim_intervals(dim_bounds)
    if not intervals:
        return float("inf")

    dist = []
    for lo, hi in intervals:
        lo -= delta
        hi += delta
        if lo - EPS <= x <= hi + EPS:
            dist.append(0.0)
        elif x < lo:
            dist.append(lo - x)
        else:
            dist.append(x - hi)
    return float(min(dist))


def state_in_bounds(state: np.ndarray, bounds: Sequence[Sequence[float]], dims: Sequence[int], delta: float) -> bool:
    for d in dims:
        if d >= len(state) or d >= len(bounds):
            return False
        if not one_dim_inside(float(state[d]), bounds[d], delta):
            return False
    return True


def max_violation(state: np.ndarray, bounds: Sequence[Sequence[float]], dims: Sequence[int], delta: float) -> float:
    values = []
    for d in dims:
        if d >= len(state) or d >= len(bounds):
            values.append(float("inf"))
        else:
            values.append(one_dim_violation(float(state[d]), bounds[d], delta))
    return float(max(values)) if values else 0.0


def initial_bound_contains(init_state: np.ndarray, initial_bounds: Sequence[Sequence[float]], grid: GridInfo) -> bool:
    dims = list(range(grid.ndim))
    return state_in_bounds(init_state, initial_bounds, dims, delta=0.0)


def find_cell(init_state: np.ndarray, grid: GridInfo, cells: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    idx = grid.point_to_linear_index(init_state)
    if idx is not None and 0 <= idx < len(cells):
        cell = cells[idx]
        bounds = cell.get("bounds", [])
        if bounds and initial_bound_contains(init_state, bounds[0], grid):
            return idx, cell

    # Fallback for ordering mismatch or floating point edge cases.
    for j, cell in enumerate(cells):
        bounds = cell.get("bounds", [])
        if bounds and initial_bound_contains(init_state, bounds[0], grid):
            return j, cell

    return None, None


def desired_state_count(traj_len: int) -> int:
    if MAX_STEPS is None:
        return traj_len
    return min(traj_len, int(MAX_STEPS) + 1)


def compare_traj(traj: np.ndarray, traj_index: int, grid: GridInfo, cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    init_state = traj[0]
    cell_idx, cell = find_cell(init_state, grid, cells)
    desired = desired_state_count(len(traj))

    row = {
        "traj_index": int(traj_index),
        "cell_index": "" if cell_idx is None else int(cell_idx),
        "cell_status": "no_cell",
        "desired_states": int(desired),
        "checked_states": 0,
        "inside_states": 0,
        "inside_ratio": 0.0,
        "fully_inside": False,
        "first_out_step": "",
        "max_violation": np.nan,
    }

    if cell is None:
        return row

    if "error_msg" in cell:
        row["cell_status"] = "error"
    elif cell.get("result") is True:
        row["cell_status"] = "safe"
    elif cell.get("result") is False:
        row["cell_status"] = "unsafe"
    else:
        row["cell_status"] = "unknown"

    bounds = cell.get("bounds", [])
    checked = min(len(bounds), desired)
    row["checked_states"] = int(checked)

    inside_flags = []
    violations = []
    for t in range(checked):
        inside = state_in_bounds(traj[t], bounds[t], CHECK_DIMS, DELTA)
        inside_flags.append(inside)
        violations.append(max_violation(traj[t], bounds[t], CHECK_DIMS, DELTA))

    inside_count = int(np.sum(inside_flags)) if inside_flags else 0
    row["inside_states"] = inside_count
    row["inside_ratio"] = float(inside_count / desired) if desired else 0.0
    row["max_violation"] = float(np.max(violations)) if violations else np.nan

    first_out = None
    for t, inside in enumerate(inside_flags):
        if not inside:
            first_out = t
            break
    if first_out is None and checked < desired:
        first_out = checked

    row["first_out_step"] = "" if first_out is None else int(first_out)
    row["fully_inside"] = (
        row["cell_status"] != "error"
        and checked == desired
        and first_out is None
        and inside_count == desired
    )
    return row


def compare_set(trajs: np.ndarray, grid: GridInfo, cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [compare_traj(trajs[i], i, grid, cells) for i in range(len(trajs))]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    fully = sum(bool(r["fully_inside"]) for r in rows)
    ratios = [float(r["inside_ratio"]) for r in rows]
    return {
        "total": total,
        "fully": int(fully),
        "rate": float(fully / total) if total else 0.0,
        "mean_ratio": float(np.mean(ratios)) if ratios else 0.0,
    }


def _finite_violation(row: Dict[str, Any]) -> float:
    value = float(row.get("max_violation", 0.0))
    if not np.isfinite(value):
        return float("inf")
    return value


def choose_best_worst(rows: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
    """Return [(worst_idx, label), (best_idx, label)]."""
    if not rows:
        return []

    # Worst: lower containment ratio is worse; for ties, larger violation is worse.
    worst_row = min(
        rows,
        key=lambda r: (
            float(r["inside_ratio"]),
            -_finite_violation(r),
            int(r["traj_index"]),
        ),
    )

    # Best: higher containment ratio is better; fully contained is preferred;
    # for ties, smaller violation is better.
    best_row = max(
        rows,
        key=lambda r: (
            float(r["inside_ratio"]),
            bool(r["fully_inside"]),
            -_finite_violation(r),
            -int(r["traj_index"]),
        ),
    )

    return [
        (int(worst_row["traj_index"]), "Worst containment"),
        (int(best_row["traj_index"]), "Best containment"),
    ]


def draw_tube(ax: Any, bounds: Sequence[Any], max_states: int, cmap_name: str) -> Optional[ScalarMappable]:
    if not bounds:
        return None

    xdim, ydim = PLOT_DIMS
    n = min(len(bounds), max_states)
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=max(n - 1, 1))
    first = True

    for t in range(n):
        b = bounds[t]
        if len(b) <= max(xdim, ydim):
            continue
        for x0, x1 in dim_intervals(b[xdim]):
            for y0, y1 in dim_intervals(b[ydim]):
                ax.add_patch(Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=cmap(norm(t)),
                    linewidth=1.2,
                    alpha=0.55,
                    label="reachable tube" if first else None,
                ))
                first = False

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


def draw_initial_cell(ax: Any, initial_bounds: Sequence[Any]) -> None:
    xdim, ydim = PLOT_DIMS
    if len(initial_bounds) <= max(xdim, ydim):
        return

    first = True
    for x0, x1 in dim_intervals(initial_bounds[xdim]):
        for y0, y1 in dim_intervals(initial_bounds[ydim]):
            ax.add_patch(Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="green",
                linewidth=2.0,
                label="initial cell" if first else None,
            ))
            first = False


def add_goal_lines(ax: Any, safety: Dict[str, Any], grid: GridInfo) -> None:
    kwargs = safety.get("verifier", {}).get("kwargs", {})
    xdim, ydim = PLOT_DIMS

    if "goal_position_threshold" in kwargs:
        goal = float(kwargs["goal_position_threshold"])
        pos_dim = 0
        if pos_dim == xdim:
            ax.axvline(goal, linestyle="--", linewidth=1.1, label=f"goal={goal:g}")
        elif pos_dim == ydim:
            ax.axhline(goal, linestyle="--", linewidth=1.1, label=f"goal={goal:g}")

    if "goal_angle_threshold" in kwargs:
        goal = float(kwargs["goal_angle_threshold"])
        angle_dim = 2 if grid.ndim >= 4 else 0
        if angle_dim == xdim:
            ax.axvline(-goal, linestyle="--", linewidth=1.0, label=f"goal=±{goal:g}")
            ax.axvline(goal, linestyle="--", linewidth=1.0)
        elif angle_dim == ydim:
            ax.axhline(-goal, linestyle="--", linewidth=1.0, label=f"goal=±{goal:g}")
            ax.axhline(goal, linestyle="--", linewidth=1.0)


def plot_set(
    out_path: Path,
    title_name: str,
    trajs: np.ndarray,
    rows: List[Dict[str, Any]],
    grid: GridInfo,
    cells: List[Dict[str, Any]],
    safety: Dict[str, Any],
    cmap_name: str,
) -> None:
    row_by_idx = {int(r["traj_index"]): r for r in rows}
    panels = choose_best_worst(rows)
    if not panels:
        print(f"[Warning] no trajectories for {title_name}")
        return

    while len(panels) < 2:
        panels.append(panels[-1])
    panels = panels[:2]

    summary = summarize(rows)
    xdim, ydim = PLOT_DIMS

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    color = "tab:red" if title_name.lower().startswith("real") else "tab:blue"

    for ax, (idx, panel_label) in zip(axes, panels):
        traj = trajs[idx]
        row = row_by_idx[idx]
        _, cell = find_cell(traj[0], grid, cells)
        bounds = None if cell is None else cell.get("bounds", [])
        desired = int(row["desired_states"])

        if bounds:
            sm = draw_tube(ax, bounds, max_states=min(len(bounds), desired), cmap_name=cmap_name)
            if sm is not None:
                cbar = fig.colorbar(sm, ax=ax, pad=0.02)
                cbar.set_label("time step")
            draw_initial_cell(ax, bounds[0])

        plot_traj = traj[:desired]
        ax.plot(
            plot_traj[:, xdim],
            plot_traj[:, ydim],
            marker="s",
            markersize=3.0,
            linewidth=1.4,
            color=color,
            label=title_name,
        )
        ax.plot(
            plot_traj[0, xdim],
            plot_traj[0, ydim],
            marker="o",
            markersize=6.0,
            color="green",
            linestyle="None",
            label="initial state",
        )

        add_goal_lines(ax, safety, grid)

        status = "FULLY CONTAINED" if row["fully_inside"] else "NOT fully contained"
        first_out = "None" if row["first_out_step"] == "" else str(row["first_out_step"])
        box = (
            f"{status}\n"
            f"inside: {row['inside_states']}/{row['desired_states']} ({pct(row['inside_ratio'])})\n"
            f"checked: {row['checked_states']}\n"
            f"first out: {first_out}\n"
            f"max violation: {float(row['max_violation']):.4g}\n"
            f"cell: {row['cell_index']} ({row['cell_status']})"
        )
        ax.text(
            0.02,
            0.02,
            box,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="0.75"),
        )

        ax.set_title(panel_label, fontsize=10)
        ax.set_xlabel(grid.names[xdim] if xdim < len(grid.names) else f"state_{xdim}")
        ax.set_ylabel(grid.names[ydim] if ydim < len(grid.names) else f"state_{ydim}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    horizon = f"max_steps={MAX_STEPS}" if MAX_STEPS is not None else "full trajectory"
    fig.suptitle(
        f"{title_name} vs reachable tube | "
        f"fully contained: {summary['fully']}/{summary['total']} ({pct(summary['rate'])}) | "
        f"mean step containment: {pct(summary['mean_ratio'])} | "
        f"plot dims={PLOT_DIMS} | check dims={CHECK_DIMS} | {horizon}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate real/DWM trajectory vs reachable-tube comparison figures. "
            "Default paths are built in, and terminal arguments can override them."
        )
    )

    # Environment selects the default state dimensions used for plotting and containment.
    parser.add_argument(
        "--env",
        choices=["cartpole", "mountain_car", "pendulum"],
        default=DEFAULT_ENV,
        help=(
            "Environment used to select default dimensions: "
            "cartpole=(0,2), mountain_car=(0,1), pendulum=(0,1). "
            f"Default: {DEFAULT_ENV}"
        ),
    )

    # Optional paths. If omitted, the editable defaults at the top are used.
    parser.add_argument("--safety", type=Path, default=None,
                        help=f"Path to safety result. Default: {DEFAULT_SAFETY_PATH}")
    parser.add_argument("--real", type=Path, default=None,
                        help=f"Path to real trajectories. Default: {DEFAULT_REAL_TRAJ_PATH}")
    parser.add_argument("--dwm", type=Path, default=None,
                        help=f"Path to DWM trajectories. Default: {DEFAULT_DWM_TRAJ_PATH}")
    parser.add_argument("--outdir", type=Path, default=None,
                        help=f"Output directory for figures. Default: {DEFAULT_OUT_DIR}")

    # NPZ keys.
    parser.add_argument("--real-key", default="test_traj",
                        help="Array key in real_trajectories.npz. Default: test_traj")
    parser.add_argument("--dwm-key", default="test_traj",
                        help="Array key in the DWM trajectory NPZ. Default: test_traj")

    # Plot and checking settings.
    parser.add_argument(
        "--plot-dims",
        type=int,
        nargs=2,
        default=None,
        help=(
            "Override the environment default plotting dimensions. "
            "Defaults: cartpole=0 2; mountain_car/pendulum=0 1."
        ),
    )
    parser.add_argument(
        "--check-dims",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Override the environment default containment dimensions. "
            "Defaults: cartpole=0 2; mountain_car/pendulum=0 1."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Compare only the first K transition steps, checking states 0..K. Default: full trajectory.")
    parser.add_argument("--delta", type=float, default=0.0,
                        help="Inflate reachable tube bounds by this amount during containment checking. Default: 0.0")
    # Utility.
    parser.add_argument("--print-keys", action="store_true",
                        help="Only print keys inside real/dwm npz files, then exit.")
    parser.add_argument("--dpi", type=int, default=230,
                        help="Figure DPI. Default: 230")

    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    global SAFETY_PATH, REAL_TRAJ_PATH, DWM_TRAJ_PATH, OUT_DIR
    global REAL_KEY, DWM_KEY, ENV_NAME, PLOT_DIMS, CHECK_DIMS, MAX_STEPS
    global DELTA, PRINT_KEYS_ONLY, DPI

    SAFETY_PATH = args.safety or DEFAULT_SAFETY_PATH
    REAL_TRAJ_PATH = args.real or DEFAULT_REAL_TRAJ_PATH
    DWM_TRAJ_PATH = args.dwm or DEFAULT_DWM_TRAJ_PATH
    OUT_DIR = args.outdir or DEFAULT_OUT_DIR

    REAL_KEY = args.real_key
    DWM_KEY = args.dwm_key

    ENV_NAME = args.env
    env_defaults = ENV_DEFAULT_DIMS[ENV_NAME]

    selected_plot_dims = args.plot_dims if args.plot_dims is not None else env_defaults["plot_dims"]
    selected_check_dims = args.check_dims if args.check_dims is not None else env_defaults["check_dims"]

    PLOT_DIMS = tuple(int(x) for x in selected_plot_dims)
    CHECK_DIMS = tuple(int(x) for x in selected_check_dims)
    MAX_STEPS = args.max_steps
    DELTA = float(args.delta)
    PRINT_KEYS_ONLY = bool(args.print_keys)
    DPI = int(args.dpi)

def validate_paths() -> None:
    for label, path in [
        ("SAFETY_PATH", SAFETY_PATH),
        ("REAL_TRAJ_PATH", REAL_TRAJ_PATH),
        ("DWM_TRAJ_PATH", DWM_TRAJ_PATH),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")


def main() -> None:
    args = parse_args()
    apply_args(args)
    validate_paths()

    if PRINT_KEYS_ONLY:
        print("real keys:", npz_keys(REAL_TRAJ_PATH))
        print("dwm keys :", npz_keys(DWM_TRAJ_PATH))
        return

    safety, grid, cells = load_safety_result(SAFETY_PATH)
    real_traj = ensure_traj_shape(load_npz_array(REAL_TRAJ_PATH, REAL_KEY), f"real[{REAL_KEY}]")
    dwm_traj = ensure_traj_shape(load_npz_array(DWM_TRAJ_PATH, DWM_KEY), f"dwm[{DWM_KEY}]")

    if max(PLOT_DIMS) >= real_traj.shape[2] or max(PLOT_DIMS) >= dwm_traj.shape[2]:
        raise ValueError(f"PLOT_DIMS={PLOT_DIMS} exceeds trajectory dimension")
    if max(CHECK_DIMS) >= grid.ndim:
        raise ValueError(f"CHECK_DIMS={CHECK_DIMS} exceeds grid dimension {grid.ndim}: {grid.names}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("========== Loaded ==========")
    print(f"env    : {ENV_NAME} ({ENV_DEFAULT_DIMS[ENV_NAME]['description']})")
    print(f"safety : {SAFETY_PATH}")
    print(f"real   : {REAL_TRAJ_PATH} | key={REAL_KEY} | shape={real_traj.shape}")
    print(f"dwm    : {DWM_TRAJ_PATH} | key={DWM_KEY} | shape={dwm_traj.shape}")
    print(f"outdir : {OUT_DIR}")
    print(f"grid   : {grid.names} | nums={grid.nums.tolist()} | cells={len(cells)}")
    print(f"plot   : {PLOT_DIMS}")
    print(f"check  : {CHECK_DIMS}")
    print(f"steps  : {MAX_STEPS}")

    real_rows = compare_set(real_traj, grid, cells)
    dwm_rows = compare_set(dwm_traj, grid, cells)

    for name, rows in [("Real trajectory", real_rows), ("DWM trajectory", dwm_rows)]:
        s = summarize(rows)
        print(f"\n[{name}]")
        print(f"  checked trajectories : {s['total']}")
        print(f"  fully contained      : {s['fully']}/{s['total']} ({pct(s['rate'])})")
        print(f"  mean step containment: {pct(s['mean_ratio'])}")

    real_out = OUT_DIR / "real_vs_reachable_tube.png"
    dwm_out = OUT_DIR / "dwm_vs_reachable_tube.png"

    plot_set(real_out, "Real trajectory", real_traj, real_rows, grid, cells, safety, cmap_name="Oranges")
    plot_set(dwm_out, "DWM trajectory", dwm_traj, dwm_rows, grid, cells, safety, cmap_name="Blues")

    print(f"\n[Saved] {real_out}")
    print(f"[Saved] {dwm_out}")


if __name__ == "__main__":
    main()
