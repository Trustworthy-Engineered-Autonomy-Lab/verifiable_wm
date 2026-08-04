#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trajectory vs reachable-tube containment checking and plotting.

signed_tube_margin.py drives this module: it sets the module-level PLOT_DIMS /
CHECK_DIMS / DELTA / MAX_STEPS for the environment under test, calls
compare_set() to score every trajectory against its own tube cell, and
plot_set() to render the worst- and best-containment trajectories side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# The caller sets these before calling compare_set()/plot_set().
# CartPole compares cart position x (dim 0) against pole angle theta (dim 2);
# MountainCar, Pendulum and the braking system all compare dims (0, 1).
PLOT_DIMS: Tuple[int, int] = (0, 1)
CHECK_DIMS: Tuple[int, ...] = (0, 1)
DELTA = 0.0
MAX_STEPS: Optional[int] = None
DPI = 230

EPS = 1e-10


def pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


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
        "compared_states": 0,
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
    row["compared_states"] = int(checked)
    row["checked_states"] = int(checked)

    inside_flags = []
    violations = []
    for t in range(checked):
        inside = state_in_bounds(traj[t], bounds[t], CHECK_DIMS, DELTA)
        inside_flags.append(inside)
        violations.append(max_violation(traj[t], bounds[t], CHECK_DIMS, DELTA))

    inside_count = int(np.sum(inside_flags)) if inside_flags else 0
    row["inside_states"] = inside_count
    row["inside_ratio"] = float(inside_count / checked) if checked else 0.0
    row["max_violation"] = float(np.max(violations)) if violations else np.nan

    first_out = None
    for t, inside in enumerate(inside_flags):
        if not inside:
            first_out = t
            break
    row["first_out_step"] = "" if first_out is None else int(first_out)
    row["fully_inside"] = (
        row["cell_status"] != "error"
        and checked > 0
        and first_out is None
        and inside_count == checked
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
    best_candidates = [
        row for row in rows
        if int(row["traj_index"]) != int(worst_row["traj_index"])
    ] or rows
    best_row = max(
        best_candidates,
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


def resolve_panels(
    rows: List[Dict[str, Any]],
    panel_indices: Optional[Sequence[int]] = None,
) -> List[Tuple[int, str]]:
    if panel_indices is None:
        return choose_best_worst(rows)
    indices = tuple(int(index) for index in panel_indices)
    if len(indices) != 2:
        raise ValueError("exactly two panel indices are required")
    if indices[0] == indices[1]:
        raise ValueError("panel indices must be distinct")
    available = {int(row["traj_index"]) for row in rows}
    missing = [index for index in indices if index not in available]
    if missing:
        raise ValueError(f"panel indices are absent from trajectories: {missing}")
    return [(index, f"Trajectory {index}") for index in indices]


def draw_tube(
    ax: Any,
    bounds: Sequence[Any],
    max_states: int,
    cmap_name: str,
    delta: float = 0.0,
) -> Optional[ScalarMappable]:
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
                x0 -= delta
                x1 += delta
                y0 -= delta
                y1 += delta
                ax.add_patch(Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=cmap(norm(t)),
                    linewidth=1.2,
                    alpha=0.55,
                    label=("inflated reachable tube" if delta else "reachable tube") if first else None,
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
    panel_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    row_by_idx = {int(r["traj_index"]): r for r in rows}
    panels = resolve_panels(rows, panel_indices)
    if not panels:
        print(f"[Warning] no trajectories for {title_name}")
        return []

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
        compared = int(row["compared_states"])

        if bounds:
            sm = draw_tube(
                ax,
                bounds,
                max_states=compared,
                cmap_name=cmap_name,
                delta=DELTA,
            )
            if sm is not None:
                cbar = fig.colorbar(sm, ax=ax, pad=0.02)
                cbar.set_label("time step")
            draw_initial_cell(ax, bounds[0])

        plot_traj = traj[:compared]
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
            traj[0, xdim],
            traj[0, ydim],
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
            f"inside: {row['inside_states']}/{row['compared_states']} ({pct(row['inside_ratio'])})\n"
            f"requested: {row['desired_states']}\n"
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
        f"delta={DELTA:.10g} | plot dims={PLOT_DIMS} | check dims={CHECK_DIMS} | {horizon}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return [int(index) for index, _ in panels]
