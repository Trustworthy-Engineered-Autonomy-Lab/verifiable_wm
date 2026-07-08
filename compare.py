#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified trajectory-vs-reachable-tube comparison.

Only outputs two figures:
1) real_vs_reachable_tube_examples.png
2) dwm_vs_reachable_tube_examples.png

Designed for:
- mountain_car : state [pos, vel], default plot dims 0 1
- pendulum     : state [theta, omega], default plot dims 0 1
- cartpole     : state [x, x_dot, theta, theta_dot], default plot/check dims 0 1

Meaning of fully contained:
A trajectory is fully contained if every checked state is inside the corresponding
reachable tube bound for the chosen verification horizon and chosen check dimensions.

New option:
--check-dims controls which state dimensions are used for containment checking.
For CartPole, the default is now position/velocity only: --plot-dims 0 1 --check-dims 0 1
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


EPS = 1e-10


ENV_DEFAULTS = {
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
    "cartpole": {
        # Final comparison choice: cart position vs cart velocity.
        "plot_dims": (0, 1),
        "check_dims": (0, 1),
        "description": "cart position vs cart velocity",
        # If states.npz is 2-D while trajectory is 4-D, this is usually [x, theta].
        "decoder_state_indices": (0, 2),
    },
}


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

    @property
    def total_cells(self) -> int:
        return int(np.prod(self.nums))

    def linear_index(self, indices: Sequence[int]) -> int:
        idx = 0
        for i, n in zip(indices, self.nums):
            idx = idx * int(n) + int(i)
        return int(idx)

    def point_to_indices(self, point: np.ndarray) -> Optional[List[int]]:
        point = np.asarray(point, dtype=float)
        if point.shape[0] < self.ndim:
            return None

        inds: List[int] = []
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
            inds.append(i)

        return inds

    def point_to_linear_index(self, point: np.ndarray) -> Optional[int]:
        inds = self.point_to_indices(point)
        if inds is None:
            return None
        return self.linear_index(inds)


def pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def kind_display_name(kind: str) -> str:
    if kind == "real":
        return "Real trajectory"
    if kind == "dwm":
        return "DWM trajectory"
    return kind


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_metadata(path: Optional[Path]) -> Optional[Path]:
    """
    Correct metadata search.

    Important:
    If path is a .npz file, do NOT try to parse that .npz as JSON.
    Instead, look for metadata.json in the same directory.
    """
    if path is None:
        return None

    path = Path(path)
    candidates: List[Path] = []

    if path.suffix.lower() == ".json" and path.name == "metadata.json":
        candidates.append(path)

    candidates.append(path.parent / "metadata.json")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def load_metadata(*paths: Optional[Path]) -> Optional[Dict[str, Any]]:
    for path in paths:
        metadata_path = find_metadata(path)
        if metadata_path is None:
            continue
        try:
            data = load_json(metadata_path)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def npz_keys(path: Path) -> List[str]:
    with np.load(path, allow_pickle=False) as z:
        return list(z.files)


def load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(z.files)}")
        return np.asarray(z[key])


def choose_key(keys: Sequence[str], split: str, kind: str) -> str:
    if kind == "traj":
        candidates = [
            f"{split}_traj",
            f"{split}_real_traj",
            f"{split}_dwm_traj",
            f"{split}_trajectories",
            f"{split}_real_trajectories",
            f"{split}_dwm_trajectories",
            "traj",
            "trajectories",
            split,
        ]
    elif kind == "states":
        candidates = [
            f"{split}_states",
            f"{split}_initial_states",
            f"{split}_init_states",
            "states",
            "initial_states",
            "init_states",
        ]
    else:
        raise ValueError(kind)

    for c in candidates:
        if c in keys:
            return c

    low = {k.lower(): k for k in keys}
    for lk, original in low.items():
        if split.lower() in lk:
            if kind == "traj" and ("traj" in lk or "trajectory" in lk):
                return original
            if kind == "states" and ("state" in lk or "init" in lk):
                return original

    raise KeyError(
        f"Cannot auto-detect {kind} key for split={split}. "
        f"Available keys: {list(keys)}"
    )


def ensure_traj_shape(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"{name} should have shape (N, T+1, dim), got {arr.shape}")
    return arr.astype(float, copy=False)


def ensure_state_shape(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} should have shape (N, dim), got {arr.shape}")
    return arr.astype(float, copy=False)


def load_safety_result(path: Path) -> Tuple[Dict[str, Any], GridInfo, List[Dict[str, Any]]]:
    data = load_json(path)
    if "grid" not in data or "cells" not in data:
        raise ValueError(f"{path} does not look like a safety result: missing grid/cells")

    dims = data["grid"]["dims"]
    names = [d.get("name", f"dim{i}") for i, d in enumerate(dims)]
    starts = np.array([float(d["start"]) for d in dims], dtype=float)
    stops = np.array([float(d["stop"]) for d in dims], dtype=float)
    nums = np.array([int(d["num"]) for d in dims], dtype=int)
    steps = np.array([
        float(d.get("step", (float(d["stop"]) - float(d["start"])) / int(d["num"])))
        for d in dims
    ], dtype=float)

    return data, GridInfo(names, starts, stops, nums, steps), data["cells"]


def dim_intervals(dim_bounds: Sequence[float]) -> List[Tuple[float, float]]:
    """
    Convert one dimension's bound into interval pairs.

    Normal case:
      [low, high] -> [(low, high)]

    Pendulum angle wrapping case can appear as:
      [low1, high1, low2, high2] -> [(low1, high1), (low2, high2)]

    This avoids falsely rejecting states when theta wraps around ±pi.
    """
    arr = np.asarray(dim_bounds, dtype=float).reshape(-1)
    if arr.size < 2:
        return []

    # If a bound row has an odd number of entries, use the first pair.
    # This should not normally happen, but prevents a hard crash.
    if arr.size % 2 == 1:
        arr = arr[:2]

    out: List[Tuple[float, float]] = []
    for i in range(0, arr.size, 2):
        lo = float(arr[i])
        hi = float(arr[i + 1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if lo <= hi:
            out.append((lo, hi))
        else:
            out.append((hi, lo))
    return out


def one_dim_inside(x: float, dim_bounds: Sequence[float], delta: float = 0.0) -> bool:
    intervals = dim_intervals(dim_bounds)
    for lo, hi in intervals:
        if x >= lo - delta - EPS and x <= hi + delta + EPS:
            return True
    return False


def one_dim_violation(x: float, dim_bounds: Sequence[float], delta: float = 0.0) -> float:
    intervals = dim_intervals(dim_bounds)
    if not intervals:
        return float("inf")

    distances = []
    for lo, hi in intervals:
        lo = lo - delta
        hi = hi + delta
        if lo - EPS <= x <= hi + EPS:
            distances.append(0.0)
        elif x < lo:
            distances.append(lo - x)
        else:
            distances.append(x - hi)
    return float(min(distances))


def state_in_bounds(state: np.ndarray, bounds: Sequence[Sequence[float]], delta: float = 0.0) -> bool:
    state = np.asarray(state, dtype=float)
    d = min(len(state), len(bounds))
    for k in range(d):
        if not one_dim_inside(float(state[k]), bounds[k], delta=delta):
            return False
    return True


def violation_amount(state: np.ndarray, bounds: Sequence[Sequence[float]], delta: float = 0.0) -> float:
    state = np.asarray(state, dtype=float)
    d = min(len(state), len(bounds))
    violations = [
        one_dim_violation(float(state[k]), bounds[k], delta=delta)
        for k in range(d)
    ]
    return float(np.max(violations)) if violations else 0.0


def normalize_check_dims(check_dims: Optional[Sequence[int]], grid: GridInfo) -> List[int]:
    """
    If check_dims is None, check all grid dimensions.
    Otherwise, only these selected state dimensions are used in containment checking.
    """
    if check_dims is None:
        return list(range(grid.ndim))

    dims = [int(d) for d in check_dims]
    if len(dims) == 0:
        raise ValueError("--check-dims was provided but empty")

    bad = [d for d in dims if d < 0 or d >= grid.ndim]
    if bad:
        raise ValueError(f"--check-dims contains invalid dimensions {bad}; grid dims are {grid.names}")

    # keep order, remove duplicates
    unique: List[int] = []
    for d in dims:
        if d not in unique:
            unique.append(d)
    return unique


def state_in_bounds_checked(
    state: np.ndarray,
    bounds: Sequence[Sequence[float]],
    check_dims: Sequence[int],
    delta: float = 0.0,
) -> bool:
    """
    Containment check restricted to selected dimensions.

    Example:
      check_dims=[0, 1] means only x and x_dot are checked.
      Other dimensions, such as theta and theta_dot, are ignored for the containment result.
    """
    state = np.asarray(state, dtype=float)

    for k in check_dims:
        if k >= len(state) or k >= len(bounds):
            return False
        if not one_dim_inside(float(state[k]), bounds[k], delta=delta):
            return False

    return True


def violation_amount_checked(
    state: np.ndarray,
    bounds: Sequence[Sequence[float]],
    check_dims: Sequence[int],
    delta: float = 0.0,
) -> float:
    """
    Maximum violation restricted to selected check dimensions.
    """
    state = np.asarray(state, dtype=float)
    violations: List[float] = []

    for k in check_dims:
        if k >= len(state) or k >= len(bounds):
            violations.append(float("inf"))
        else:
            violations.append(one_dim_violation(float(state[k]), bounds[k], delta=delta))

    return float(np.max(violations)) if violations else 0.0


def cell_status(cell: Optional[Dict[str, Any]]) -> str:
    if cell is None:
        return "no_cell"
    if "error_msg" in cell:
        return "error"
    if cell.get("result") is True:
        return "safe"
    if cell.get("result") is False:
        return "unsafe"
    return "unknown"


def find_cell_for_initial_state(
    init_state: np.ndarray,
    grid: GridInfo,
    cells: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    idx = grid.point_to_linear_index(init_state)

    if idx is not None and 0 <= idx < len(cells):
        cell = cells[idx]
        bounds = cell.get("bounds", [])
        if bounds and state_in_bounds(init_state, bounds[0], 0.0):
            return idx, cell

    # Fallback for rare ordering/rounding mismatch.
    for j, cell in enumerate(cells):
        bounds = cell.get("bounds", [])
        if bounds and state_in_bounds(init_state, bounds[0], 0.0):
            return j, cell

    return None, None


def decode_state_indices(
    metadata: Optional[Dict[str, Any]],
    env_name: Optional[str],
    states_dim: Optional[int],
    traj_dim: int,
) -> Optional[np.ndarray]:
    if metadata is not None:
        indices = metadata.get("decoder_state_indices")
        if indices is not None:
            return np.asarray(indices, dtype=int)

    if env_name in ENV_DEFAULTS:
        defaults = ENV_DEFAULTS[env_name]
        if "decoder_state_indices" in defaults and states_dim is not None:
            indices = np.asarray(defaults["decoder_state_indices"], dtype=int)
            if len(indices) == states_dim and traj_dim >= len(indices):
                return indices

    return None


def choose_init_states(
    traj: np.ndarray,
    states: Optional[np.ndarray],
    mode: str,
    metadata: Optional[Dict[str, Any]] = None,
    env_name: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    traj_init = np.asarray(traj[:, 0, :], dtype=float)

    if mode == "traj":
        return traj_init, "traj[:,0,:]"

    if states is None:
        if mode == "states":
            raise ValueError("--init-source states requested, but states.npz was not provided")
        return traj_init, "traj[:,0,:]"

    states_arr = np.asarray(states, dtype=float)

    if states_arr.shape == traj_init.shape:
        return states_arr, "states.npz"

    indices = decode_state_indices(
        metadata=metadata,
        env_name=env_name,
        states_dim=states_arr.shape[1] if states_arr.ndim == 2 else None,
        traj_dim=traj_init.shape[1],
    )

    if (
        indices is not None
        and states_arr.ndim == 2
        and traj_init.ndim == 2
        and states_arr.shape[1] == len(indices)
        and traj_init.shape[1] >= np.max(indices) + 1
    ):
        aligned = traj_init.copy()
        aligned[:, indices] = states_arr
        return aligned, f"traj[:,0,:] with states.npz inserted at dims {indices.tolist()}"

    if mode == "states":
        raise ValueError(
            f"states shape {states_arr.shape} cannot be aligned to trajectory initial shape {traj_init.shape}"
        )

    return traj_init, "traj[:,0,:]"


def filter_indices_in_grid(init_states: np.ndarray, grid: GridInfo) -> np.ndarray:
    keep = []
    for i, s in enumerate(init_states):
        if grid.point_to_linear_index(s) is not None:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def desired_state_count(max_steps: Optional[int], traj_len: int) -> int:
    if max_steps is None:
        return traj_len
    return min(traj_len, max_steps + 1)


def compare_single_trajectory(
    traj: np.ndarray,
    init_state: np.ndarray,
    traj_index: int,
    kind: str,
    split: str,
    grid: GridInfo,
    cells: List[Dict[str, Any]],
    delta: float = 0.0,
    max_steps: Optional[int] = None,
    check_dims: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    active_check_dims = normalize_check_dims(check_dims, grid)
    cell_idx, cell = find_cell_for_initial_state(init_state, grid, cells)
    desired = desired_state_count(max_steps, len(traj))

    row: Dict[str, Any] = {
        "kind": kind,
        "split": split,
        "traj_index": int(traj_index),
        "cell_index": "" if cell_idx is None else int(cell_idx),
        "cell_status": cell_status(cell),
        "traj_len": int(traj.shape[0]),
        "desired_states": int(desired),
        "checked_states": 0,
        "inside_states": 0,
        "inside_ratio": 0.0,
        "fully_inside": False,
        "first_out_step": "",
        "max_violation": np.nan,
        "horizon_mismatch": False,
        "check_dims": active_check_dims,
    }

    if cell is None:
        return row

    bounds = cell.get("bounds", [])
    checked = min(len(bounds), desired)
    row["checked_states"] = int(checked)
    row["horizon_mismatch"] = bool(len(bounds) < desired)

    inside_flags: List[bool] = []
    violations: List[float] = []

    for t in range(checked):
        ok = state_in_bounds_checked(traj[t], bounds[t], active_check_dims, delta)
        inside_flags.append(ok)
        violations.append(violation_amount_checked(traj[t], bounds[t], active_check_dims, delta))

    inside_states = int(np.sum(inside_flags)) if inside_flags else 0
    row["inside_states"] = inside_states
    row["inside_ratio"] = float(inside_states / desired) if desired else 0.0
    row["max_violation"] = float(np.max(violations)) if violations else np.nan

    first_out: Optional[int] = None
    for t, ok in enumerate(inside_flags):
        if not ok:
            first_out = t
            break

    # Strict horizon: if tube is shorter than the requested horizon, it is not fully contained.
    if first_out is None and checked < desired:
        first_out = checked

    row["first_out_step"] = "" if first_out is None else int(first_out)
    row["fully_inside"] = (
        cell_status(cell) != "error"
        and checked == desired
        and first_out is None
        and inside_states == desired
    )
    return row


def compare_set(
    traj: np.ndarray,
    init_states: np.ndarray,
    kind: str,
    split: str,
    grid: GridInfo,
    cells: List[Dict[str, Any]],
    delta: float,
    only_in_grid: bool,
    max_steps: Optional[int],
    check_dims: Optional[Sequence[int]],
) -> List[Dict[str, Any]]:
    idxs = np.arange(len(traj))
    if only_in_grid:
        idxs = filter_indices_in_grid(init_states, grid)

    rows = []
    for i in idxs:
        rows.append(compare_single_trajectory(
            traj=traj[i],
            init_state=init_states[i],
            traj_index=int(i),
            kind=kind,
            split=split,
            grid=grid,
            cells=cells,
            delta=delta,
            max_steps=max_steps,
            check_dims=check_dims,
        ))
    return rows


def build_summary_row(
    rows: List[Dict[str, Any]],
    kind: str,
    split: str,
    delta: float,
    only_in_grid: bool,
    max_steps: Optional[int],
    check_dims: Sequence[int],
) -> Dict[str, Any]:
    total = len(rows)
    fully_contained = int(sum(bool(r["fully_inside"]) for r in rows))
    not_fully_contained = total - fully_contained
    horizon_mismatch = int(sum(bool(r["horizon_mismatch"]) for r in rows))
    inside_ratios = [float(r["inside_ratio"]) for r in rows]

    return {
        "kind": kind,
        "split": split,
        "total": total,
        "fully_contained": fully_contained,
        "not_fully_contained": not_fully_contained,
        "containment_rate": fully_contained / total if total else 0.0,
        "mean_inside_step_ratio": float(np.mean(inside_ratios)) if inside_ratios else 0.0,
        "min_inside_step_ratio": float(np.min(inside_ratios)) if inside_ratios else 0.0,
        "horizon_mismatch_count": horizon_mismatch,
        "delta": float(delta),
        "only_in_grid": bool(only_in_grid),
        "max_steps": max_steps,
        "check_dims": list(check_dims),
    }


def print_summary(row: Dict[str, Any]) -> None:
    total = int(row["total"])
    fully = int(row["fully_contained"])
    not_fully = int(row["not_fully_contained"])

    print()
    print(f"[{kind_display_name(str(row['kind']))}]")
    print(f"  checked trajectories  : {total}")
    if total:
        print(f"  fully contained       : {fully}/{total} ({pct(row['containment_rate'])})")
        print(f"  not fully contained   : {not_fully}/{total}")
    else:
        print("  fully contained       : 0/0")
        print("  not fully contained   : 0/0")
    print(f"  mean step containment : {pct(row['mean_inside_step_ratio'])}")
    print(f"  worst step containment: {pct(row['min_inside_step_ratio'])}")
    print(f"  horizon mismatch cells: {row['horizon_mismatch_count']}")
    print(f"  max_steps             : {row['max_steps']}")
    print(f"  check_dims            : {row['check_dims']}")


def bounds_for_index(
    init_state: np.ndarray,
    grid: GridInfo,
    cells: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[List[Any]]]:
    idx, cell = find_cell_for_initial_state(init_state, grid, cells)
    if cell is None:
        return None, None
    bounds = cell.get("bounds", [])
    if not bounds:
        return idx, None
    return idx, bounds


def draw_reachable_tube(
    ax: Any,
    bounds: Sequence[Any],
    plot_dims: Tuple[int, int],
    cmap_name: str = "Blues",
    linewidth: float = 1.2,
    alpha: float = 0.55,
    label: str = "Reachable tube",
    max_states: Optional[int] = None,
) -> Optional[ScalarMappable]:
    if bounds is None or len(bounds) == 0:
        return None

    xdim, ydim = plot_dims
    n_total = len(bounds)
    n = n_total if max_states is None else min(n_total, max_states)

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=max(n - 1, 1))
    first = True

    for t in range(n):
        b = bounds[t]
        if len(b) <= max(xdim, ydim):
            continue

        x_intervals = dim_intervals(b[xdim])
        y_intervals = dim_intervals(b[ydim])

        for x0, x1 in x_intervals:
            for y0, y1 in y_intervals:
                rect = Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=cmap(norm(t)),
                    linewidth=linewidth,
                    alpha=alpha,
                    label=label if first else None,
                )
                ax.add_patch(rect)
                first = False

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


def find_dim_by_name(grid: GridInfo, keywords: Sequence[str], fallback: Optional[int] = None) -> Optional[int]:
    for i, name in enumerate(grid.names):
        lname = name.lower()
        if any(k.lower() in lname for k in keywords):
            return i
    return fallback


def add_goal_lines(ax: Any, safety: Dict[str, Any], grid: GridInfo, plot_dims: Tuple[int, int]) -> None:
    kwargs = safety.get("verifier", {}).get("kwargs", {})
    xdim, ydim = plot_dims

    if "goal_position_threshold" in kwargs:
        goal = float(kwargs["goal_position_threshold"])
        pos_dim = find_dim_by_name(grid, ["pos", "position"], fallback=0)
        if pos_dim == xdim:
            ax.axvline(goal, color="gray", linestyle="--", linewidth=1.2, label=f"goal pos={goal:g}")
        elif pos_dim == ydim:
            ax.axhline(goal, color="gray", linestyle="--", linewidth=1.2, label=f"goal pos={goal:g}")

    if "goal_angle_threshold" in kwargs:
        goal = float(kwargs["goal_angle_threshold"])
        angle_dim = find_dim_by_name(grid, ["angle", "theta"], fallback=2 if grid.ndim >= 4 else 0)
        if angle_dim == xdim:
            ax.axvline(-goal, color="gray", linestyle="--", linewidth=1.0, label=f"goal angle=±{goal:g}")
            ax.axvline(goal, color="gray", linestyle="--", linewidth=1.0)
        elif angle_dim == ydim:
            ax.axhline(-goal, color="gray", linestyle="--", linewidth=1.0, label=f"goal angle=±{goal:g}")
            ax.axhline(goal, color="gray", linestyle="--", linewidth=1.0)


def pick_example_indices(rows: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
    if not rows:
        return []

    sorted_by_ratio = sorted(rows, key=lambda r: (float(r["inside_ratio"]), int(r["traj_index"])))
    worst = sorted_by_ratio[0]
    best = sorted_by_ratio[-1]

    examples: List[Tuple[int, str]] = [(int(worst["traj_index"]), "Worst containment example")]

    if int(best["traj_index"]) != int(worst["traj_index"]):
        if bool(best["fully_inside"]):
            examples.append((int(best["traj_index"]), "Fully contained example"))
        else:
            examples.append((int(best["traj_index"]), "Best available, still not fully contained"))
    else:
        examples.append((int(best["traj_index"]), "Same trajectory shown again"))

    return examples


def plot_examples(
    out_path: Path,
    kind: str,
    trajs: np.ndarray,
    init_states: np.ndarray,
    rows_by_idx: Dict[int, Dict[str, Any]],
    grid: GridInfo,
    cells: List[Dict[str, Any]],
    safety: Dict[str, Any],
    summary_row: Dict[str, Any],
    plot_dims: Tuple[int, int],
    max_steps: Optional[int],
) -> None:
    examples = pick_example_indices(list(rows_by_idx.values()))
    if not examples:
        print(f"[Warning] No {kind} trajectories to plot.")
        return

    while len(examples) < 2:
        examples.append(examples[-1])

    xdim, ydim = plot_dims
    max_state_index = max(xdim, ydim)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    traj_color = "#d62728" if kind == "real" else "#1f77b4"
    traj_label = "Ground Truth trajectory" if kind == "real" else "DWM trajectory"
    cmap_name = "Oranges" if kind == "real" else "Blues"
    sms = []

    for ax, (idx, panel_title) in zip(axes, examples[:2]):
        traj = trajs[idx]
        if traj.shape[1] <= max_state_index:
            raise ValueError(f"Trajectory dim {traj.shape[1]} is too small for --plot-dims {plot_dims}")

        init_state = init_states[idx]
        row = rows_by_idx[idx]
        _, bounds = bounds_for_index(init_state, grid, cells)

        desired = int(row["desired_states"])
        max_states_to_plot = min(desired, len(bounds)) if bounds is not None else desired

        if bounds is not None:
            sm = draw_reachable_tube(
                ax=ax,
                bounds=bounds,
                plot_dims=plot_dims,
                cmap_name=cmap_name,
                label="Reachable tube",
                max_states=max_states_to_plot,
            )
            if sm is not None:
                sms.append((sm, ax))

        plot_traj = traj[:desired]
        ax.plot(
            plot_traj[:, xdim],
            plot_traj[:, ydim],
            color=traj_color,
            marker="s",
            markersize=3.0,
            linewidth=1.4,
            label=traj_label,
        )

        add_goal_lines(ax, safety, grid, plot_dims)

        first_out = row["first_out_step"]
        first_out_text = "None" if str(first_out) == "" else str(first_out)
        max_v = row["max_violation"]
        status_text = "FULLY CONTAINED" if bool(row["fully_inside"]) else "NOT fully contained"

        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel(grid.names[xdim] if xdim < len(grid.names) else f"state_{xdim}")
        ax.set_ylabel(grid.names[ydim] if ydim < len(grid.names) else f"state_{ydim}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        box_text = (
            f"{status_text}\n"
            f"traj idx: {idx}\n"
            f"inside states: {row['inside_states']}/{row['desired_states']} ({pct(row['inside_ratio'])})\n"
            f"checked states: {row['checked_states']}\n"
            f"check dims: {row['check_dims']}\n"
            f"first out step: {first_out_text}\n"
            f"max violation: {float(max_v):.4g}\n"
            f"cell: {row['cell_index']} ({row['cell_status']})"
        )
        ax.text(
            0.02,
            0.02,
            box_text,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="0.75"),
        )

    for sm, ax in sms:
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Time step")

    total = int(summary_row["total"])
    fully = int(summary_row["fully_contained"])
    horizon_note = f"max_steps={max_steps}" if max_steps is not None else "full trajectory"
    title = (
        f"{kind_display_name(kind)} vs reachable tube | "
        f"fully contained: {fully}/{total} ({pct(summary_row['containment_rate'])}) | "
        f"mean step containment: {pct(summary_row['mean_inside_step_ratio'])} | "
        f"check dims={summary_row['check_dims']} | "
        f"{horizon_note}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate two trajectory-vs-reachable-tube figures for multiple environments.")

    p.add_argument("--env", choices=["cartpole", "mountain_car", "pendulum"], default=None)
    p.add_argument("--safety", type=Path, default=None)
    p.add_argument("--real", type=Path, default=None)
    p.add_argument("--dwm", type=Path, default=None)
    p.add_argument("--states", type=Path, default=None)

    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--real-key", default=None)
    p.add_argument("--dwm-key", default=None)
    p.add_argument("--state-key", default=None)

    p.add_argument("--outdir", type=Path, default=None)

    p.add_argument("--real-delta", type=float, default=0.0)
    p.add_argument("--dwm-delta", type=float, default=0.0)
    p.add_argument("--delta", type=float, default=None)

    p.add_argument("--init-source", choices=["auto", "states", "traj"], default="auto")
    p.add_argument("--only-in-grid", action="store_true")
    p.add_argument("--plot-dims", type=int, nargs=2, default=None,
                   help="Two state dimensions to plot. Defaults by env: mountain_car/pendulum=0 1, cartpole=2 3.")
    p.add_argument("--check-dims", type=int, nargs="+", default=None,
                   help="State dimensions used for containment checking. Default by env; for cartpole default is 0 1. Example: --check-dims 0 1")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Maximum transition steps to compare. Example: --max-steps 20 checks states t=0..20.")
    p.add_argument("--print-keys", action="store_true")

    return p.parse_args()


def resolve_paths_and_defaults(args: argparse.Namespace) -> None:
    root = Path.cwd()

    if args.env is not None:
        if args.safety is None:
            args.safety = root / "results" / args.env / "safety_result.json"
        if args.real is None:
            args.real = root / "trajectories" / args.env / "real_trajectories.npz"
        if args.dwm is None:
            args.dwm = root / "trajectories" / args.env / "dwm_trajectories.npz"
        if args.states is None:
            default_states = root / "trajectories" / args.env / "states.npz"
            if default_states.exists():
                args.states = default_states
        if args.outdir is None:
            args.outdir = root / "results" / args.env / "compare_tube"

    if args.outdir is None:
        args.outdir = Path("compare_outputs")

    if args.safety is None or args.real is None or args.dwm is None:
        raise SystemExit("Please provide --safety --real --dwm, or use --env <cartpole|mountain_car|pendulum>.")


def default_plot_dims(env_name: Optional[str], grid: GridInfo) -> Tuple[int, int]:
    if env_name in ENV_DEFAULTS:
        dims = ENV_DEFAULTS[env_name]["plot_dims"]
        return int(dims[0]), int(dims[1])

    if grid.ndim >= 2:
        return 0, 1

    raise ValueError(f"Cannot plot a grid with ndim={grid.ndim}; need at least 2 dimensions.")


def default_check_dims(env_name: Optional[str], grid: GridInfo) -> Optional[Tuple[int, ...]]:
    """
    Default containment-check dimensions.

    If env has a check_dims default, use it.
    Otherwise return None, meaning all grid dimensions.
    """
    if env_name in ENV_DEFAULTS and "check_dims" in ENV_DEFAULTS[env_name]:
        dims = ENV_DEFAULTS[env_name]["check_dims"]
        return tuple(int(d) for d in dims)
    return None


def main() -> None:
    args = parse_args()

    if args.delta is not None:
        args.real_delta = args.delta
        args.dwm_delta = args.delta

    resolve_paths_and_defaults(args)

    if args.print_keys:
        print("real keys :", npz_keys(args.real))
        print("dwm keys  :", npz_keys(args.dwm))
        if args.states is not None:
            print("state keys:", npz_keys(args.states))
        return

    args.outdir.mkdir(parents=True, exist_ok=True)

    safety, grid, cells = load_safety_result(args.safety)

    plot_dims = tuple(args.plot_dims) if args.plot_dims is not None else default_plot_dims(args.env, grid)
    plot_dims = (int(plot_dims[0]), int(plot_dims[1]))

    if max(plot_dims) >= grid.ndim:
        raise ValueError(f"--plot-dims {plot_dims} is invalid for grid dims {grid.names}")

    requested_check_dims = args.check_dims
    if requested_check_dims is None:
        requested_check_dims = default_check_dims(args.env, grid)
    active_check_dims = normalize_check_dims(requested_check_dims, grid)

    metadata = load_metadata(args.real, args.dwm, args.states)

    real_key = args.real_key or choose_key(npz_keys(args.real), args.split, "traj")
    dwm_key = args.dwm_key or choose_key(npz_keys(args.dwm), args.split, "traj")

    real_traj = ensure_traj_shape(load_npz_array(args.real, real_key), f"real[{real_key}]")
    dwm_traj = ensure_traj_shape(load_npz_array(args.dwm, dwm_key), f"dwm[{dwm_key}]")

    states_arr = None
    if args.states is not None:
        state_key = args.state_key or choose_key(npz_keys(args.states), args.split, "states")
        states_arr = ensure_state_shape(load_npz_array(args.states, state_key), f"states[{state_key}]")

    if real_traj.shape[2] < grid.ndim:
        raise ValueError(f"real trajectory dim {real_traj.shape[2]} < grid dim {grid.ndim}")
    if dwm_traj.shape[2] < grid.ndim:
        raise ValueError(f"dwm trajectory dim {dwm_traj.shape[2]} < grid dim {grid.ndim}")

    real_init, real_init_source = choose_init_states(
        real_traj, states_arr, args.init_source, metadata=metadata, env_name=args.env
    )
    dwm_init, dwm_init_source = choose_init_states(
        dwm_traj, states_arr, args.init_source, metadata=metadata, env_name=args.env
    )

    print("========== Loaded ==========")
    print(f"env        : {args.env}")
    print(f"safety     : {args.safety}")
    print(f"real       : {args.real} | key={real_key} | shape={real_traj.shape}")
    print(f"dwm        : {args.dwm} | key={dwm_key} | shape={dwm_traj.shape}")
    print(f"states     : {args.states}")
    print(f"grid names : {grid.names}")
    print(f"grid nums  : {grid.nums.tolist()} | cells={len(cells)}")
    print(f"plot dims  : {plot_dims} -> {grid.names[plot_dims[0]]}, {grid.names[plot_dims[1]]}")
    print(f"check dims : {active_check_dims} -> {[grid.names[d] for d in active_check_dims]}")
    print(f"max_steps  : {args.max_steps}")
    print(f"real init  : {real_init_source}")
    print(f"dwm init   : {dwm_init_source}")

    real_rows = compare_set(
        traj=real_traj,
        init_states=real_init,
        kind="real",
        split=args.split,
        grid=grid,
        cells=cells,
        delta=args.real_delta,
        only_in_grid=args.only_in_grid,
        max_steps=args.max_steps,
        check_dims=active_check_dims,
    )

    dwm_rows = compare_set(
        traj=dwm_traj,
        init_states=dwm_init,
        kind="dwm",
        split=args.split,
        grid=grid,
        cells=cells,
        delta=args.dwm_delta,
        only_in_grid=args.only_in_grid,
        max_steps=args.max_steps,
        check_dims=active_check_dims,
    )

    real_summary = build_summary_row(
        real_rows, "real", args.split, args.real_delta, args.only_in_grid, args.max_steps, active_check_dims
    )
    dwm_summary = build_summary_row(
        dwm_rows, "dwm", args.split, args.dwm_delta, args.only_in_grid, args.max_steps, active_check_dims
    )

    print_summary(real_summary)
    print_summary(dwm_summary)

    real_rows_by_idx = {int(r["traj_index"]): r for r in real_rows}
    dwm_rows_by_idx = {int(r["traj_index"]): r for r in dwm_rows}

    plot_examples(
        out_path=args.outdir / "real_vs_reachable_tube_examples.png",
        kind="real",
        trajs=real_traj,
        init_states=real_init,
        rows_by_idx=real_rows_by_idx,
        grid=grid,
        cells=cells,
        safety=safety,
        summary_row=real_summary,
        plot_dims=plot_dims,
        max_steps=args.max_steps,
    )

    plot_examples(
        out_path=args.outdir / "dwm_vs_reachable_tube_examples.png",
        kind="dwm",
        trajs=dwm_traj,
        init_states=dwm_init,
        rows_by_idx=dwm_rows_by_idx,
        grid=grid,
        cells=cells,
        safety=safety,
        summary_row=dwm_summary,
        plot_dims=plot_dims,
        max_steps=args.max_steps,
    )

    print(f"[Saved] {args.outdir / 'real_vs_reachable_tube_examples.png'}")
    print(f"[Saved] {args.outdir / 'dwm_vs_reachable_tube_examples.png'}")


if __name__ == "__main__":
    main()
