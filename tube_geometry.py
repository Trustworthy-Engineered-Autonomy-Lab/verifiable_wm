"""
Pure grid/interval math for reachable-tube containment checks.

No matplotlib, no argparse — this module only answers "is this state inside
this cell's bounds", so it can be imported and unit-tested on its own (e.g.
the pendulum angle wrap-around handling in dim_intervals is exactly the kind
of edge case worth a dedicated test).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
