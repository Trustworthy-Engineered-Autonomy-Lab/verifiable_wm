"""
Per-trajectory containment comparison and summary statistics.

Builds on tube_geometry's containment math; knows nothing about matplotlib
or argparse. Output is plain dict rows that tube_plot.py (or a future
CSV/JSON exporter) can consume.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tube_geometry import (
    GridInfo,
    cell_status,
    find_cell_for_initial_state,
    normalize_check_dims,
    state_in_bounds_checked,
    violation_amount_checked,
)

# Per-env plotting/checking defaults and the CartPole decoder_state_indices
# quirk (its Decoder only ever sees [position, angle], never the full 4-D
# state). mountain_car / pendulum decoders take the full state, so they
# don't need an entry here.
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


def pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def kind_display_name(kind: str) -> str:
    if kind == "real":
        return "Real trajectory"
    if kind == "dwm":
        return "DWM trajectory"
    return kind


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
