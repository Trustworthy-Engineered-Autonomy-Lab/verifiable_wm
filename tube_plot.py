"""
Matplotlib rendering for the trajectory-vs-reachable-tube figures.

Consumes GridInfo/cells (tube_geometry) and per-trajectory rows
(tube_report); doesn't do any containment math of its own.
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

from tube_geometry import GridInfo, dim_intervals, find_cell_for_initial_state
from tube_report import kind_display_name, pct


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
