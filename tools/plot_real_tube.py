"""Render the worst and best real-trajectory containment panels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


def _intervals(values: Sequence[float]) -> list[tuple[float, float]]:
    array = np.asarray(values, dtype=float).reshape(-1)
    return [
        (min(float(array[index]), float(array[index + 1])),
         max(float(array[index]), float(array[index + 1])))
        for index in range(0, len(array), 2)
    ]


def _select_panels(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    worst = min(
        rows,
        key=lambda row: (
            float(row["inside_ratio"]),
            float(row["signed_margin"]),
            int(row["traj_index"]),
        ),
    )
    remaining = [row for row in rows if row["traj_index"] != worst["traj_index"]]
    best = max(
        remaining or rows,
        key=lambda row: (
            float(row["inside_ratio"]),
            float(row["signed_margin"]),
            -int(row["traj_index"]),
        ),
    )
    return [worst, best]


def _draw_tube(
    ax: Any,
    bounds_history: Sequence[Any],
    dims: Sequence[int],
) -> ScalarMappable:
    xdim, ydim = dims
    cmap = plt.get_cmap("Oranges")
    norm = Normalize(vmin=0, vmax=max(len(bounds_history) - 1, 1))
    first = True
    for time_index, bounds in enumerate(bounds_history):
        for x0, x1 in _intervals(bounds[xdim]):
            for y0, y1 in _intervals(bounds[ydim]):
                ax.add_patch(Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=cmap(norm(time_index)),
                    linewidth=1.2,
                    alpha=0.55,
                    label="reachable tube" if first else None,
                ))
                first = False
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    return scalar


def _draw_initial_cell(
    ax: Any,
    initial_bounds: Sequence[Any],
    dims: Sequence[int],
) -> None:
    xdim, ydim = dims
    first = True
    for x0, x1 in _intervals(initial_bounds[xdim]):
        for y0, y1 in _intervals(initial_bounds[ydim]):
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


def _add_goal_lines(ax: Any, payload: dict[str, Any], dims: Sequence[int]) -> None:
    kwargs = payload.get("verifier", {}).get("kwargs", {})
    xdim, ydim = dims
    if "goal_position_threshold" in kwargs:
        goal = float(kwargs["goal_position_threshold"])
        if xdim == 0:
            ax.axvline(goal, linestyle="--", linewidth=1.1, label=f"goal={goal:g}")
        elif ydim == 0:
            ax.axhline(goal, linestyle="--", linewidth=1.1, label=f"goal={goal:g}")
    if "goal_angle_threshold" in kwargs:
        goal = float(kwargs["goal_angle_threshold"])
        angle_dim = 2 if len(payload.get("grid", {}).get("dims", [])) >= 4 else 0
        if xdim == angle_dim:
            ax.axvline(-goal, linestyle="--", linewidth=1.0, label=f"goal=±{goal:g}")
            ax.axvline(goal, linestyle="--", linewidth=1.0)
        elif ydim == angle_dim:
            ax.axhline(-goal, linestyle="--", linewidth=1.0, label=f"goal=±{goal:g}")
            ax.axhline(goal, linestyle="--", linewidth=1.0)


def plot_real_tube(
    output_path: Path,
    *,
    trajectories: np.ndarray,
    rows: Sequence[dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    grid_names: Sequence[str],
    dims: Sequence[int],
    tube_payload: dict[str, Any],
    coverage: float,
) -> None:
    """Write one compare-style figure containing only real trajectories."""
    panels = _select_panels(rows)
    labels = ("Worst containment", "Best containment")
    xdim, ydim = dims
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    for axis, row, panel_label in zip(axes, panels, labels):
        trajectory = trajectories[int(row["traj_index"])]
        bounds = cells[int(row["cell_index"])]["bounds"]
        scalar = _draw_tube(axis, bounds, dims)
        colorbar = figure.colorbar(scalar, ax=axis, pad=0.02)
        colorbar.set_label("time step")
        _draw_initial_cell(axis, bounds[0], dims)
        axis.plot(
            trajectory[:, xdim],
            trajectory[:, ydim],
            marker="s",
            markersize=3.0,
            linewidth=1.4,
            color="tab:red",
            label="Real trajectory",
        )
        axis.plot(
            trajectory[0, xdim],
            trajectory[0, ydim],
            marker="o",
            markersize=6.0,
            color="green",
            linestyle="None",
            label="initial state",
        )
        _add_goal_lines(axis, tube_payload, dims)
        first_out = "None" if row["first_out_step"] is None else row["first_out_step"]
        status = "FULLY CONTAINED" if row["fully_inside"] else "NOT fully contained"
        axis.text(
            0.02,
            0.02,
            (
                f"{status}\n"
                f"inside: {row['inside_states']}/{row['compared_states']} "
                f"({100.0 * row['inside_ratio']:.2f}%)\n"
                f"first out: {first_out}\n"
                f"max violation: {row['max_violation']:.4g}\n"
                f"cell: {row['cell_index']}"
            ),
            transform=axis.transAxes,
            fontsize=8,
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="0.75"),
        )
        axis.set_title(panel_label, fontsize=10)
        axis.set_xlabel(grid_names[xdim])
        axis.set_ylabel(grid_names[ydim])
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle(
        f"Real trajectories vs reachable tube | Coverage: {100.0 * coverage:.2f}% | "
        f"plot/check dims={tuple(dims)}",
        fontsize=12,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=230)
    plt.close(figure)
