#!/usr/bin/env python3
"""Plot the worst Brake DWM trajectory against its sampled reachable tube."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.patches import Patch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import signed_tube_margin as stm


DEFAULT_SAFETY = PROJECT_ROOT / (
    "results/brake_system/sampled_tube/recomputed_20260728/"
    "sampled_reachable_tube_old_seed_728.json"
)
DEFAULT_REAL = (
    PROJECT_ROOT / "safety_results/brake_system/real_trajectories.npz"
)
DEFAULT_MODEL = PROJECT_ROOT / (
    "safety_results/brake_system/dwm_trajectories_saliency.npz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "results/brake_system/signed_tube_margin/recomputed_20260728/"
    "b_dwm_sampled/diagnostics"
)


def _validated_trajectories(
    real_trajectories: np.ndarray,
    model_trajectories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    real_trajectories = np.asarray(real_trajectories)
    model_trajectories = np.asarray(model_trajectories)
    if real_trajectories.ndim != 3:
        raise ValueError(
            "real trajectories must have shape (N, T, state_dim)"
        )
    if model_trajectories.shape != real_trajectories.shape:
        raise ValueError("real and DWM trajectories must have equal shapes")
    return real_trajectories, model_trajectories


def _build_case(
    real_trajectories: np.ndarray,
    model_trajectories: np.ndarray,
    cells: Sequence[dict[str, Any]],
    row: dict[str, Any],
    dims: Sequence[int],
) -> dict[str, Any]:
    trajectory_index = int(row["traj_index"])
    cell_index = int(row["cell_index"])
    bounds = cells[cell_index]["bounds"]
    real = np.asarray(real_trajectories[trajectory_index], dtype=float)
    model = np.asarray(model_trajectories[trajectory_index], dtype=float)
    if len(bounds) != len(real):
        raise ValueError("selected tube and trajectories have unequal horizons")

    real_margins = np.asarray(
        [
            stm.state_signed_margin(state, step_bounds, dims)
            for state, step_bounds in zip(real, bounds)
        ],
        dtype=float,
    )
    model_margins = np.asarray(
        [
            stm.state_signed_margin(state, step_bounds, dims)
            for state, step_bounds in zip(model, bounds)
        ],
        dtype=float,
    )
    first_out = next(
        (
            step
            for step, margin in enumerate(real_margins)
            if margin < -stm.EPS
        ),
        None,
    )
    state_difference = np.abs(real - model)
    return {
        "trajectory_index": trajectory_index,
        "cell_index": cell_index,
        "first_violating_step": first_out,
        "worst_real_margin": float(real_margins.min()),
        "real_states": real,
        "model_states": model,
        "tube_bounds": bounds,
        "real_margins": real_margins,
        "model_margins": model_margins,
        "state_difference": state_difference,
        "max_abs_state_difference": float(state_difference.max()),
    }


def build_diagnostic(
    real_trajectories: np.ndarray,
    model_trajectories: np.ndarray,
    grid: stm.GridInfo,
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int] = (0, 1),
) -> dict[str, Any]:
    real_trajectories, model_trajectories = _validated_trajectories(
        real_trajectories, model_trajectories
    )
    valid = [
        row
        for row in stm.evaluate_set(
            real_trajectories, grid, cells, dims
        )
        if row["status"] == "valid"
    ]
    if not valid:
        raise ValueError("no real trajectory matches a valid tube cell")
    worst = min(valid, key=lambda row: float(row["signed_margin"]))
    return _build_case(
        real_trajectories, model_trajectories, cells, worst, dims
    )


def build_random_diagnostics(
    real_trajectories: np.ndarray,
    model_trajectories: np.ndarray,
    grid: stm.GridInfo,
    cells: Sequence[dict[str, Any]],
    *,
    count: int = 6,
    seed: int = 728,
    dims: Sequence[int] = (0, 1),
) -> list[dict[str, Any]]:
    real_trajectories, model_trajectories = _validated_trajectories(
        real_trajectories, model_trajectories
    )
    valid = [
        row
        for row in stm.evaluate_set(
            real_trajectories, grid, cells, dims
        )
        if row["status"] == "valid"
    ]
    if int(count) <= 0:
        raise ValueError("random trajectory count must be positive")
    if int(count) > len(valid):
        raise ValueError(
            "requested more trajectories than valid tube matches"
        )
    selected = np.random.default_rng(int(seed)).choice(
        np.arange(len(valid)), int(count), replace=False
    )
    return [
        _build_case(
            real_trajectories,
            model_trajectories,
            cells,
            valid[int(index)],
            dims,
        )
        for index in selected
    ]


def _interval(bounds: Sequence[float]) -> tuple[float, float]:
    pairs = stm.interval_pairs(bounds)
    if len(pairs) != 1:
        raise ValueError("Brake diagnostic expects one interval per dimension")
    return pairs[0]


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def random_figure_layout() -> dict[str, Any]:
    return {
        "title_y": 0.975,
        "legend_y": 0.925,
        "legend_columns": 5,
        "subplot_top": 0.86,
        "subplot_right": 0.88,
        "colorbar_rect": (0.91, 0.15, 0.015, 0.67),
    }


def random_tube_style() -> dict[str, Any]:
    return {
        "edge_color": "darkorange",
        "edge_alpha": 0.95,
        "face_alpha": 0.14,
        "linewidth": 2.2,
        "rectangle_zorder": 3,
        "trajectory_zorder": 4,
        "legend_label": "Tube rectangles",
    }


def write_diagnostic(
    payload: dict[str, Any],
    *,
    safety_path: Path,
    real_path: Path,
    model_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "brake_dwm_sampled_tube_diagnostic.png"
    json_path = output_dir / "brake_dwm_sampled_tube_diagnostic.json"

    real = np.asarray(payload["real_states"], dtype=float)
    model = np.asarray(payload["model_states"], dtype=float)
    bounds = payload["tube_bounds"]
    steps = np.arange(len(real))
    lows = np.asarray(
        [[_interval(step_bounds[dim])[0] for dim in (0, 1)]
         for step_bounds in bounds],
        dtype=float,
    )
    highs = np.asarray(
        [[_interval(step_bounds[dim])[1] for dim in (0, 1)]
         for step_bounds in bounds],
        dtype=float,
    )
    first_out = payload["first_violating_step"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    phase, distance, velocity, margins = axes.flat
    colors = colormaps["Oranges"](
        np.linspace(0.25, 0.95, len(steps))
    )
    for step, color in zip(steps, colors):
        phase.add_patch(
            Rectangle(
                (lows[step, 0], lows[step, 1]),
                highs[step, 0] - lows[step, 0],
                highs[step, 1] - lows[step, 1],
                facecolor=color,
                edgecolor=color,
                alpha=0.16,
                linewidth=1.5,
            )
        )
    phase.plot(
        real[:, 0], real[:, 1], "o-", color="tab:red", label="Real"
    )
    phase.plot(
        model[:, 0],
        model[:, 1],
        "s--",
        color="tab:blue",
        label="DWM",
    )
    phase.scatter(
        real[0, 0], real[0, 1], s=110, color="green", zorder=5,
        label="Initial state",
    )
    if first_out is not None:
        phase.scatter(
            real[first_out, 0],
            real[first_out, 1],
            s=150,
            facecolors="none",
            edgecolors="black",
            linewidths=2,
            zorder=6,
            label="First real violation",
        )
    phase.set(
        title="Phase plane: paired trajectories and per-step raw tube",
        xlabel="distance",
        ylabel="velocity",
    )
    phase.legend()

    labels = (("Distance", "distance"), ("Velocity", "velocity"))
    for dim, (axis, (title, ylabel)) in enumerate(
        zip((distance, velocity), labels)
    ):
        axis.fill_between(
            steps,
            lows[:, dim],
            highs[:, dim],
            color="tab:orange",
            alpha=0.25,
            label="Raw tube",
        )
        axis.plot(steps, real[:, dim], "o-", color="tab:red", label="Real")
        axis.plot(
            steps, model[:, dim], "s--", color="tab:blue", label="DWM"
        )
        if first_out is not None:
            axis.axvline(
                first_out, color="black", linestyle=":", label="First out"
            )
        axis.set(title=f"{title} by step", xlabel="step", ylabel=ylabel)
        axis.set_xticks(steps)
        axis.legend()

    real_margins = np.asarray(payload["real_margins"], dtype=float)
    model_margins = np.asarray(payload["model_margins"], dtype=float)
    per_step_difference = np.asarray(
        payload["state_difference"], dtype=float
    ).max(axis=1)
    margins.axhline(0.0, color="black", linewidth=1)
    margins.plot(
        steps, real_margins, "o-", color="tab:red", label="Real margin"
    )
    margins.plot(
        steps,
        model_margins,
        "s--",
        color="tab:blue",
        label="DWM margin",
    )
    difference_axis = margins.twinx()
    difference_axis.plot(
        steps,
        per_step_difference,
        "d-.",
        color="tab:purple",
        label="max |Real-DWM|",
    )
    if first_out is not None:
        margins.axvline(first_out, color="black", linestyle=":")
    margins.set(
        title="Signed margins and paired state difference",
        xlabel="step",
        ylabel="signed margin",
    )
    margins.set_xticks(steps)
    difference_axis.set_ylabel("max absolute state difference")
    handles, labels_left = margins.get_legend_handles_labels()
    handles_right, labels_right = (
        difference_axis.get_legend_handles_labels()
    )
    margins.legend(handles + handles_right, labels_left + labels_right)

    fig.suptitle(
        "Brake DWM sampled-tube diagnosis | "
        f"trajectory={payload['trajectory_index']} | "
        f"cell={payload['cell_index']} | "
        f"first out={first_out} | "
        f"worst real margin={payload['worst_real_margin']:.8g} | "
        "max |Real-DWM|="
        f"{payload['max_abs_state_difference']:.8g}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    saved = {
        **payload,
        "sources": {
            "safety": str(safety_path),
            "real": str(real_path),
            "model": str(model_path),
        },
    }
    json_path.write_text(
        json.dumps(_json_ready(saved), indent=2), encoding="utf-8"
    )
    return png_path, json_path


def write_worst_complete_trajectory(
    payload: dict[str, Any],
    *,
    safety_path: Path,
    real_path: Path,
    model_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = (
        output_dir
        / "brake_dwm_sampled_tube_worst_complete_trajectory.png"
    )
    json_path = (
        output_dir
        / "brake_dwm_sampled_tube_worst_complete_trajectory.json"
    )

    style = random_tube_style()
    bounds = payload["tube_bounds"]
    norm = Normalize(vmin=0, vmax=max(len(bounds) - 1, 1))
    cmap = colormaps["Oranges"]
    fig, axis = plt.subplots(figsize=(12, 7))
    for step, step_bounds in enumerate(bounds):
        low_dis, high_dis = _interval(step_bounds[0])
        low_vel, high_vel = _interval(step_bounds[1])
        axis.add_patch(
            Rectangle(
                (low_dis, low_vel),
                high_dis - low_dis,
                high_vel - low_vel,
                facecolor=to_rgba(cmap(norm(step)), style["face_alpha"]),
                edgecolor=to_rgba(
                    style["edge_color"], style["edge_alpha"]
                ),
                linewidth=style["linewidth"],
                zorder=style["rectangle_zorder"],
            )
        )

    real = np.asarray(payload["real_states"], dtype=float)
    model = np.asarray(payload["model_states"], dtype=float)
    axis.plot(
        real[:, 0],
        real[:, 1],
        "o-",
        color="tab:red",
        label="Real",
        zorder=style["trajectory_zorder"],
    )
    axis.plot(
        model[:, 0],
        model[:, 1],
        "s--",
        color="tab:blue",
        label="DWM",
        zorder=style["trajectory_zorder"],
    )
    axis.scatter(
        real[0, 0],
        real[0, 1],
        s=100,
        color="green",
        zorder=5,
        label="Initial state",
    )
    first_out = payload["first_violating_step"]
    if first_out is not None:
        axis.scatter(
            real[first_out, 0],
            real[first_out, 1],
            s=160,
            facecolors="none",
            edgecolors="black",
            linewidths=2.2,
            zorder=6,
            label="First real violation",
        )
    handles, labels = axis.get_legend_handles_labels()
    handles.append(
        Patch(
            facecolor=to_rgba("orange", style["face_alpha"]),
            edgecolor=style["edge_color"],
            linewidth=style["linewidth"],
            label=style["legend_label"],
        )
    )
    labels.append(style["legend_label"])
    axis.legend(handles, labels, loc="best")
    axis.set(
        xlabel="distance",
        ylabel="velocity",
        title=(
            "Brake worst complete Real/DWM trajectory with sampled tubes\n"
            f"traj {payload['trajectory_index']} | "
            f"cell {payload['cell_index']} | "
            f"first out {first_out} | "
            f"margin {payload['worst_real_margin']:.9g}"
        ),
    )
    axis.grid(alpha=0.25)
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=0.02
    )
    colorbar.set_label("tube time step")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    saved = {
        **payload,
        "selection": "minimum real signed margin",
        "sources": {
            "safety": str(safety_path),
            "real": str(real_path),
            "model": str(model_path),
        },
    }
    json_path.write_text(
        json.dumps(_json_ready(saved), indent=2), encoding="utf-8"
    )
    return png_path, json_path


def write_random_diagnostics(
    payloads: Sequence[dict[str, Any]],
    *,
    seed: int,
    safety_path: Path,
    real_path: Path,
    model_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    if len(payloads) != 6:
        raise ValueError("random diagnostic requires exactly six trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = (
        output_dir
        / "brake_dwm_sampled_tube_random6_complete_trajectories.png"
    )
    json_path = (
        output_dir
        / "brake_dwm_sampled_tube_random6_complete_trajectories.json"
    )

    max_step = max(len(payload["tube_bounds"]) - 1 for payload in payloads)
    norm = Normalize(vmin=0, vmax=max(max_step, 1))
    cmap = colormaps["Oranges"]
    layout = random_figure_layout()
    tube_style = random_tube_style()
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), sharex=True, sharey=True)
    for axis, payload in zip(axes.flat, payloads):
        bounds = payload["tube_bounds"]
        for step, step_bounds in enumerate(bounds):
            low_dis, high_dis = _interval(step_bounds[0])
            low_vel, high_vel = _interval(step_bounds[1])
            color = cmap(norm(step))
            axis.add_patch(
                Rectangle(
                    (low_dis, low_vel),
                    high_dis - low_dis,
                    high_vel - low_vel,
                    facecolor=to_rgba(color, tube_style["face_alpha"]),
                    edgecolor=to_rgba(
                        tube_style["edge_color"],
                        tube_style["edge_alpha"],
                    ),
                    linewidth=tube_style["linewidth"],
                    zorder=tube_style["rectangle_zorder"],
                )
            )
        real = np.asarray(payload["real_states"], dtype=float)
        model = np.asarray(payload["model_states"], dtype=float)
        axis.plot(
            real[:, 0],
            real[:, 1],
            "o-",
            color="tab:red",
            label="Real",
            zorder=tube_style["trajectory_zorder"],
        )
        axis.plot(
            model[:, 0],
            model[:, 1],
            "s--",
            color="tab:blue",
            label="DWM",
            zorder=tube_style["trajectory_zorder"],
        )
        axis.scatter(
            real[0, 0],
            real[0, 1],
            s=75,
            color="green",
            zorder=5,
            label="Initial state",
        )
        first_out = payload["first_violating_step"]
        if first_out is not None:
            axis.scatter(
                real[first_out, 0],
                real[first_out, 1],
                s=110,
                facecolors="none",
                edgecolors="black",
                linewidths=1.8,
                zorder=6,
                label="First real violation",
            )
        axis.set_title(
            f"traj {payload['trajectory_index']} | "
            f"cell {payload['cell_index']} | "
            f"first out {first_out}\n"
            f"margin {payload['worst_real_margin']:.6g}"
        )
        axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("distance")
    for axis in axes[:, 0]:
        axis.set_ylabel("velocity")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(
        Patch(
            facecolor=to_rgba("orange", tube_style["face_alpha"]),
            edgecolor=tube_style["edge_color"],
            linewidth=tube_style["linewidth"],
            label=tube_style["legend_label"],
        )
    )
    labels.append(tube_style["legend_label"])
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, layout["legend_y"]),
        ncol=layout["legend_columns"],
    )
    colorbar_axis = fig.add_axes(layout["colorbar_rect"])
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axis,
        label="tube time step",
    )
    fig.suptitle(
        f"Brake complete Real/DWM trajectories with sampled tubes "
        f"(random seed {seed})",
        fontsize=15,
        y=layout["title_y"],
    )
    fig.subplots_adjust(
        top=layout["subplot_top"],
        right=layout["subplot_right"],
        hspace=0.28,
        wspace=0.14,
    )
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    saved = {
        "seed": int(seed),
        "selected_trajectory_indices": [
            int(payload["trajectory_index"]) for payload in payloads
        ],
        "sources": {
            "safety": str(safety_path),
            "real": str(real_path),
            "model": str(model_path),
        },
        "trajectories": payloads,
    }
    json_path.write_text(
        json.dumps(_json_ready(saved), indent=2), encoding="utf-8"
    )
    return png_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safety", type=Path, default=DEFAULT_SAFETY)
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    safety = json.loads(args.safety.read_text(encoding="utf-8"))
    grid, cells = stm.load_safety_result(args.safety)
    real = stm.load_trajectory(args.real, "test_traj")
    model = stm.load_trajectory(args.model, "test_traj")
    payload = build_diagnostic(real, model, grid, cells)
    payload["seed"] = int(safety["seed"])
    png_path, json_path = write_diagnostic(
        payload,
        safety_path=args.safety,
        real_path=args.real,
        model_path=args.model,
        output_dir=args.output_dir,
    )
    worst_png_path, worst_json_path = write_worst_complete_trajectory(
        payload,
        safety_path=args.safety,
        real_path=args.real,
        model_path=args.model,
        output_dir=args.output_dir,
    )
    random_payloads = build_random_diagnostics(
        real, model, grid, cells, count=6, seed=728
    )
    random_png_path, random_json_path = write_random_diagnostics(
        random_payloads,
        seed=728,
        safety_path=args.safety,
        real_path=args.real,
        model_path=args.model,
        output_dir=args.output_dir,
    )
    print(f"diagnostic plot: {png_path}")
    print(f"diagnostic values: {json_path}")
    print(f"worst complete plot: {worst_png_path}")
    print(f"worst complete values: {worst_json_path}")
    print(f"random-six plot: {random_png_path}")
    print(f"random-six values: {random_json_path}")


if __name__ == "__main__":
    main()
