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
from matplotlib.patches import Rectangle

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


def build_diagnostic(
    real_trajectories: np.ndarray,
    model_trajectories: np.ndarray,
    grid: stm.GridInfo,
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int] = (0, 1),
) -> dict[str, Any]:
    real_trajectories = np.asarray(real_trajectories)
    model_trajectories = np.asarray(model_trajectories)
    if real_trajectories.ndim != 3:
        raise ValueError(
            "real trajectories must have shape (N, T, state_dim)"
        )
    if model_trajectories.shape != real_trajectories.shape:
        raise ValueError("real and DWM trajectories must have equal shapes")

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
    trajectory_index = int(worst["traj_index"])
    cell_index = int(worst["cell_index"])
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
    print(f"diagnostic plot: {png_path}")
    print(f"diagnostic values: {json_path}")


if __name__ == "__main__":
    main()
