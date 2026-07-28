#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a predictor tube from three trajectories in every cell."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_ENVIRONMENT,
    DEFAULT_GRID_RESULT_PATH,
    DEFAULT_REAL_PATH,
    DEFAULT_SAMPLES_PER_CELL,
    DEFAULT_TRAJECTORY_OUTPUT_PATH,
    DEFAULT_TUBE_OUTPUT_PATH,
    absolute_path,
    ensure_parent,
    resolve_device,
    set_seed,
    validate_environment,
)
from predictor_model import load_predictor_checkpoint
from tube_utils import (
    build_raw_tubes,
    load_grid,
    save_tube_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample exactly three points per cell, predict the checkpoint "
            "horizon, save all cells' trajectories in one NPZ file, and build "
            "their min/max envelope."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        "--env-name",
        dest="env",
        default=DEFAULT_ENVIRONMENT,
        help=(
            "Environment label written to the NPZ/JSON, for example "
            "brake_system, cartpole, mountain_car, or pendulum."
        ),
    )
    parser.add_argument(
        "--grid-result",
        type=Path,
        default=DEFAULT_GRID_RESULT_PATH,
        help="Input safety_result.json path.",
    )
    parser.add_argument(
        "--real",
        type=Path,
        default=DEFAULT_REAL_PATH,
        help=(
            "Reference real_trajectories.npz. Its split names, action arrays, "
            "and metadata are mirrored in predictor_trajectories.npz."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Input predictor checkpoint path.",
    )
    parser.add_argument(
        "--tube-output",
        type=Path,
        default=DEFAULT_TUBE_OUTPUT_PATH,
        help="Output predictor_tube.json path.",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=DEFAULT_TRAJECTORY_OUTPUT_PATH,
        help=(
            "Output predictor_trajectories.npz path."
        ),
    )
    parser.add_argument(
        "--samples-per-cell",
        type=int,
        default=DEFAULT_SAMPLES_PER_CELL,
        help="Total sampled initial states in each cell (not per dimension).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help=(
            "Number of transition steps. If omitted, use the checkpoint horizon."
        ),
    )
    parser.add_argument("--cell-batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_cell < 2:
        raise ValueError("--samples-per-cell must be at least 2")
    if args.horizon is not None and args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.cell_batch_size <= 0:
        raise ValueError("--cell-batch-size must be positive")

    args.env = validate_environment(args.env)
    args.grid_result = absolute_path(args.grid_result)
    args.real = absolute_path(args.real)
    args.checkpoint = absolute_path(args.checkpoint)
    args.tube_output = absolute_path(args.tube_output)
    args.trajectory_output = absolute_path(args.trajectory_output)
    if args.trajectory_output.suffix.lower() != ".npz":
        raise ValueError("--trajectory-output must end with .npz")

    for label, path in (
        ("grid result", args.grid_result),
        ("real trajectories", args.real),
        ("checkpoint", args.checkpoint),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    ensure_parent(args.tube_output)
    ensure_parent(args.trajectory_output)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    print("========== Tube paths ==========")
    print(f"grid result       : {args.grid_result}")
    print(f"real trajectories : {args.real}")
    print(f"checkpoint        : {args.checkpoint}")
    print(f"tube output       : {args.tube_output}")
    print(f"trajectory output : {args.trajectory_output}")
    print(f"environment       : {args.env}")
    print(f"device            : {device}")

    model, mean, std, checkpoint = load_predictor_checkpoint(
        args.checkpoint, device
    )
    checkpoint_environment = checkpoint.get("environment")
    if (
        checkpoint_environment is not None
        and str(checkpoint_environment) != args.env
    ):
        raise ValueError(
            f"checkpoint environment={checkpoint_environment!r} does not match "
            f"--env={args.env!r}"
        )
    horizon = model.horizon if args.horizon is None else int(args.horizon)
    if model.horizon < horizon:
        raise ValueError(
            f"checkpoint only predicts {model.horizon} steps, "
            f"but --horizon={horizon} was requested"
        )

    source_grid, grid, cell_bounds = load_grid(args.grid_result)
    if grid.ndim != model.state_dim:
        raise ValueError(
            f"grid_dim={grid.ndim} does not match model state_dim={model.state_dim}"
        )

    raw_tubes = build_raw_tubes(
        model=model,
        mean=mean,
        std=std,
        cell_bounds=cell_bounds,
        samples_per_cell=args.samples_per_cell,
        horizon=horizon,
        batch_size=args.cell_batch_size,
        device=device,
        trajectory_output_path=args.trajectory_output,
        reference_trajectory_path=args.real,
        environment=args.env,
    )

    save_tube_json(
        path=args.tube_output,
        source_grid=source_grid,
        grid=grid,
        cell_bounds=cell_bounds,
        raw_tubes=raw_tubes,
        trajectory_path=args.trajectory_output,
        checkpoint=checkpoint,
        grid_path=args.grid_result,
        checkpoint_path=args.checkpoint,
        samples_per_cell=args.samples_per_cell,
        environment=args.env,
    )
    print(f"[Saved] all cell trajectories: {args.trajectory_output}")
    print(f"[Saved] predictor tube: {args.tube_output}")


if __name__ == "__main__":
    main()
