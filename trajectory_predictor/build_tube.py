#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a 20-step predictor tube from three trajectories in every cell."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_GRID_RESULT_PATH,
    DEFAULT_HORIZON,
    DEFAULT_SAMPLES_PER_CELL,
    DEFAULT_TUBE_OUTPUT_PATH,
    absolute_path,
    ensure_parent,
    resolve_device,
    set_seed,
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
            "Sample exactly three points per cell, predict 20 steps, save all "
            "cells' trajectories in one NPZ file, and build their min/max envelope."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--grid-result", type=Path, default=DEFAULT_GRID_RESULT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tube-output", type=Path, default=DEFAULT_TUBE_OUTPUT_PATH)
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=None,
        help=(
            "Single NPZ file containing every cell's trajectories. "
            "Defaults to predictor_trajectories.npz beside --tube-output."
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
        default=DEFAULT_HORIZON,
        help="Number of transition steps; output contains horizon+1 states.",
    )
    parser.add_argument("--cell-batch-size", type=int, default=1024)
    parser.add_argument("--env-name", default="pendulum")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_cell < 2:
        raise ValueError("--samples-per-cell must be at least 2")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.cell_batch_size <= 0:
        raise ValueError("--cell-batch-size must be positive")

    args.grid_result = absolute_path(args.grid_result)
    args.checkpoint = absolute_path(args.checkpoint)
    args.tube_output = absolute_path(args.tube_output)
    if args.trajectory_output is None:
        args.trajectory_output = (
            args.tube_output.parent / "predictor_trajectories.npz"
        )
    else:
        args.trajectory_output = absolute_path(args.trajectory_output)
    if args.trajectory_output.suffix.lower() != ".npz":
        raise ValueError("--trajectory-output must end with .npz")

    for label, path in (
        ("grid result", args.grid_result),
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
    print(f"checkpoint        : {args.checkpoint}")
    print(f"tube output       : {args.tube_output}")
    print(f"trajectory output : {args.trajectory_output}")
    print(f"device            : {device}")

    model, mean, std, checkpoint = load_predictor_checkpoint(
        args.checkpoint, device
    )
    if model.horizon < args.horizon:
        raise ValueError(
            f"checkpoint only predicts {model.horizon} steps, "
            f"but --horizon={args.horizon} was requested"
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
        horizon=args.horizon,
        batch_size=args.cell_batch_size,
        device=device,
        trajectory_output_path=args.trajectory_output,
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
        environment=args.env_name,
    )
    print(f"[Saved] all cell trajectories: {args.trajectory_output}")
    print(f"[Saved] predictor tube: {args.tube_output}")


if __name__ == "__main__":
    main()
