#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the sampled raw tube, conformalize it, and evaluate on test_traj."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_GRID_RESULT_PATH,
    DEFAULT_REAL_PATH,
    DEFAULT_TUBE_OUTPUT_PATH,
    absolute_path,
    ensure_parent,
    resolve_device,
    set_seed,
)
from data_utils import load_real_trajectories
from predictor_model import load_predictor_checkpoint
from tube_utils import (
    build_raw_tubes,
    calibration_scores,
    conformal_delta,
    evaluate_tubes,
    inflate_tubes,
    load_grid,
    save_tube_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a conformal predictor tube.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL_PATH)
    parser.add_argument("--grid-result", type=Path, default=DEFAULT_GRID_RESULT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tube-output", type=Path, default=DEFAULT_TUBE_OUTPUT_PATH)
    parser.add_argument("--samples-per-dim", type=int, default=11)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cell-batch-size", type=int, default=1024)
    parser.add_argument("--env-name", default="mountain_car")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_dim < 2:
        raise ValueError("--samples-per-dim must be at least 2")
    if args.cell_batch_size <= 0:
        raise ValueError("--cell-batch-size must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    args.real = absolute_path(args.real)
    args.grid_result = absolute_path(args.grid_result)
    args.checkpoint = absolute_path(args.checkpoint)
    args.tube_output = absolute_path(args.tube_output)

    for label, path in (
        ("real trajectories", args.real),
        ("grid result", args.grid_result),
        ("checkpoint", args.checkpoint),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    ensure_parent(args.tube_output)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    print("========== Tube paths ==========")
    print(f"real trajectories : {args.real}")
    print(f"grid result       : {args.grid_result}")
    print(f"checkpoint        : {args.checkpoint}")
    print(f"tube output       : {args.tube_output}")
    print(f"device            : {device}")

    splits = load_real_trajectories(args.real)
    model, mean, std, checkpoint = load_predictor_checkpoint(
        args.checkpoint, device
    )

    protocol = checkpoint.get("training_protocol", {})
    if protocol.get("calibration_source") != "val_traj":
        raise ValueError(
            "checkpoint does not reserve val_traj for calibration; "
            "retrain it with train_predictor.py"
        )

    source_grid, grid, cell_bounds = load_grid(args.grid_result)
    if grid.ndim != model.state_dim:
        raise ValueError(
            f"grid_dim={grid.ndim} does not match model state_dim={model.state_dim}"
        )

    expected = (model.horizon + 1, model.state_dim)
    for key in ("val_traj", "test_traj"):
        if splits[key].shape[1:] != expected:
            raise ValueError(
                f"{key} shape {splits[key].shape} does not match expected (*, {expected})"
            )

    raw_tubes = build_raw_tubes(
        model=model,
        mean=mean,
        std=std,
        cell_bounds=cell_bounds,
        samples_per_dim=args.samples_per_dim,
        batch_size=args.cell_batch_size,
        device=device,
    )

    scores, calibration_outside = calibration_scores(
        splits["val_traj"], raw_tubes, grid
    )
    delta, rank = conformal_delta(scores, args.alpha)
    certified_tubes = inflate_tubes(raw_tubes, delta)

    raw_evaluation = evaluate_tubes(splits["test_traj"], raw_tubes, grid)
    certified_evaluation = evaluate_tubes(
        splits["test_traj"], certified_tubes, grid
    )

    print("========== Conformal calibration ==========")
    print(f"calibration scores  : {len(scores)}")
    print(f"outside grid        : {calibration_outside}")
    print(f"alpha               : {args.alpha}")
    print(f"conformal rank      : {rank}")
    print(f"conformal delta     : {delta:.10f}")

    print("========== Test containment ==========")
    print(
        "raw tube       : "
        f"{raw_evaluation['fully_contained']}/"
        f"{raw_evaluation['in_grid_trajectories']} "
        f"({100.0 * raw_evaluation['containment_rate']:.2f}%)"
    )
    print(
        "conformal tube : "
        f"{certified_evaluation['fully_contained']}/"
        f"{certified_evaluation['in_grid_trajectories']} "
        f"({100.0 * certified_evaluation['containment_rate']:.2f}%)"
    )

    save_tube_json(
        path=args.tube_output,
        source_grid=source_grid,
        grid=grid,
        cell_bounds=cell_bounds,
        raw_tubes=raw_tubes,
        certified_tubes=certified_tubes,
        scores=scores,
        calibration_outside=calibration_outside,
        delta=delta,
        rank=rank,
        raw_evaluation=raw_evaluation,
        certified_evaluation=certified_evaluation,
        checkpoint=checkpoint,
        real_path=args.real,
        grid_path=args.grid_result,
        checkpoint_path=args.checkpoint,
        samples_per_dim=args.samples_per_dim,
        alpha=args.alpha,
        environment=args.env_name,
    )
    print(f"[Saved] predictor tube: {args.tube_output}")


if __name__ == "__main__":
    main()
