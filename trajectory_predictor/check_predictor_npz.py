#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check predictor_trajectories.npz against its real-trajectory reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SPLIT_KEYS = ("train_traj", "val_traj", "test_traj")
CELL_KEYS = (
    "cell_indices",
    "initial_bounds",
    "initial_states",
    "trajectories",
    "lower",
    "upper",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate real_trajectories-compatible predictor output."
    )
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.real, allow_pickle=False) as real, np.load(
        args.predictor, allow_pickle=False
    ) as predictor:
        expected_splits = [key for key in SPLIT_KEYS if key in real.files]
        if not expected_splits:
            raise KeyError(f"{args.real} contains no standard trajectory splits")

        for key in expected_splits:
            if key not in predictor.files:
                raise KeyError(f"predictor output is missing {key}")
            if predictor[key].shape != real[key].shape:
                raise ValueError(
                    f"{key} shape mismatch: predictor={predictor[key].shape}, "
                    f"real={real[key].shape}"
                )
            if predictor[key].dtype != np.dtype(np.float32):
                raise ValueError(
                    f"{key} must use float32, got {predictor[key].dtype}"
                )
            if not np.isfinite(predictor[key]).all():
                raise ValueError(f"{key} contains NaN or Inf")

            action_key = key.replace("_traj", "_actions")
            if action_key in real.files:
                if action_key not in predictor.files:
                    raise KeyError(f"predictor output is missing {action_key}")
                if predictor[action_key].shape != real[action_key].shape:
                    raise ValueError(
                        f"{action_key} shape mismatch: "
                        f"predictor={predictor[action_key].shape}, "
                        f"real={real[action_key].shape}"
                    )

        missing_cell_keys = [
            key for key in CELL_KEYS if key not in predictor.files
        ]
        if missing_cell_keys:
            raise KeyError(
                f"predictor output is missing cell arrays {missing_cell_keys}"
            )

        trajectories = predictor["trajectories"]
        initial_states = predictor["initial_states"]
        initial_bounds = predictor["initial_bounds"]
        lower = predictor["lower"]
        upper = predictor["upper"]
        if trajectories.ndim != 4:
            raise ValueError(
                "trajectories must have shape "
                "(cells, samples, horizon+1, state_dim)"
            )
        cells, samples, state_count, state_dim = trajectories.shape
        expected_shapes = {
            "cell_indices": (cells,),
            "initial_states": (cells, samples, state_dim),
            "initial_bounds": (cells, state_dim, 2),
            "lower": (cells, state_count, state_dim),
            "upper": (cells, state_count, state_dim),
        }
        for key, expected in expected_shapes.items():
            if predictor[key].shape != expected:
                raise ValueError(
                    f"{key} shape mismatch: got {predictor[key].shape}, "
                    f"expected {expected}"
                )

        print("NPZ format check passed")
        print(f"reference splits : {expected_splits}")
        print(f"cell trajectories: {trajectories.shape}")


if __name__ == "__main__":
    main()
