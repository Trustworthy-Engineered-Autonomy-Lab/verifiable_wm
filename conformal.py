#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split-conformal calibration of the reachable-tube inflation radius Gamma.

Follows the paper's Theorem 1 ("Confident Reachable Tube Containment"):
non-conformity score delta_i = max_t ||s_t - s_hat_t||_2 over a held-out
calibration split, then Gamma_{1-alpha} is the finite-sample-corrected
(1-alpha) quantile of {delta_i}. Applying that Gamma to inflate every
checked dimension of the reachable tube (as compare.py's --delta already
does) is a valid, if conservative, sufficient condition for containment
under the L2 bound.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


EXPECTED_HORIZON = 20
ENV_HORIZON = {
    "cartpole": 20,
    "mountain_car": 20,
    "pendulum": 20,
    "brake_system": 10,
}


def nonconformity_scores(
    real: np.ndarray,
    dwm: np.ndarray,
    *,
    dims: Sequence[int],
    circular_dims: Sequence[int] = (),
    period: float = 2 * math.pi,
) -> np.ndarray:
    real = np.asarray(real, dtype=float)
    dwm = np.asarray(dwm, dtype=float)
    if real.shape != dwm.shape or real.ndim != 3:
        raise ValueError(
            f"trajectory shape mismatch: real={real.shape}, dwm={dwm.shape}"
        )

    diff = dwm[..., list(dims)] - real[..., list(dims)]
    if circular_dims:
        dim_positions = {d: i for i, d in enumerate(dims)}
        diff = diff.copy()
        for dim in circular_dims:
            if dim not in dim_positions:
                continue
            i = dim_positions[dim]
            diff[..., i] = (diff[..., i] + period / 2) % period - period / 2

    l2_per_step = np.linalg.norm(diff, ord=2, axis=-1)
    return l2_per_step.max(axis=1)


def _conformal_quantile_with_rank(
    scores: np.ndarray, *, alpha: float
) -> tuple[float, int]:
    scores = np.asarray(scores, dtype=float)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = scores.shape[0]
    if n == 0:
        raise ValueError("calibration set must contain at least one score")

    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf, k

    return float(np.sort(scores)[k - 1]), k


def conformal_quantile(scores: np.ndarray, *, alpha: float) -> float:
    gamma, _ = _conformal_quantile_with_rank(scores, alpha=alpha)
    return gamma


def _validate_calibration_artifacts(
    *,
    real_traj: np.ndarray,
    dwm_traj: np.ndarray,
    real_actions: np.ndarray,
    dwm_actions: np.ndarray,
    real_horizon: int,
    dwm_horizon: int,
    expected_horizon: int = EXPECTED_HORIZON,
) -> None:
    if real_horizon != expected_horizon or dwm_horizon != expected_horizon:
        raise ValueError(
            "conformal calibration requires horizon "
            f"{expected_horizon}, got real={real_horizon}, dwm={dwm_horizon}"
        )

    if real_traj.shape != dwm_traj.shape or real_traj.ndim != 3:
        raise ValueError(
            f"trajectory shape mismatch: real={real_traj.shape}, dwm={dwm_traj.shape}"
        )
    expected_traj_shape = (real_traj.shape[0], expected_horizon + 1)
    if real_traj.shape[:2] != expected_traj_shape:
        raise ValueError(
            "trajectory horizon mismatch: expected "
            f"(N, {expected_horizon + 1}, d), got {real_traj.shape}"
        )

    if real_actions.shape != dwm_actions.shape or real_actions.ndim != 3:
        raise ValueError(
            f"actions shape mismatch: real={real_actions.shape}, "
            f"dwm={dwm_actions.shape}"
        )
    expected_action_shape = (real_traj.shape[0], expected_horizon)
    if real_actions.shape[:2] != expected_action_shape:
        raise ValueError(
            "actions shape mismatch: expected "
            f"(N, {expected_horizon}, a), got {real_actions.shape}"
        )

    arrays = {
        "real trajectories": real_traj,
        "DWM trajectories": dwm_traj,
        "real actions": real_actions,
        "DWM actions": dwm_actions,
    }
    for label, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{label} must contain only finite values")

    if not np.array_equal(real_traj[:, 0, :], dwm_traj[:, 0, :]):
        raise ValueError("real and DWM initial state mismatch")


def calibrate_gamma(
    *,
    real_path: Path,
    dwm_path: Path,
    dims: Sequence[int],
    alpha: float,
    split: str = "val",
    circular_dims: Sequence[int] = (),
    period: float = 2 * math.pi,
    horizon: int = EXPECTED_HORIZON,
) -> dict:
    real_key = f"{split}_traj"
    with np.load(real_path, allow_pickle=False) as real_data, np.load(
        dwm_path, allow_pickle=False
    ) as dwm_data:
        real_traj = real_data[real_key]
        dwm_traj = dwm_data[real_key]
        actions_key = f"{split}_actions"
        real_actions = real_data[actions_key]
        dwm_actions = dwm_data[actions_key]
        real_horizon = int(real_data["rollout_steps"].item())
        real_starv_config = str(real_data["starv_config"].item())
        real_controller_weights = str(real_data["controller_weights"].item())
        dwm_horizon = int(dwm_data["rollout_steps"].item())
        dwm_starv_config = str(dwm_data["starv_config"].item())
        dwm_controller_weights = str(dwm_data["controller_weights"].item())
        decoder_weights = str(dwm_data["decoder_weights"].item())

    if real_starv_config != dwm_starv_config:
        raise ValueError(
            "starv_config mismatch: "
            f"real={real_starv_config}, dwm={dwm_starv_config}"
        )
    if real_controller_weights != dwm_controller_weights:
        raise ValueError(
            "controller_weights mismatch: "
            f"real={real_controller_weights}, dwm={dwm_controller_weights}"
        )

    _validate_calibration_artifacts(
        real_traj=real_traj,
        dwm_traj=dwm_traj,
        real_actions=real_actions,
        dwm_actions=dwm_actions,
        real_horizon=real_horizon,
        dwm_horizon=dwm_horizon,
        expected_horizon=horizon,
    )

    scores = nonconformity_scores(
        real_traj,
        dwm_traj,
        dims=dims,
        circular_dims=circular_dims,
        period=period,
    )
    gamma, rank = _conformal_quantile_with_rank(scores, alpha=alpha)

    return {
        "split": split,
        "n": int(scores.shape[0]),
        "alpha": float(alpha),
        "rank": rank,
        "horizon": horizon,
        "norm": "L2",
        "dims": list(dims),
        "circular_dims": list(circular_dims),
        "gamma": gamma,
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "real_path": str(real_path),
        "dwm_path": str(dwm_path),
        "decoder_weights": decoder_weights,
        "starv_config": real_starv_config,
        "controller_weights": real_controller_weights,
    }


ENV_CHECK_DIMS = {
    "cartpole": (0, 2),
    "mountain_car": (0, 1),
    "pendulum": (0, 1),
    "brake_system": (0, 1),
}
ENV_CIRCULAR_DIMS = {
    "pendulum": (0,),
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=sorted(ENV_CHECK_DIMS))
    parser.add_argument(
        "--variant", default="saliency", help="DWM trajectory variant suffix"
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--real", type=Path, default=None)
    parser.add_argument("--dwm", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    data_dir = Path("datasets") / args.env / "data" / "dataset_v1"
    real_path = args.real or data_dir / "real_trajectories.npz"
    dwm_path = args.dwm or data_dir / f"dwm_trajectories_{args.variant}.npz"

    result = calibrate_gamma(
        real_path=real_path,
        dwm_path=dwm_path,
        dims=ENV_CHECK_DIMS[args.env],
        alpha=args.alpha,
        split=args.split,
        circular_dims=ENV_CIRCULAR_DIMS.get(args.env, ()),
        horizon=ENV_HORIZON[args.env],
    )
    result["env"] = args.env
    result["variant"] = args.variant

    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
