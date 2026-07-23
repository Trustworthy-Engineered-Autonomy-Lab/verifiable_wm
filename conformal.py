#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split-conformal calibration of the reachable-tube inflation Delta.

Follows the paper's Theorem 1 ("Confident Reachable Tube Containment"):
non-conformity score delta_i = max_t ||s_t - s_hat_t||_2 over a held-out
calibration split, then Delta_{1-alpha} is the finite-sample-corrected
(1-alpha) quantile of {delta_i}. Applying that Delta to inflate every
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


def conformal_quantile(scores: np.ndarray, *, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("calibration set must contain at least one score")

    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf

    return float(np.sort(scores)[k - 1])


def calibrate_delta(
    *,
    real_path: Path,
    dwm_path: Path,
    dims: Sequence[int],
    alpha: float,
    split: str = "val",
    circular_dims: Sequence[int] = (),
    period: float = 2 * math.pi,
) -> dict:
    real_key = f"{split}_traj"
    with np.load(real_path, allow_pickle=False) as real_data, np.load(
        dwm_path, allow_pickle=False
    ) as dwm_data:
        real_traj = real_data[real_key]
        dwm_traj = dwm_data[real_key]

    scores = nonconformity_scores(
        real_traj,
        dwm_traj,
        dims=dims,
        circular_dims=circular_dims,
        period=period,
    )
    delta = conformal_quantile(scores, alpha=alpha)

    return {
        "split": split,
        "n": int(scores.shape[0]),
        "alpha": float(alpha),
        "dims": list(dims),
        "circular_dims": list(circular_dims),
        "delta": delta,
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
    }


ENV_CHECK_DIMS = {
    "cartpole": (0, 2),
    "mountain_car": (0, 1),
    "pendulum": (0, 1),
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

    result = calibrate_delta(
        real_path=real_path,
        dwm_path=dwm_path,
        dims=ENV_CHECK_DIMS[args.env],
        alpha=args.alpha,
        split=args.split,
        circular_dims=ENV_CIRCULAR_DIMS.get(args.env, ()),
    )
    result["env"] = args.env
    result["variant"] = args.variant

    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
