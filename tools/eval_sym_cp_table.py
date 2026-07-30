#!/usr/bin/env python3
"""Fill one Table `tab:tube-comparison` block: symbolic tube, raw / CP-D / CP-R.

For a StarV `safety_result*.json` (the symbolic reachable tube over a grid of
initial cells) and the paired real / surrogate rollout NPZ files, this reports
the three rows of the paper's "(a) Symbolic reachability" block:

  Sym  --             raw tube, no inflation
  Sym (inflate) CP-D  inflated by Delta_{1-alpha}, the conformal quantile of the
                      trajectory-discrepancy score  delta_i = max_t ||s_t - s^_t||_p
                      (p=2 by default: the paper's appendix Delta values only
                      reproduce under L2, matching conformal.py)
  Sym (inflate) CP-R  inflated by Gamma_{1-alpha}, the conformal quantile of the
                      tube-robustness score        gamma_i = max_t sd(s_t, R_t)

Both scores are calibrated on one split (default `val`) and coverage is reported
on a disjoint split (default `test`).  Coverage counts a real trajectory as
covered when *every* state over the horizon lies inside the (inflated) tube of
the cell its initial state falls into.  Area is the per-cell, time-averaged 2-D
box area of the tube over t=1..T, then averaged over all cells (unnormalized,
same convention as the paper's A-bar column).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import signed_tube_margin as stm  # noqa: E402
from conformal import conformal_quantile_with_rank  # noqa: E402


ENV_DIMS = stm.ENV_DIMS
# Angle dimensions whose discrepancy has to be wrapped into (-pi, pi].
ENV_CIRCULAR_DIMS = {"pendulum": (0,)}


def discrepancy_scores(
    real: np.ndarray,
    surrogate: np.ndarray,
    dims: Sequence[int],
    *,
    order: float,
    circular_dims: Sequence[int] = (),
    period: float = 2 * math.pi,
) -> np.ndarray:
    """CP-D score: max_t ||s_t - s^_t||_order over the selected dimensions."""
    real = np.asarray(real, dtype=float)
    surrogate = np.asarray(surrogate, dtype=float)
    if real.shape != surrogate.shape or real.ndim != 3:
        raise ValueError(f"shape mismatch: real={real.shape}, surrogate={surrogate.shape}")
    if not np.allclose(real[:, 0, :], surrogate[:, 0, :]):
        raise ValueError("real and surrogate rollouts do not share initial states")

    diff = surrogate[..., list(dims)] - real[..., list(dims)]
    for dim in circular_dims:
        if dim in tuple(dims):
            index = list(dims).index(dim)
            diff[..., index] = (diff[..., index] + period / 2) % period - period / 2
    return np.linalg.norm(diff, ord=order, axis=-1).max(axis=1)


_PAD_LOW = 1e18
_PAD_HIGH = -1e18


class TubeAreas:
    """Vectorized per-cell tube area as a function of the inflation delta.

    Each (cell, t, dim) bound is a union of one or more (possibly overlapping
    or nested) intervals.  Bounds are stored sorted and zero-padded so that the
    union length under a symmetric inflation can be swept in a few vector ops,
    which keeps a 5000-cell x 20-step tube cheap to re-evaluate at many deltas.
    """

    def __init__(self, cells: Sequence[dict[str, Any]], dims: Sequence[int]):
        parsed: list[list[tuple[float, float]]] = []
        n_cells = 0
        n_steps = None
        max_intervals = 1
        for cell in cells:
            if "error_msg" in cell:
                continue
            history = cell.get("bounds", [])
            if len(history) < 2:
                raise ValueError("tube history needs an initial state and >=1 future state")
            if n_steps is None:
                n_steps = len(history) - 1
            elif n_steps != len(history) - 1:
                raise ValueError("cells disagree on horizon length")
            n_cells += 1
            for bounds in history[1:]:  # t=0 is the initial cell itself
                for dim in dims:
                    intervals = sorted(stm.interval_pairs(bounds[dim]))
                    max_intervals = max(max_intervals, len(intervals))
                    parsed.append(intervals)
        if not n_cells:
            raise ValueError("no error-free cells in the safety result")

        lows = np.full((len(parsed), max_intervals), _PAD_LOW)
        highs = np.full((len(parsed), max_intervals), _PAD_HIGH)
        for index, intervals in enumerate(parsed):
            lows[index, : len(intervals)] = [low for low, _ in intervals]
            highs[index, : len(intervals)] = [high for _, high in intervals]

        shape = (n_cells, n_steps, len(dims), max_intervals)
        self.lows = lows.reshape(shape)
        self.highs = highs.reshape(shape)
        self.n_cells = n_cells
        self.n_steps = n_steps

    def _union_lengths(self, delta: float) -> np.ndarray:
        """Sweep the sorted intervals, accumulating the inflated union length."""
        total = np.zeros(self.lows.shape[:-1])
        running_max = np.full(self.lows.shape[:-1], -np.inf)
        for index in range(self.lows.shape[-1]):
            low = self.lows[..., index] - delta
            high = self.highs[..., index] + delta
            start = np.maximum(low, running_max)
            total += np.maximum(0.0, high - start)
            running_max = np.maximum(running_max, high)
        return total

    def per_cell(self, delta: float = 0.0) -> np.ndarray:
        if delta < 0.0:
            raise ValueError("tube inflation delta must be nonnegative")
        return np.prod(self._union_lengths(delta), axis=-1).mean(axis=1)


def row(
    label: str,
    delta: float,
    margins: np.ndarray,
    areas: TubeAreas,
) -> dict[str, Any]:
    """Coverage on the eval split and mean/std tube area at one inflation."""
    per_cell = areas.per_cell(delta)
    return {
        "row": label,
        "delta": float(delta),
        "coverage": float((margins >= -delta).mean()),
        "covered": int((margins >= -delta).sum()),
        "n_eval": int(margins.size),
        "area_mean": float(per_cell.mean()),
        "area_std": float(per_cell.std()),
        "n_cells": int(areas.n_cells),
    }


def trajectory_margins(
    trajectories: np.ndarray,
    grid: stm.GridInfo,
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int],
) -> np.ndarray:
    """Smallest signed margin per trajectory (positive = inside the raw tube)."""
    rows = stm.evaluate_set(trajectories, grid, cells, dims, 0.0)
    invalid = [r for r in rows if r["status"] != "valid"]
    if invalid:
        raise ValueError(
            f"{len(invalid)} trajectories could not be matched to a tube cell, "
            f"first error: {invalid[0]['error']}"
        )
    return np.asarray([r["signed_margin"] for r in rows], dtype=float)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, choices=sorted(ENV_DIMS))
    parser.add_argument("--safety", type=Path, required=True, help="StarV safety_result JSON")
    parser.add_argument("--real", type=Path, required=True, help="real trajectory NPZ")
    parser.add_argument("--surrogate", type=Path, required=True, help="DWM / cGAN trajectory NPZ")
    parser.add_argument("--decoder", default="DWM", help="decoder name used in the printed LaTeX rows")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cal-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--check-dims", type=int, nargs=2, default=None)
    parser.add_argument("--norm", default="l2", choices=["l1", "l2"], help="CP-D score norm")
    parser.add_argument("--no-wrap", action="store_true", help="disable angle wrapping in the CP-D score")
    parser.add_argument("--output", type=Path, default=None, help="write the full summary JSON here")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.cal_split == args.eval_split:
        raise SystemExit("calibration and evaluation splits must be disjoint")

    dims = tuple(args.check_dims) if args.check_dims is not None else ENV_DIMS[args.env]
    circular = () if args.no_wrap else ENV_CIRCULAR_DIMS.get(args.env, ())
    order = 1 if args.norm == "l1" else 2

    grid, cells = stm.load_safety_result(args.safety)
    areas = TubeAreas(cells, dims)

    real = np.load(args.real, allow_pickle=False)
    surrogate = np.load(args.surrogate, allow_pickle=False)
    cal_margins = trajectory_margins(real[f"{args.cal_split}_traj"].astype(float), grid, cells, dims)
    eval_margins = trajectory_margins(real[f"{args.eval_split}_traj"].astype(float), grid, cells, dims)

    # CP-D: trajectory discrepancy between the real and surrogate rollouts.
    delta_scores = discrepancy_scores(
        real[f"{args.cal_split}_traj"],
        surrogate[f"{args.cal_split}_traj"],
        dims,
        order=order,
        circular_dims=circular,
    )
    delta_cp, delta_rank = conformal_quantile_with_rank(delta_scores, alpha=args.alpha)

    # CP-R: signed distance of the real trajectory to the tube boundary.
    gamma_scores = -cal_margins
    gamma_cp, gamma_rank = conformal_quantile_with_rank(gamma_scores, alpha=args.alpha)
    gamma_inflation = max(0.0, gamma_cp)

    rows = [
        row("Sym --", 0.0, eval_margins, areas),
        row("Sym (inflate) CP-D", delta_cp, eval_margins, areas),
        row("Sym (inflate) CP-R", gamma_inflation, eval_margins, areas),
    ]

    summary = {
        "env": args.env,
        "decoder": args.decoder,
        "check_dims": list(dims),
        "alpha": args.alpha,
        "cal_split": args.cal_split,
        "eval_split": args.eval_split,
        "horizon": areas.n_steps,
        "inputs": {
            "safety": str(args.safety),
            "real": str(args.real),
            "surrogate": str(args.surrogate),
        },
        "cp_d": {
            "norm": args.norm.upper(),
            "circular_dims": list(circular),
            "n_cal": int(delta_scores.size),
            "rank": delta_rank,
            "Delta": delta_cp,
            "score_min": float(delta_scores.min()),
            "score_median": float(np.median(delta_scores)),
            "score_max": float(delta_scores.max()),
        },
        "cp_r": {
            "n_cal": int(gamma_scores.size),
            "rank": gamma_rank,
            "Gamma": gamma_cp,
            "inflation": gamma_inflation,
            "score_min": float(gamma_scores.min()),
            "score_median": float(np.median(gamma_scores)),
            "score_max": float(gamma_scores.max()),
        },
        "rows": rows,
    }

    print(f"env={args.env} decoder={args.decoder} dims={dims} alpha={args.alpha} "
          f"cal={args.cal_split}({delta_scores.size}) eval={args.eval_split}({eval_margins.size})")
    print(f"  CP-D Delta = {delta_cp:.6f}  ({args.norm.upper()}, rank {delta_rank}, wrap dims {list(circular)})")
    print(f"  CP-R Gamma = {gamma_cp:.6f}  (rank {gamma_rank}, inflation {gamma_inflation:.6f})")
    print()
    for item in rows:
        print(f"  {item['row']:<20} delta={item['delta']:.6f}  "
              f"Cov={item['coverage'] * 100:6.2f}%  "
              f"Abar={item['area_mean']:.6f} +- {item['area_std']:.6f}")
    print()
    print("LaTeX (Cov & Abar):")
    for item in rows:
        cp = {"Sym --": "--", "Sym (inflate) CP-D": "CP-D", "Sym (inflate) CP-R": "CP-R"}[item["row"]]
        construction = "Sym" if cp == "--" else "Sym (inflate)"
        print(f"  {args.decoder} & {construction} & {cp} "
              f"&{item['coverage'] * 100:.2f}\\% &{item['area_mean']:.6f} \\\\")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nsummary: {args.output}")


if __name__ == "__main__":
    main()
