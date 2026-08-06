#!/usr/bin/env python3
"""Calibrate and write an inflated StarV-format reachable tube."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import evaluate_tube as evaluation


def conformal_quantile_with_rank(
    scores: Sequence[float], *, alpha: float
) -> tuple[float, int]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not len(values):
        raise ValueError("at least one score is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    rank = math.ceil((len(values) + 1) * (1.0 - alpha))
    if rank > len(values):
        raise ValueError(
            f"calibration set is too small for alpha={alpha}: rank={rank}, n={len(values)}"
        )
    return float(np.sort(values)[rank - 1]), rank


def calibration_gamma(
    calibration_trajectories: np.ndarray,
    grid: evaluation.GridInfo,
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int],
    *,
    alpha: float,
) -> tuple[float, float, int]:
    rows = evaluation.evaluate_trajectories(
        calibration_trajectories, grid, cells, dims
    )
    scores = -np.asarray([row["signed_margin"] for row in rows], dtype=float)
    gamma_raw, rank = conformal_quantile_with_rank(scores, alpha=alpha)
    gamma = max(0.0, gamma_raw)
    if not np.isfinite(gamma):
        raise ValueError("calibrated inflation gamma must be finite")
    return float(gamma), float(-gamma_raw), int(rank)


def inflate_cells(
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int],
    epsilons: Sequence[float],
) -> list[dict[str, Any]]:
    """Copy cells and expand selected future bounds, preserving bounds[0]."""
    if len(dims) != len(epsilons):
        raise ValueError("each selected dimension needs one inflation epsilon")
    if any(not np.isfinite(value) or value < 0.0 for value in epsilons):
        raise ValueError("inflation epsilons must be finite and nonnegative")
    inflated = copy.deepcopy(cells)
    for cell_index, cell in enumerate(inflated):
        history = cell.get("bounds", [])
        if not history:
            raise ValueError(f"tube cell {cell_index} has no bounds")
        for state_bounds in history[1:]:
            for dim, epsilon in zip(dims, epsilons):
                if dim < 0 or dim >= len(state_bounds):
                    raise ValueError(
                        f"tube cell {cell_index} does not contain selected dimension {dim}"
                    )
                values = np.asarray(state_bounds[dim], dtype=float).reshape(-1)
                if values.size < 2 or values.size % 2:
                    raise ValueError("tube bounds must contain low/high pairs")
                if not np.all(np.isfinite(values)):
                    raise ValueError("tube bounds must be finite")
                values[0::2] -= float(epsilon)
                values[1::2] += float(epsilon)
                state_bounds[dim] = values.tolist()
    return inflated


def inflate_tube_payload(
    payload: dict[str, Any],
    dims: Sequence[int],
    gamma: float,
) -> dict[str, Any]:
    if "cells" not in payload:
        raise ValueError("tube payload is missing cells")
    inflated = copy.deepcopy(payload)
    inflated["cells"] = inflate_cells(
        payload["cells"], dims, [gamma] * len(dims)
    )
    return inflated


def _require_writable_outputs(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing output(s): " + ", ".join(existing)
        )


def run_inflation(
    *,
    tube_path: Path,
    calibration_path: Path,
    calibration_key: str,
    dims: Sequence[int],
    alpha: float,
    output_path: Path,
    calibration_output_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    output_path = Path(output_path)
    calibration_output_path = Path(calibration_output_path)
    _require_writable_outputs(
        (output_path, calibration_output_path), overwrite
    )
    payload, grid, cells = evaluation.load_tube(tube_path)
    trajectories = evaluation.load_trajectories(
        calibration_path, calibration_key
    )
    gamma, rank_margin, rank = calibration_gamma(
        trajectories, grid, cells, dims, alpha=alpha
    )
    inflated_payload = inflate_tube_payload(payload, dims, gamma)
    calibration = {
        "method": "finite-sample conformal quantile of negative signed trajectory margins",
        "gamma": float(gamma),
        "signed_margin_at_rank": float(rank_margin),
        "rank": int(rank),
        "alpha": float(alpha),
        "calibration_size": int(len(trajectories)),
        "check_dims": [int(dim) for dim in dims],
        "inflates_initial_cell": False,
        "source_tube": str(Path(tube_path)),
        "calibration_trajectories": str(Path(calibration_path)),
        "calibration_key": calibration_key,
        "inflated_tube": str(output_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(inflated_payload, indent=2), encoding="utf-8"
    )
    calibration_output_path.write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    return calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="tube postprocess config JSON")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = evaluation.load_postprocess_config(args.config, "inflation")
    calibration = run_inflation(**config, overwrite=args.overwrite)
    print(f"gamma: {calibration['gamma']:.12g}")
    print(f"inflated tube: {calibration['inflated_tube']}")


if __name__ == "__main__":
    main()
