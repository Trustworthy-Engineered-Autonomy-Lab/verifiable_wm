#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score a StarV safety map against the camera roll-out ground truth.

Both maps label every initial cell safe or unsafe: the prediction comes from
verify.py, the ground truth from running the true closed loop with real camera
images (tools/gt_grid_eval.py for the gym benchmarks, tools/gt_brake_grid_eval.py
for the braking system). Comparing them cell by cell gives the precision,
recall and F1 behind Figure "4case_wm_verification" -- a verified-safe cell
that the camera loop also reaches the goal from is a true positive, and a cell
the verifier refuses but the camera loop clears is the conservatism the figure
shades light green.

    python -m tools.compare_ground_truth \\
        --safety results/pendulum/safety_result.json \\
        --gt datasets/pendulum/ground_truth/gt_grid.npz

    python -m tools.compare_ground_truth \\
        --safety results/brake_system/safety_result.json \\
        --gt datasets/brake_system/ground_truth/gt_grid_out_0.01

Soundness is the interesting direction: a false positive means the verifier
certified a cell the real system fails, which must not happen.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.visualize import safety_matrix_from_cells, varying_dims  # noqa: E402


def load_prediction(safety_path: Path):
    """Return the verifier's safety map and the names of its varying dims."""
    data = json.loads(Path(safety_path).read_text(encoding="utf-8"))
    dims = data["grid"]["dims"]
    return safety_matrix_from_cells(dims, data["cells"]), [
        str(dim.get("name", "")) for dim in varying_dims(dims)
    ]


def load_ground_truth(gt_path: Path, dim_names):
    """Return a 1 = safe ground-truth map laid out like the prediction.

    Two producers, two conventions:
      *.npz  tools/gt_grid_eval.py, already grid-shaped in the StarV dimension
             order with 1 = the goal was reached from every sample.
      *.npy  tools/gt_brake_grid_eval.py, indexed [velocity, distance] with
             0 = safe, i.e. transposed relative to the brake grid's (dis, vel)
             order and inverted.
    """
    gt_path = Path(gt_path)
    if gt_path.is_dir():
        gt_path = gt_path / "judge.npy"

    if gt_path.suffix == ".npz":
        with np.load(gt_path, allow_pickle=False) as data:
            if "judge" not in data:
                raise KeyError(f"{gt_path} has no 'judge' array")
            judge = np.asarray(data["judge"])
            recorded = [str(name) for name in data["dims"]] if "dims" in data else None
        judge = judge.squeeze()
        if recorded is not None:
            varying = [name for name in recorded if name in dim_names]
            if varying and varying != list(dim_names):
                raise ValueError(
                    f"ground-truth dimension order {varying} does not match the "
                    f"safety map's {list(dim_names)}"
                )
        return judge.astype(np.int8)

    if gt_path.suffix == ".npy":
        judge = np.asarray(np.load(gt_path))
        if judge.ndim != 2:
            raise ValueError(f"expected a 2-D judge matrix, got {judge.shape}")
        return (judge == 0).astype(np.int8).T

    raise ValueError(f"unsupported ground-truth file: {gt_path}")


def confusion(prediction: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Treat 'verified safe' as the positive class."""
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"shape mismatch: prediction={prediction.shape}, "
            f"ground truth={ground_truth.shape}"
        )
    tp = int(((prediction == 1) & (ground_truth == 1)).sum())
    fp = int(((prediction == 1) & (ground_truth == 0)).sum())
    fn = int(((prediction == 0) & (ground_truth == 1)).sum())
    tn = int(((prediction == 0) & (ground_truth == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else float("nan")
    )
    return {
        "cells": int(prediction.size),
        "pred_safe_rate": float((prediction == 1).mean()),
        "gt_safe_rate": float((ground_truth == 1).mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / prediction.size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safety", type=Path, required=True,
                        help="safety_result.json written by verify.py")
    parser.add_argument("--gt", type=Path, required=True,
                        help="gt_grid.npz, judge.npy, or the directory holding judge.npy")
    args = parser.parse_args()

    prediction, dim_names = load_prediction(args.safety)
    ground_truth = load_ground_truth(args.gt, dim_names)
    result = confusion(prediction, ground_truth)
    result["dims"] = dim_names
    result["safety_path"] = str(args.safety)
    result["gt_path"] = str(args.gt)
    print(json.dumps(result, indent=2))

    out_name = args.safety.stem.replace("safety_result", "gt_comparison", 1)
    if out_name == args.safety.stem:
        out_name = f"{out_name}_gt_comparison"
    out_path = args.safety.with_name(out_name + ".json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
