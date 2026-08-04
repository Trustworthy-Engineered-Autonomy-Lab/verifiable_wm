#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare brake-system StarV safety map against the CARLA camera ground truth.

Ground truth: datasets/brake_system/ground_truth/gt_grid_out_*/judge.npy, a (40, 40)
matrix indexed [vel_cell, dist_cell] where 0 = safe and 1 = unsafe (see
gt_brake_grid_eval.py: judge[j, i] = 0 if is_safe else 1).

Prediction: results/brake_system/safety_result.json from verify.py, where a
cell is safe iff the reachable distance lower bound stayed > 0 for all steps.

    python tools/compare_brake_ground_truth.py \
        --safety results/brake_system/safety_result.json \
        --gt aebs_carla/ground_truth/gt_grid_out_0.01
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.visualize import safety_matrix_from_cells  # noqa: E402


def load_prediction(safety_path):
    with open(safety_path) as f:
        data = json.load(f)
    return safety_matrix_from_cells(data["grid"]["dims"], data["cells"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--safety", type=Path,
        default=PROJECT_ROOT / "results/brake_system/safety_result.json",
    )
    parser.add_argument(
        "--gt", type=Path,
        default=PROJECT_ROOT / "datasets/brake_system/ground_truth/gt_grid_out_0.01",
    )
    args = parser.parse_args()

    pred = load_prediction(args.safety)
    # judge is (vel, dist) with 0 = safe; convert to (dist, vel) with 1 = safe
    # to match the prediction layout (grid dims order: dis, vel).
    judge = (np.load(args.gt / "judge.npy") == 0).astype(np.int8).T
    if pred.shape != judge.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, gt={judge.shape}")

    tp = int(((pred == 1) & (judge == 1)).sum())
    fp = int(((pred == 1) & (judge == 0)).sum())
    fn = int(((pred == 0) & (judge == 1)).sum())
    tn = int(((pred == 0) & (judge == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else float("nan")
    )

    result = {
        "cells": int(pred.size),
        "pred_safe_rate": float((pred == 1).mean()),
        "gt_safe_rate": float((judge == 1).mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / pred.size,
        "safety_path": str(args.safety),
        "gt_path": str(args.gt),
    }
    print(json.dumps(result, indent=2))

    out_name = Path(args.safety).stem.replace("safety_result", "gt_comparison", 1) + ".json"
    out_path = Path(args.safety).with_name(out_name)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
