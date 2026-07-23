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
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_prediction(safety_path):
    with open(safety_path) as f:
        data = json.load(f)

    cells = data["cells"]
    dims = data["grid"]["dims"]
    edges = [
        np.linspace(dim["start"], dim["stop"], dim["num"] + 1)
        for dim in dims
    ]
    shape = tuple(dim["num"] for dim in dims)
    pred = np.full(shape, -1, dtype=np.int8)

    for cell in cells:
        init = np.array(cell["bounds"][0])  # (dims, 2): [lo, hi] per dim
        idx = tuple(
            int(np.clip(np.searchsorted(edges[d], init[d].mean()) - 1, 0, shape[d] - 1))
        for d in range(len(dims))
        )
        pred[idx] = 1 if cell.get("result") else 0

    if (pred < 0).any():
        raise ValueError(f"{(pred < 0).sum()} grid cells missing from {safety_path}")
    return pred


SAFE_COLOR = "#0ca30c"      # status good
UNSAFE_COLOR = "#d03b3b"    # status critical
AGREE_SAFE = "#f0f0ee"
AGREE_UNSAFE = "#9a9a94"
FN_COLOR = "#fab219"        # conservative: StarV unsafe, camera safe
FP_COLOR = "#d03b3b"        # dangerous: StarV safe, camera unsafe


def plot_maps(pred, gt, extent, out_path, title_suffix=""):
    """Three safety maps: StarV prediction, camera ground truth, disagreement.

    pred/gt are (n_dis, n_vel) with 1 = safe; extent = (dis_lo, dis_hi,
    vel_lo, vel_hi) in physical units.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    dis_lo, dis_hi, vel_lo, vel_hi = extent
    imshow_kw = dict(
        origin="lower",
        extent=(vel_lo, vel_hi, dis_lo, dis_hi),
        aspect="equal",
        interpolation="nearest",
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)

    binary_cmap = ListedColormap([UNSAFE_COLOR, SAFE_COLOR])
    for ax, mat, name in (
        (axes[0], pred, "StarV reachability"),
        (axes[1], gt, "Camera ground truth"),
    ):
        ax.imshow(mat, cmap=binary_cmap, vmin=0, vmax=1, **imshow_kw)
        # texture overlay on unsafe cells: color never carries meaning alone
        ax.contourf(
            np.linspace(vel_lo, vel_hi, mat.shape[1]),
            np.linspace(dis_lo, dis_hi, mat.shape[0]),
            (mat == 0).astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["////"],
        )
        ax.set_title(name, fontsize=11)

    # disagreement map: 0 agree-safe, 1 agree-unsafe, 2 FN, 3 FP
    disagree = np.zeros_like(pred)
    disagree[(pred == 0) & (gt == 0)] = 1
    disagree[(pred == 0) & (gt == 1)] = 2
    disagree[(pred == 1) & (gt == 0)] = 3
    axes[2].imshow(
        disagree,
        cmap=ListedColormap([AGREE_SAFE, AGREE_UNSAFE, FN_COLOR, FP_COLOR]),
        vmin=0, vmax=3, **imshow_kw,
    )
    axes[2].contourf(
        np.linspace(vel_lo, vel_hi, disagree.shape[1]),
        np.linspace(dis_lo, dis_hi, disagree.shape[0]),
        (disagree == 3).astype(float),
        levels=[0.5, 1.5], colors="none", hatches=["xxxx"],
    )
    axes[2].set_title("Disagreement", fontsize=11)

    for ax in axes:
        ax.set_xlabel("initial velocity $v_0$ (m/s)", fontsize=10)
        ax.tick_params(labelsize=9, length=0)
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
    axes[0].set_ylabel("initial distance $d_0$ (m)", fontsize=10)

    axes[1].legend(
        handles=[
            Patch(facecolor=SAFE_COLOR, label="safe"),
            Patch(facecolor=UNSAFE_COLOR, hatch="////", label="unsafe"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
        frameon=False, fontsize=9,
    )
    axes[2].legend(
        handles=[
            Patch(facecolor=AGREE_SAFE, edgecolor="#c3c2b7", label="agree: safe"),
            Patch(facecolor=AGREE_UNSAFE, label="agree: unsafe"),
            Patch(facecolor=FN_COLOR, label="StarV conservative"),
            Patch(facecolor=FP_COLOR, hatch="xxxx", label="StarV unsound"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
        frameon=False, fontsize=9,
    )

    fig.suptitle(f"Brake system safety map{title_suffix}", fontsize=13)
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out_path}")


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

    out_path = Path(args.safety).with_name("gt_comparison.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
