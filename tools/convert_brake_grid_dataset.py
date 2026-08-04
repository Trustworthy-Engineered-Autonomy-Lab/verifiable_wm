#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flatten the braking grid capture into state->image and image->action sets.

Ported from aebs_carla/convert_grid_dataset.py. collect_brake_grid_dataset.py
stores one rollout per grid cell, shaped (cells, steps, ...); training wants a
flat sample list. Steps that were never executed (the rollout stopped at a
collision) carry NaN actions and are dropped here.

    python tools/convert_brake_grid_dataset.py \\
        --input_npz datasets/brake_system/grid_dataset/grid_dataset.npz \\
        --decoder_out datasets/brake_system/dataset_decoder.npz

The states stay in physical units (metres, m/s), matching the grid that
config/starv_verification/brake_system.json verifies over -- they are not
rescaled by the DDPG policy's normalisation. tools/make_brake_decoder_dataset.py
then splits the decoder file into the repo's train/val/test format.
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_npz", type=Path,
        default=Path("datasets/brake_system/grid_dataset/grid_dataset.npz"),
    )
    parser.add_argument(
        "--decoder_out", type=Path,
        default=Path("datasets/brake_system/dataset_decoder.npz"),
    )
    parser.add_argument(
        "--controller_out", type=Path,
        default=Path("datasets/brake_system/dataset_controller.npz"),
    )
    parser.add_argument(
        "--use_init_only", action="store_true",
        help="Keep only t=0 of every cell instead of the whole rollout.",
    )
    parser.add_argument(
        "--drop_zero_images", action="store_true",
        help="Drop samples whose frame is entirely black (a dropped capture).",
    )
    return parser.parse_args()


def load_grid_capture(path: Path):
    with np.load(path, allow_pickle=False) as data:
        missing = {"states", "images", "actions"}.difference(data.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        states = np.asarray(data["states"])     # (cells, steps + 1, 2)
        images = np.asarray(data["images"])     # (cells, steps, H, W)
        actions = np.asarray(data["actions"])   # (cells, steps)

    if states.ndim != 3 or states.shape[2] != 2:
        raise ValueError(f"expected states (cells, steps+1, 2), got {states.shape}")
    if images.ndim != 4:
        raise ValueError(f"expected images (cells, steps, H, W), got {images.shape}")
    if actions.ndim != 2:
        raise ValueError(f"expected actions (cells, steps), got {actions.shape}")
    if images.shape[0] != states.shape[0] or images.shape[1] != actions.shape[1]:
        raise ValueError(
            "states/images/actions disagree: "
            f"states={states.shape}, images={images.shape}, actions={actions.shape}"
        )
    return states, images, actions


def flatten(states, images, actions, *, use_init_only, drop_zero_images):
    n_cells, n_steps = actions.shape
    sample_states, sample_images, sample_actions = [], [], []
    seen = 0

    for cell in range(n_cells):
        for t in ([0] if use_init_only else range(n_steps)):
            seen += 1
            # A step that never ran leaves its action NaN.
            if not np.isfinite(actions[cell, t]):
                continue
            if not np.all(np.isfinite(states[cell, t])):
                continue
            if drop_zero_images and not np.any(images[cell, t]):
                continue

            sample_states.append(states[cell, t].astype(np.float32))
            sample_images.append(images[cell, t].astype(np.uint8))
            sample_actions.append(np.array([actions[cell, t]], dtype=np.float32))

    if not sample_states:
        raise RuntimeError(f"no valid samples in the capture ({seen} slots inspected)")

    flat_states = np.stack(sample_states).astype(np.float32)
    # (N,H,W) -> (N,H,W,1), the layout every other benchmark's capture uses.
    flat_images = np.stack(sample_images)[..., None].astype(np.uint8)
    flat_actions = np.stack(sample_actions).astype(np.float32)
    return flat_states, flat_images, flat_actions, seen


def main():
    args = parse_args()
    print(f"[Load] {args.input_npz}")
    states, images, actions = load_grid_capture(args.input_npz)
    print(f"[Info] states={states.shape} images={images.shape} actions={actions.shape}")

    flat_states, flat_images, flat_actions, seen = flatten(
        states, images, actions,
        use_init_only=args.use_init_only,
        drop_zero_images=args.drop_zero_images,
    )

    for path in (args.decoder_out, args.controller_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.decoder_out,
        states=flat_states,                 # (N, 2) metres and m/s
        images=flat_images,                 # (N, 96, 96, 1) uint8
        distance_m=flat_states[:, 0],
        velocity=flat_states[:, 1],
    )
    np.savez_compressed(
        args.controller_out,
        images=flat_images,
        actions=flat_actions,               # (N, 1)
    )

    print(f"[Saved] {args.decoder_out}  states={flat_states.shape} images={flat_images.shape}")
    print(f"[Saved] {args.controller_out}")
    print(f"[Info] kept {len(flat_states)} / {seen} candidate step slots")


if __name__ == "__main__":
    main()
