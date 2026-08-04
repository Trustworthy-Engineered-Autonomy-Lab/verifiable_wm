#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture the braking benchmark's CARLA frames over the verification grid.

Ported from aebs_carla/collect_grid_dataset.py, the first step of the only
benchmark whose images do not come from a gym renderer. One initial state is
drawn per grid cell, the DDPG policy drives the car for the verification
horizon, and every visited state is stored together with the frame the camera
saw there.

    python tools/collect_brake_grid_dataset.py \\
        --controller_path logs/best_model.zip \\
        --out_dir datasets/brake_system/grid_dataset

The resulting grid_dataset.npz is turned into the repo's decoder dataset by
tools/convert_brake_grid_dataset.py followed by
tools/make_brake_decoder_dataset.py.

Needs a CARLA server (CARLA_HOST/CARLA_PORT) and stable-baselines3 with the
pretrained DDPG policy; neither is distributed with the repository.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import carla_frame_to_gray  # noqa: E402
from tools.carla_aebs_client import (  # noqa: E402
    ENV_ID,
    make_env,
    safe_render_rgb,
    set_env_state,
)

# The DDPG policy was trained on normalized observations; these are the
# scales it expects, not a property of the dynamics.
DISTANCE_SCALE = 60.0
VELOCITY_SCALE = 30.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller_path", type=str, default="logs/best_model.zip",
                        help="Pretrained stable-baselines3 DDPG policy")
    parser.add_argument("--out_dir", type=str, default="datasets/brake_system/grid_dataset")
    parser.add_argument("--out_npz", type=str, default="grid_dataset.npz")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--env_id", type=str, default=ENV_ID)
    # Defaults describe the 40 x 40 grid in config/starv_verification/brake_system.json.
    parser.add_argument("--distance_min", type=float, default=6.00)
    parser.add_argument("--distance_max", type=float, default=6.40)
    parser.add_argument("--distance_step", type=float, default=0.01)
    parser.add_argument("--velocity_min", type=float, default=6.00)
    parser.add_argument("--velocity_max", type=float, default=6.40)
    parser.add_argument("--velocity_step", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--v_lead", type=float, default=0.0)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--restart_every", type=int, default=400,
                        help="Checkpoint and restart CARLA every N cells")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from checkpoint_latest.npz if it exists")
    return parser.parse_args()


def step_ddpg_dynamics(dist_m, vel_val, action, dt, v_lead=0.0):
    """Advance the state under the DDPG policy's action convention.

    Deliberately not dynamic.Brake.step: the DDPG policy emits the brake
    command directly in [0, 1], whereas the vision closed loop that gets
    verified feeds a sigmoid output through brake = 0.5 * (a + 1). Only the
    coverage of the captured states depends on this, since the dataset is
    consumed as a state -> image mapping, but the two are not interchangeable.
    """
    brake = float(np.clip(action, 0.0, 1.0))
    decel = 0.009 * brake + 0.0042
    next_dist = max(0.0, dist_m + (v_lead - vel_val) * dt)
    next_vel = max(0.0, vel_val - decel * dt)
    return next_dist, next_vel


def empty_arrays(n_cells, n_steps, image_size):
    return {
        "states": np.full((n_cells, n_steps + 1, 2), np.nan, dtype=np.float32),
        "images": np.zeros((n_cells, n_steps, image_size, image_size), dtype=np.uint8),
        "actions": np.full((n_cells, n_steps), np.nan, dtype=np.float32),
        "collision": np.zeros(n_cells, dtype=bool),
        "init_states": np.full((n_cells, 2), np.nan, dtype=np.float32),
    }


def load_or_init(out_dir, resume, n_cells, n_steps, image_size):
    checkpoint = Path(out_dir) / "checkpoint_latest.npz"
    if resume and checkpoint.is_file():
        print(f"[Resume] loading {checkpoint}")
        with np.load(checkpoint, allow_pickle=False) as saved:
            arrays = {key: np.asarray(saved[key]).copy() for key in
                      ("states", "images", "actions", "collision", "init_states")}
            start_cell = int(saved["cells_done"])
        print(f"[Resume] cells_done = {start_cell}")
        return arrays, start_cell

    print("[Resume] no checkpoint, starting from scratch")
    return empty_arrays(n_cells, n_steps, image_size), 0


def save_npz(path, arrays, dist_edges, vel_edges, cells_done=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload["dist_edges"] = dist_edges
    payload["vel_edges"] = vel_edges
    if cells_done is not None:
        payload["cells_done"] = np.array(int(cells_done))
    np.savez_compressed(path, **payload)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    dist_edges = np.arange(
        args.distance_min, args.distance_max + 1e-9, args.distance_step, dtype=np.float32
    )
    vel_edges = np.arange(
        args.velocity_min, args.velocity_max + 1e-9, args.velocity_step, dtype=np.float32
    )
    n_dist = len(dist_edges) - 1
    n_vel = len(vel_edges) - 1
    n_cells = n_dist * n_vel
    print(f"Grid: {n_dist} dist x {n_vel} vel = {n_cells} cells, horizon {args.steps}")

    arrays, start_cell = load_or_init(
        args.out_dir, args.resume, n_cells, args.steps, args.image_size
    )

    from stable_baselines3 import DDPG

    print(f"Loading DDPG from {args.controller_path} ...")
    policy = DDPG.load(args.controller_path, device=args.device)

    env = make_env(args.env_id)
    checkpoint_path = Path(args.out_dir) / "checkpoint_latest.npz"

    try:
        for j in range(n_vel):
            vel_lb, vel_ub = vel_edges[j], vel_edges[j + 1]

            for i in range(n_dist):
                cell = j * n_dist + i
                if cell < start_cell:
                    continue

                if cell > 0 and cell % args.restart_every == 0:
                    save_npz(checkpoint_path, arrays, dist_edges, vel_edges, cells_done=cell)
                    print(f"[Checkpoint] cell {cell} -> {checkpoint_path}; restarting CARLA")
                    try:
                        env.close()
                    except Exception:
                        pass
                    time.sleep(5.0)
                    env = make_env(args.env_id)

                dist_lb, dist_ub = dist_edges[i], dist_edges[i + 1]
                dist = float(rng.uniform(dist_lb, dist_ub))
                vel = float(rng.uniform(vel_lb, vel_ub))

                arrays["init_states"][cell] = (dist, vel)
                arrays["states"][cell, 0] = (dist, vel)
                print(f"[Cell {cell + 1:4d}/{n_cells}]  "
                      f"dist in [{dist_lb:.2f},{dist_ub:.2f}]  "
                      f"vel in [{vel_lb:.2f},{vel_ub:.2f}]  "
                      f"init=({dist:.4f}, {vel:.4f})")

                collision = False
                for t in range(args.steps):
                    _, env = set_env_state(env, dist, vel, env_id=args.env_id)

                    gray = carla_frame_to_gray(safe_render_rgb(env), args.image_size)
                    arrays["images"][cell, t] = np.rint(gray * 255.0).astype(np.uint8)

                    observation = np.array(
                        [dist / DISTANCE_SCALE, vel / VELOCITY_SCALE], dtype=np.float32
                    )
                    action_arr, _ = policy.predict(observation, deterministic=True)
                    action = float(np.clip(action_arr[0], 0.0, 1.0))
                    arrays["actions"][cell, t] = action

                    dist, vel = step_ddpg_dynamics(dist, vel, action, args.dt, args.v_lead)
                    arrays["states"][cell, t + 1] = (dist, vel)

                    if dist <= 0.0:
                        collision = True
                        print(f"    collision at step {t + 1}")
                        break

                arrays["collision"][cell] = collision
                start_cell = cell + 1
                save_npz(checkpoint_path, arrays, dist_edges, vel_edges, cells_done=start_cell)

    finally:
        try:
            env.close()
        except Exception:
            pass

    output_path = Path(args.out_dir) / args.out_npz
    save_npz(output_path, arrays, dist_edges, vel_edges)
    n_collision = int(arrays["collision"].sum())
    print(f"\n[Saved] {output_path}")
    print(f"  states  : {arrays['states'].shape}")
    print(f"  images  : {arrays['images'].shape}")
    print(f"  safe    : {n_cells - n_collision} / {n_cells}")
    print(f"  collision: {n_collision} / {n_cells}")


if __name__ == "__main__":
    main()
