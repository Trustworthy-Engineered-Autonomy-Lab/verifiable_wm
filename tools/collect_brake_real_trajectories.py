#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect real (CARLA camera) closed-loop trajectories for the brake system.

The real system is: CARLA render -> grayscale 96x96 -> Controller(sigmoid)
-> analytic brake dynamics (identical to dynamic.Brake.step). CARLA is only
the camera; the state itself advances analytically, matching
aebs_carla/gt_brake_grid_eval.py.

The CARLA server (Docker) runs on the decaf machine; the client side (this
script) connects over the network, so it can run directly on this machine.
Override the target with CARLA_HOST / CARLA_PORT if the server moves.

    python tools/collect_brake_real_trajectories.py --splits test
    python tools/collect_brake_real_trajectories.py --splits train val test

Afterwards run:

    python conformal.py --env brake_system --variant old --alpha 0.05
"""

import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import Controller  # noqa: E402  (repo controller == BrakeController)
from utils import load_config, load_state_splits  # noqa: E402

ENV_ID = "AdvancedEmergencyBrakingSystemWithRendering-v0"
STARV_CONFIG = "config/starv_verification/brake_system.json"
CONTROLLER_WEIGHTS = "dwm_weight/now_weight/brake_system/controller.pth"
ROLLOUT_STEPS = 10
DT = 0.1
V_LEAD = 0.0
IMAGE_SIZE = 96


def step_dynamics(dist, vel, action):
    # Must stay identical to dynamic.Brake.step (the WM rollout dynamics).
    brake = float(np.clip(0.5 * (action + 1.0), 0.0, 1.0))
    decel = 0.009 * brake + 0.0042
    next_dist = max(0.0, dist + (V_LEAD - vel) * DT)
    next_vel = max(0.0, vel - decel * DT)
    return next_dist, next_vel


def rgb_to_gray_tensor(img_rgb, device):
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_rgb).convert("L")
    pil = pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    gray = np.array(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(gray[None, None]).to(device)


def make_env():
    # Instantiate the raw env instead of gym.make: the wrapper chain of
    # newer gym assumes the (obs, info) reset API, which this CARLA env
    # (old gym style) does not follow. We only use reset_to_state and the
    # camera frame, so no wrapper behavior is lost.
    import env as _env

    env = _env.CarlaAEBSEnv(disable_rendering=False, image_observation=False)
    env.reset()
    time.sleep(2.0)
    return env


def drain_sensor_queues(env, max_drain=50):
    unwrapped = getattr(env, "unwrapped", env)
    queues = list(getattr(unwrapped, "_queues", []))
    for attr in ("_collision_queue", "collision_queue"):
        if hasattr(unwrapped, attr):
            queues.append(getattr(unwrapped, attr))
    for q in queues:
        for _ in range(max_drain):
            try:
                q.get_nowait()
            except Exception:
                break


def reset_to_state(env, dist, vel, max_tries=6, sleep_sec=2.0):
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            drain_sensor_queues(env)
            env.unwrapped.reset_to_state(float(dist), float(vel))
            return env
        except queue.Empty as err:
            last_err = err
            print(
                f"[Warn] queue.Empty on reset (d={dist:.4f}, v={vel:.4f}), "
                f"try {attempt}/{max_tries}"
            )
            time.sleep(sleep_sec * attempt)

    print(f"[Warn] all retries failed ({last_err!r}); restarting CARLA env")
    try:
        env.close()
    except Exception:
        pass
    time.sleep(5.0)
    env = make_env()
    drain_sensor_queues(env)
    env.unwrapped.reset_to_state(float(dist), float(vel))
    return env


def current_frame(env):
    img = getattr(env.unwrapped, "image", None)
    if img is None:
        img = env.render(mode="rgb_array")
    return np.array(img, dtype=np.uint8)


@torch.no_grad()
def rollout_camera(env, controller, init_state, device):
    dist, vel = float(init_state[0]), float(init_state[1])
    traj = np.empty((ROLLOUT_STEPS + 1, 2), dtype=np.float32)
    actions = np.empty((ROLLOUT_STEPS, 1), dtype=np.float32)
    traj[0] = (dist, vel)

    for t in range(ROLLOUT_STEPS):
        env = reset_to_state(env, dist, vel)
        image = rgb_to_gray_tensor(current_frame(env), device)
        action = float(controller(image).squeeze().item())
        actions[t, 0] = action
        dist, vel = step_dynamics(dist, vel, action)
        traj[t + 1] = (dist, vel)

    return env, traj, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits", nargs="+", default=["test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    starv_config = load_config(PROJECT_ROOT / STARV_CONFIG)
    states_path = PROJECT_ROOT / starv_config["starv_states"]["output_file"]
    output_dir = states_path.parent
    splits = load_state_splits(states_path, device)

    controller = Controller(activation="sigmoid").to(device).eval()
    controller.load_state_dict(
        torch.load(
            PROJECT_ROOT / CONTROLLER_WEIGHTS,
            map_location=device,
            weights_only=True,
        )
    )
    print(f"[Load] controller={CONTROLLER_WEIGHTS}")
    print(f"[Load] starv states={states_path}")

    env = make_env()
    arrays = {
        "rollout_steps": np.array(ROLLOUT_STEPS),
        "starv_config": np.array(STARV_CONFIG),
        "controller_weights": np.array(CONTROLLER_WEIGHTS),
    }

    try:
        for split in args.splits:
            states0 = splits[f"{split}_states"].cpu().numpy()
            n = states0.shape[0]
            traj = np.empty((n, ROLLOUT_STEPS + 1, 2), dtype=np.float32)
            actions = np.empty((n, ROLLOUT_STEPS, 1), dtype=np.float32)

            start = time.time()
            for i in range(n):
                env, traj[i], actions[i] = rollout_camera(
                    env, controller, states0[i], device
                )
                if (i + 1) % 10 == 0 or i + 1 == n:
                    elapsed = time.time() - start
                    print(
                        f"[{split}] {i + 1}/{n} "
                        f"({elapsed / (i + 1):.1f}s per rollout)"
                    )

            arrays[f"{split}_traj"] = traj
            arrays[f"{split}_actions"] = actions

            # Save after every finished split so partial progress survives
            output_path = output_dir / "real_trajectories.npz"
            np.savez_compressed(output_path, **arrays)
            print(f"[Saved] {output_path} (splits: "
                  f"{[k[:-5] for k in arrays if k.endswith('_traj')]})")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
