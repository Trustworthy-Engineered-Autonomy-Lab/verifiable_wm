#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ported from aebs_carla/gt_brake_grid_eval.py: camera closed-loop ground
# truth for the brake-system safety grid (produces judge.npy). Needs the
# CARLA server (CARLA_HOST/CARLA_PORT, default decaf:8000).
#
#   python tools/gt_brake_grid_eval.py \
#       --controller_path dwm_weight/brake_system/controller.pth \
#       --out_dir datasets/brake_system/ground_truth/gt_grid_out_0.01 \
#       --samples_per_cell 3
import os
import time
import queue
import argparse
import traceback
import sys
from pathlib import Path

import numpy as np
import gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env as _env  # noqa: F401,E402  (registers the CARLA AEBS envs)
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image


class BrakeController(nn.Module):
    def __init__(self):
        super(BrakeController, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(24 * 24, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./gt_grid_out")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--env_id", type=str,
                        default="AdvancedEmergencyBrakingSystemWithRendering-v0")
    parser.add_argument("--distance_min", type=float, default=6.00)
    parser.add_argument("--distance_max", type=float, default=6.40)
    parser.add_argument("--distance_step", type=float, default=0.01)
    parser.add_argument("--velocity_min", type=float, default=6.00)
    parser.add_argument("--velocity_max", type=float, default=6.40)
    parser.add_argument("--velocity_step", type=float, default=0.01)
    parser.add_argument("--samples_per_cell", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--distance_scale", type=float, default=60.0)
    parser.add_argument("--velocity_scale", type=float, default=30.0)
    parser.add_argument("--v_lead", type=float, default=0.0)
    parser.add_argument("--plot", action="store_true")

    # new
    parser.add_argument("--restart_every", type=int, default=200,
                        help="Save checkpoint and restart CARLA every N cells")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint if it exists")
    return parser.parse_args()


def resize_rgb_image(img: np.ndarray, image_size: int) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((image_size, image_size), Image.BILINEAR)
    return np.array(pil_img, dtype=np.uint8)


def image_to_tensor(img: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(img).float() / 255.0
    x = x.mean(dim=2, keepdim=True)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    return x


def safe_reset(env):
    out = env.reset()
    if isinstance(out, tuple):
        out = out[0]
    return out


def safe_render_rgb(env):
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "image"):
        img = env.unwrapped.image
        if img is not None:
            return np.array(img, dtype=np.uint8)
    try:
        img = env.render(mode="rgb_array")
    except TypeError:
        img = env.render()
    return np.array(img, dtype=np.uint8)


def drain_sensor_queues(env, max_drain: int = 50):
    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "_queues"):
        for q in unwrapped._queues:
            drained = 0
            while drained < max_drain:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break
    for attr in ["_collision_queue", "collision_queue"]:
        if hasattr(unwrapped, attr):
            q = getattr(unwrapped, attr)
            drained = 0
            while drained < max_drain:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break


def make_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)
    _ = safe_reset(env)
    time.sleep(2.0)
    return env


def set_env_state(env, dist_m: float, vel_val: float,
                  max_tries: int = 6, sleep_sec: float = 2.0):
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            drain_sensor_queues(env)

            if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "reset_to_state"):
                out = env.unwrapped.reset_to_state(dist_m, vel_val)
            elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "reset"):
                out = env.unwrapped.reset(state=[dist_m, vel_val])
            else:
                raise AttributeError("Cannot find reset_to_state or reset(state=...) on env.")

            if isinstance(out, tuple):
                out = out[0]
            return np.array(out, dtype=np.float32), env

        except queue.Empty as e:
            last_err = e
            wait = sleep_sec * attempt
            print(f"[Warn] queue.Empty on reset (d={dist_m:.4f}, v={vel_val:.4f}), "
                  f"try {attempt}/{max_tries}, sleeping {wait:.0f}s")
            time.sleep(wait)

        except Exception:
            raise

    print(f"[Warn] All {max_tries} retries failed. Restarting CARLA env...")
    try:
        env.close()
    except Exception:
        pass
    time.sleep(5.0)
    env = make_env(env.__spec__.id if hasattr(env, "__spec__") and env.__spec__ else
                   "AdvancedEmergencyBrakingSystemWithRendering-v0")

    drain_sensor_queues(env)
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "reset_to_state"):
        out = env.unwrapped.reset_to_state(dist_m, vel_val)
    else:
        out = env.unwrapped.reset(state=[dist_m, vel_val])
    if isinstance(out, tuple):
        out = out[0]
    return np.array(out, dtype=np.float32), env


def step_dynamics_unnorm(dist_m, vel_val, action_tanh, dt, v_lead=0.0):
    brake = float(np.clip(0.5 * (action_tanh + 1.0), 0.0, 1.0))
    decel = 0.009 * brake + 0.0042
    next_dist = dist_m + (v_lead - vel_val) * dt
    next_vel  = vel_val - decel * dt
    next_dist = max(0.0, next_dist)
    next_vel  = max(0.0, next_vel)
    return next_dist, next_vel


@torch.no_grad()
def rollout_one_sample(env,
                       controller,
                       init_dist, init_vel,
                       steps, image_size, device,
                       dt, v_lead,
                       distance_scale, velocity_scale):
    dist = float(init_dist)
    vel  = float(init_vel)

    for _ in range(steps):
        _, env = set_env_state(env, dist, vel)

        img    = safe_render_rgb(env)
        img    = resize_rgb_image(img, image_size)
        x      = image_to_tensor(img, device)
        action = controller(x).squeeze().item()

        dist, vel = step_dynamics_unnorm(dist, vel, action, dt, v_lead)

    return np.array([dist, vel], dtype=np.float32), env


def save_checkpoint(out_dir, cells_done, judge, cell_bounds, dist_edges, vel_edges):
    ckpt_path = os.path.join(out_dir, f"checkpoint_cell{cells_done}.npz")
    np.savez_compressed(
        ckpt_path,
        judge=judge,
        cell_bounds=cell_bounds,
        dist_edges=dist_edges,
        vel_edges=vel_edges,
        cells_done=np.array(cells_done, dtype=np.int32),
    )
    latest_path = os.path.join(out_dir, "checkpoint_latest.npz")
    np.savez_compressed(
        latest_path,
        judge=judge,
        cell_bounds=cell_bounds,
        dist_edges=dist_edges,
        vel_edges=vel_edges,
        cells_done=np.array(cells_done, dtype=np.int32),
    )
    print(f"[Checkpoint] saved -> {ckpt_path}")
    print(f"[Checkpoint] updated -> {latest_path}")


def load_checkpoint_if_any(args, n_vel, n_dist, dist_edges, vel_edges):
    latest_path = os.path.join(args.out_dir, "checkpoint_latest.npz")

    judge = np.ones((n_vel, n_dist), dtype=np.uint8)
    cell_bounds = np.zeros((n_vel, n_dist, 2, 2), dtype=np.float32)
    start_cell = 0

    if args.resume and os.path.exists(latest_path):
        print(f"[Resume] loading checkpoint from {latest_path}")
        ckpt = np.load(latest_path, allow_pickle=True)

        judge = ckpt["judge"]
        cell_bounds = ckpt["cell_bounds"]
        start_cell = int(ckpt["cells_done"])

        ckpt_dist_edges = ckpt["dist_edges"]
        ckpt_vel_edges = ckpt["vel_edges"]

        if not np.allclose(dist_edges, ckpt_dist_edges):
            raise ValueError("Current dist_edges do not match checkpoint.")
        if not np.allclose(vel_edges, ckpt_vel_edges):
            raise ValueError("Current vel_edges do not match checkpoint.")

        print(f"[Resume] cells_done = {start_cell}")

    return judge, cell_bounds, start_cell


def evaluate_grid(args):
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device = torch.device(
        args.device if (torch.cuda.is_available() or "cpu" in args.device) else "cpu"
    )

    controller = BrakeController().to(device)
    ckpt = torch.load(args.controller_path, map_location=device, weights_only=True)
    controller.load_state_dict(ckpt)
    controller.eval()

    dist_edges = np.arange(
        args.distance_min, args.distance_max + 1e-8, args.distance_step, dtype=np.float32
    )
    vel_edges = np.arange(
        args.velocity_min, args.velocity_max + 1e-8, args.velocity_step, dtype=np.float32
    )

    n_dist = len(dist_edges) - 1
    n_vel  = len(vel_edges)  - 1
    total_cells = n_dist * n_vel

    judge, cell_bounds, start_cell = load_checkpoint_if_any(
        args, n_vel, n_dist, dist_edges, vel_edges
    )

    env_id = args.env_id
    env = make_env(env_id)

    try:
        for j in range(n_vel):
            vel_lb = vel_edges[j]
            vel_ub = vel_edges[j + 1]

            for i in range(n_dist):
                current_cell = j * n_dist + i

                if current_cell < start_cell:
                    continue

                if current_cell > 0 and current_cell % args.restart_every == 0:
                    print(f"[Restart] reached cell {current_cell}, saving and restarting CARLA...")
                    save_checkpoint(
                        args.out_dir,
                        current_cell,
                        judge,
                        cell_bounds,
                        dist_edges,
                        vel_edges,
                    )
                    try:
                        env.close()
                    except Exception:
                        pass
                    time.sleep(5.0)
                    env = make_env(env_id)
                    print("[Restart] done.")

                dist_lb = dist_edges[i]
                dist_ub = dist_edges[i + 1]

                print(f"[Cell {current_cell + 1}/{total_cells}] "
                      f"distance:[{dist_lb:.2f}, {dist_ub:.2f}], "
                      f"velocity:[{vel_lb:.2f}, {vel_ub:.2f}]")

                finals = []
                for _ in range(args.samples_per_cell):
                    init_dist = rng.uniform(dist_lb, dist_ub)
                    init_vel  = rng.uniform(vel_lb,  vel_ub)

                    final_state, env = rollout_one_sample(
                        env=env,
                        controller=controller,
                        init_dist=init_dist,
                        init_vel=init_vel,
                        steps=args.steps,
                        image_size=args.image_size,
                        device=device,
                        dt=args.dt,
                        v_lead=args.v_lead,
                        distance_scale=args.distance_scale,
                        velocity_scale=args.velocity_scale,
                    )
                    finals.append(final_state)

                finals = np.stack(finals, axis=0)

                final_dist_lb = finals[:, 0].min()
                final_dist_ub = finals[:, 0].max()
                final_vel_lb  = finals[:, 1].min()
                final_vel_ub  = finals[:, 1].max()

                cell_bounds[j, i, 0, 0] = final_dist_lb
                cell_bounds[j, i, 0, 1] = final_dist_ub
                cell_bounds[j, i, 1, 0] = final_vel_lb
                cell_bounds[j, i, 1, 1] = final_vel_ub

                is_safe = (final_dist_lb > 0.0)
                judge[j, i] = 0 if is_safe else 1

                print(f"    final distance bound = [{final_dist_lb:.6f}, {final_dist_ub:.6f}], "
                      f"final velocity bound = [{final_vel_lb:.6f}, {final_vel_ub:.6f}], "
                      f"judge = {'SAFE' if is_safe else 'UNSAFE'}")

                start_cell = current_cell + 1

    finally:
        try:
            env.close()
        except Exception:
            pass

    save_checkpoint(
        args.out_dir,
        start_cell,
        judge,
        cell_bounds,
        dist_edges,
        vel_edges,
    )

    np.save(os.path.join(args.out_dir, "judge.npy"),       judge)
    np.save(os.path.join(args.out_dir, "cell_bounds.npy"), cell_bounds)
    np.save(os.path.join(args.out_dir, "dist_edges.npy"),  dist_edges)
    np.save(os.path.join(args.out_dir, "vel_edges.npy"),   vel_edges)

    print(f"[Saved] judge      -> {os.path.join(args.out_dir, 'judge.npy')}")
    print(f"[Saved] bounds     -> {os.path.join(args.out_dir, 'cell_bounds.npy')}")
    print(f"[Saved] dist_edges -> {os.path.join(args.out_dir, 'dist_edges.npy')}")
    print(f"[Saved] vel_edges  -> {os.path.join(args.out_dir, 'vel_edges.npy')}")

    if args.plot:
        cmap = ListedColormap(["green", "red"])
        fig, ax = plt.subplots(
            figsize=(min(12, 0.5 * n_dist + 2), min(8, 0.5 * n_vel + 2))
        )
        ax.imshow(judge, cmap=cmap, origin="lower",
                  extent=[dist_edges[0], dist_edges[-1], vel_edges[0], vel_edges[-1]],
                  aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax.set_xlabel("distance (m)")
        ax.set_ylabel("velocity (m/s)")
        ax.set_title(f"Brake system GT safety map ({n_dist}×{n_vel})\ngreen=safe, red=unsafe")
        plt.tight_layout()
        plot_path = os.path.join(args.out_dir, "grid_judge_cam.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[Saved] plot -> {plot_path}")

    return judge, cell_bounds, dist_edges, vel_edges


def main():
    args = parse_args()
    try:
        evaluate_grid(args)
    except Exception as e:
        print(f"\n[ERROR] {repr(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()