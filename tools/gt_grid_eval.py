"""Reachability ground truth via dense real closed-loop rollouts.

Generalizes aebs_carla/gt_brake_grid_eval.py's method (paper Sec. 3.3.1) to
the three gym case studies. For every grid cell: sample K initial states
uniformly inside the cell, run the TRUE closed-loop system (real render ->
controller -> real dynamics, no decoder involved) for the verifier's
num_steps, and label the cell successful iff every sampled trajectory
reaches the goal (matching each *Verifier.criteria in
starv_verification/verifiers.py).

    python -m tools.gt_grid_eval \
        --env cartpole \
        --starv-config config/starv_verification/cartpole.json \
        --sampling-config config/sampling/cartpole.json \
        --samples-per-cell 100 \
        --out datasets/cartpole/ground_truth/gt_grid.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import dynamic as dynamic_module
from sampling import load_controller
from utils import load_config, render_images, resolve_device, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# state index (after a real rollout) + comparison used by each *Verifier in
# starv_verification/verifiers.py -- kept in lockstep with those criteria.
SUCCESS_CRITERIA = {
    "cartpole": lambda final, cfg: final[:, 2].abs() <= cfg["goal_angle_threshold"],
    "mountain_car": lambda final, cfg: final[:, 0] >= cfg["goal_position_threshold"],
    "pendulum": lambda final, cfg: final[:, 0].abs() <= cfg["goal_angle_threshold"],
}

DEFAULT_GOAL_KWARGS = {
    "cartpole": "goal_angle_threshold",
    "mountain_car": "goal_position_threshold",
    "pendulum": "goal_angle_threshold",
}


def generate_grid_cells(grid):
    starts, ends = [], []
    for dim in grid["dims"]:
        vals = np.linspace(dim["start"], dim["stop"], dim["num"] + 1)
        starts.append(vals[:-1])
        ends.append(vals[1:])

    starts_mesh = np.array(np.meshgrid(*starts, indexing="ij"))
    ends_mesh = np.array(np.meshgrid(*ends, indexing="ij"))
    shape = starts_mesh.shape[1:]

    cell_starts = starts_mesh.reshape(starts_mesh.shape[0], -1).T
    cell_ends = ends_mesh.reshape(ends_mesh.shape[0], -1).T
    return cell_starts, cell_ends, shape


@torch.no_grad()
def rollout_real(dynamic, controller, states0, num_steps, device, render_batch_size):
    states = states0
    for _ in range(num_steps):
        images = render_images(dynamic, states, device, render_batch_size)
        actions = controller(images)
        states = dynamic.step(states, actions)
    return states


def evaluate_grid(env_name, starv_config, sampling_config, samples_per_cell, seed, device_name):
    device = resolve_device(device_name)
    set_seed(seed)
    rng = np.random.default_rng(seed)

    controller = load_controller(sampling_config, device)
    dynamic = getattr(dynamic_module, sampling_config["dynamic"]["name"])(
        **sampling_config["dynamic"].get("args", {})
    )

    verifier_cfg = starv_config["verifier"]["kwargs"]
    num_steps = verifier_cfg["num_steps"]
    goal_kwarg = DEFAULT_GOAL_KWARGS[env_name]
    threshold_cfg = {goal_kwarg: verifier_cfg[goal_kwarg]}
    success_fn = SUCCESS_CRITERIA[env_name]

    cell_starts, cell_ends, shape = generate_grid_cells(starv_config["grid"])
    n_cells, state_dim = cell_starts.shape

    judge = np.zeros(n_cells, dtype=np.uint8)  # 1 = safe/success
    final_bounds = np.zeros((n_cells, state_dim, 2), dtype=np.float32)

    for c in range(n_cells):
        lo, hi = cell_starts[c], cell_ends[c]
        samples = lo + (hi - lo) * rng.random((samples_per_cell, state_dim))
        states0 = torch.from_numpy(samples).float().to(device)

        final_states = rollout_real(dynamic, controller, states0, num_steps, device, render_batch_size=64)
        success = success_fn(final_states, threshold_cfg).cpu().numpy()

        judge[c] = 1 if bool(success.all()) else 0
        final_np = final_states.cpu().numpy()
        final_bounds[c, :, 0] = final_np.min(axis=0)
        final_bounds[c, :, 1] = final_np.max(axis=0)

        if (c + 1) % max(1, n_cells // 20) == 0 or c == n_cells - 1:
            print(f"[{env_name}] cell {c + 1}/{n_cells} judge={'SAFE' if judge[c] else 'UNSAFE'}")

    return {
        "judge": judge.reshape(shape),
        "cell_starts": cell_starts.reshape(*shape, state_dim),
        "cell_ends": cell_ends.reshape(*shape, state_dim),
        "final_bounds": final_bounds.reshape(*shape, state_dim, 2),
        "dims": [dim["name"] for dim in starv_config["grid"]["dims"]],
        "num_steps": num_steps,
        "samples_per_cell": samples_per_cell,
        "seed": seed,
        **threshold_cfg,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=["cartpole", "mountain_car", "pendulum"])
    parser.add_argument("--starv-config", type=Path, required=True)
    parser.add_argument("--sampling-config", type=Path, required=True)
    parser.add_argument("--samples-per-cell", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    starv_config = load_config(args.starv_config)
    sampling_config = load_config(args.sampling_config)

    result = evaluate_grid(
        args.env, starv_config, sampling_config,
        args.samples_per_cell, args.seed, args.device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **result)
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
