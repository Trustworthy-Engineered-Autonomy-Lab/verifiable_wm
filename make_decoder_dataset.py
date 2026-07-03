import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYGLET_HEADLESS", "True")

import numpy as np
import torch

from model import *
from dynamic import *
from utils import (
    load_config,
    set_seed,
    resolve_device,
    sample_uniform_states,
    render_images,
    to_numpy,
)


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_controller(config, device):
    controller_config = config["controller"]
    controller_name = controller_config["name"]
    controller_cls = globals()[controller_name]
    controller_args = controller_config.get("args", {})

    controller = controller_cls(**controller_args).to(device).eval()
    controller.load_state_dict(load_state_dict(controller_config["weights"], device))

    print(f"[Load] {controller_name}={controller_config['weights']}")
    return controller


def build_initial_state_splits(config, device):
    num_train = int(config["num_train"])
    num_val = int(config["num_val"])
    num_test = int(config["num_test"])

    if "seed_pool" in config:
        num_pool = int(config["num_pool"])
        if num_train + num_val != num_pool:
            raise ValueError("num_train + num_val must equal num_pool.")

        set_seed(int(config["seed_pool"]))
        pool_states = sample_uniform_states(num_pool, config["state_space"], device)

        train_states = pool_states[:num_train].clone()
        val_states = pool_states[num_train:].clone()

        set_seed(int(config["seed_test"]))
        test_states = sample_uniform_states(num_test, config["state_space"], device)

    else:
        set_seed(int(config.get("seed_train", 0)))
        train_states = sample_uniform_states(num_train, config["state_space"], device)

        set_seed(int(config.get("seed_val", 1)))
        val_states = sample_uniform_states(num_val, config["state_space"], device)

        set_seed(int(config.get("seed_test", 2)))
        test_states = sample_uniform_states(num_test, config["state_space"], device)

    return {
        "train_states": train_states,
        "val_states": val_states,
        "test_states": test_states,
    }


def save_dataset(config, splits):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for split_name, states in splits.items():
        arrays[split_name] = to_numpy(states)

    np.savez_compressed(output_dir / "states.npz", **arrays)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return output_dir


def save_real_trajectories(config, trajectory_splits):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for split_name, split_data in trajectory_splits.items():
        arrays[f"{split_name}_traj"] = to_numpy(split_data["traj"])
        arrays[f"{split_name}_actions"] = to_numpy(split_data["actions"])

    np.savez_compressed(output_dir / "real_trajectories.npz", **arrays)


def select_decoder_states(states, config):
    indices = config.get("decoder_state_indices", None)
    if indices is None:
        return states
    return states[:, indices]


@torch.no_grad()
def rollout_real_trajectory(states0, steps, controller, dynamic, device, render_batch_size):
    num_samples, state_dim = states0.shape
    trajectories = torch.empty(num_samples, steps + 1, state_dim, device=device)
    action_steps = []

    states = states0.clone()
    trajectories[:, 0, :] = states

    for step in range(steps):
        images = render_images(dynamic, states, device, render_batch_size)
        actions = controller(images)
        states = dynamic.step(states, actions)

        action_steps.append(actions)
        trajectories[:, step + 1, :] = states

    actions = torch.stack(action_steps, dim=1)
    return trajectories, actions


def generate_dataset(config):
    device = resolve_device(config.get("device", "auto"))
    controller = load_controller(config, device)
    steps = int(config["rollout_steps"])
    render_batch_size = int(config.get("render_batch_size", 64))

    dynamic = globals()[config["dynamic"]["name"]](**config["dynamic"].get("args", {}))
    print(f"[Dynamic] {config['dynamic']['name']}")

    state_splits = build_initial_state_splits(config, device)

    dataset = {}
    trajectory_splits = {}
    for split_name in ("train_states", "val_states", "test_states"):
        states = state_splits[split_name]
        images = render_images(dynamic, states, device, render_batch_size)

        prefix = split_name.replace("_states", "")
        decoder_states = select_decoder_states(states, config)

        dataset[f"{prefix}_states"] = decoder_states
        dataset[f"{prefix}_images"] = images

        print(f"[Render] {prefix}: states={tuple(states.shape)}, images={tuple(images.shape)}")

        traj, actions = rollout_real_trajectory(
            states,
            steps,
            controller,
            dynamic,
            device,
            render_batch_size,
        )
        trajectory_splits[prefix] = {
            "traj": traj,
            "actions": actions,
        }

        print(
            f"[Trajectory] {prefix}: "
            f"traj={tuple(traj.shape)}, "
            f"actions={tuple(actions.shape)}"
        )

    output_dir = save_dataset(config, dataset)
    save_real_trajectories(config, trajectory_splits)
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = generate_dataset(config)
    print(f"[Done] decoder dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
