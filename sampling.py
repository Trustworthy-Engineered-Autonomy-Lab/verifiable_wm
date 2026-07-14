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


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS = tuple(
    PROJECT_ROOT / "config" / "sampling" / f"{environment}.json"
    for environment in ("cartpole", "mountain_car", "pendulum")
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


def load_decoder(config, device):
    decoder_config = config["decoder"]
    decoder_name = decoder_config["name"]
    decoder_cls = globals()[decoder_name]
    decoder_args = decoder_config.get("args", {})

    decoder = decoder_cls(**decoder_args).to(device).eval()
    decoder.load_state_dict(load_state_dict(decoder_config["weights"], device))

    print(f"[Load] {decoder_name}={decoder_config['weights']}")
    return decoder


def build_initial_state_splits(config, device):
    set_seed(int(config["seed_train"]))
    train_states = sample_uniform_states(int(config["num_train"]), config["state_space"], device)

    set_seed(int(config["seed_val"]))
    val_states = sample_uniform_states(int(config["num_val"]), config["state_space"], device)

    set_seed(int(config["seed_test"]))
    test_states = sample_uniform_states(int(config["num_test"]), config["state_space"], device)

    return {
        "train": train_states,
        "val": val_states,
        "test": test_states,
    }


@torch.no_grad()
def rollout_transition(states0, steps, controller, dynamic, device, render_batch_size):
    states = states0.clone()

    all_states = []
    all_actions = []
    all_next_states = []

    for step in range(steps):
        images = render_images(dynamic, states, device, render_batch_size)
        actions = controller(images)
        next_states = dynamic.step(states, actions)

        all_states.append(states)
        all_actions.append(actions)
        all_next_states.append(next_states)

        states = next_states

    states = torch.cat(all_states, dim=0)
    actions = torch.cat(all_actions, dim=0)
    next_states = torch.cat(all_next_states, dim=0)

    return states, actions, next_states


def save_dataset(config, dataset):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for split_name, split_data in dataset.items():
        arrays[f"{split_name}_states"] = to_numpy(split_data["states"])
        arrays[f"{split_name}_actions"] = to_numpy(split_data["actions"])
        arrays[f"{split_name}_next_states"] = to_numpy(split_data["next_states"])

    np.savez_compressed(output_dir / "transition_dataset.npz", **arrays)

    return output_dir


def save_metadata(config):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return output_dir


def save_dwm_trajectories(config, trajectory_splits):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for split_name, split_data in trajectory_splits.items():
        arrays[f"{split_name}_traj"] = to_numpy(split_data["traj"])
        arrays[f"{split_name}_actions"] = to_numpy(split_data["actions"])

    np.savez_compressed(output_dir / "dwm_trajectories.npz", **arrays)


@torch.no_grad()
def rollout_dwm_trajectory(
    states0,
    steps,
    decoder,
    controller,
    dynamic,
    device,
    decoder_state_indices=None,
):
    num_samples, state_dim = states0.shape
    trajectories = torch.empty(num_samples, steps + 1, state_dim, device=device)
    action_steps = []

    states = states0.clone()
    trajectories[:, 0, :] = states

    for step in range(steps):
        decoder_states = states
        if decoder_state_indices is not None:
            decoder_states = states[:, decoder_state_indices]

        images = decoder(decoder_states)
        actions = controller(images)
        states = dynamic.step(states, actions)

        action_steps.append(actions)
        trajectories[:, step + 1, :] = states

    actions = torch.stack(action_steps, dim=1)
    return trajectories, actions


def generate_dataset(config):
    device = resolve_device(config.get("device", "auto"))
    controller = load_controller(config, device)
    decoder = load_decoder(config, device)

    steps = int(config["rollout_steps"])
    render_batch_size = int(config.get("render_batch_size", 64))
    generate_transition_dataset = bool(config.get("generate_transition_dataset", False))
    should_save_metadata = bool(config.get("save_metadata", False))
    decoder_state_indices = config.get(
        "decoder_state_indices",
        config["decoder"].get("state_indices"),
    )

    dynamic = globals()[config["dynamic"]["name"]](**config["dynamic"].get("args", {}))
    print(f"[Dynamic] {config['dynamic']['name']}")

    initial_splits = build_initial_state_splits(config, device)

    dataset = {}
    trajectory_splits = {}
    if not generate_transition_dataset:
        print("[Skip] transition dataset generation is disabled")

    for split_name, states0 in initial_splits.items():
        if generate_transition_dataset:
            states, actions, next_states = rollout_transition(
                states0,
                steps,
                controller,
                dynamic,
                device,
                render_batch_size,
            )

            dataset[split_name] = {
                "states": states,
                "actions": actions,
                "next_states": next_states,
            }

            print(
                f"[Rollout] {split_name}: "
                f"states={tuple(states.shape)}, "
                f"actions={tuple(actions.shape)}, "
                f"next_states={tuple(next_states.shape)}"
            )

        traj, dwm_actions = rollout_dwm_trajectory(
            states0,
            steps,
            decoder,
            controller,
            dynamic,
            device,
            decoder_state_indices,
        )
        trajectory_splits[split_name] = {
            "traj": traj,
            "actions": dwm_actions,
        }

        print(
            f"[DWM Trajectory] {split_name}: "
            f"traj={tuple(traj.shape)}, "
            f"actions={tuple(dwm_actions.shape)}"
        )

    output_dir = Path(config["output_dir"])
    if generate_transition_dataset:
        output_dir = save_dataset(config, dataset)

    save_dwm_trajectories(config, trajectory_splits)

    if should_save_metadata:
        save_metadata(config)
    else:
        print("[Skip] metadata generation is disabled")

    return output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        help="Config JSON files. If omitted, all local make_dataset configs are used.",
    )
    return parser.parse_args()


def resolve_config_path(path):
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def main():
    args = parse_args()
    config_paths = args.configs or DEFAULT_CONFIGS

    for config_path in config_paths:
        config_path = resolve_config_path(config_path)
        print(f"[Config] {config_path}")
        config = load_config(config_path)
        output_dir = generate_dataset(config)
        print(f"[Done] sampling outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
