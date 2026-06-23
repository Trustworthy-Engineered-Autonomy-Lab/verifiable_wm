import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYGLET_HEADLESS", "True")

import numpy as np
import torch
import torch.nn.functional as F

from model import *
from dynamic import *


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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


def sample_uniform_states(num_samples, state_space, device):
    columns = []
    for dim in state_space:
        low = float(dim["low"])
        high = float(dim["high"])
        columns.append(low + (high - low) * torch.rand(num_samples, 1, device=device))
    return torch.cat(columns, dim=1).float()


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


def rgb_to_gray_01(frames_rgb):
    frames = frames_rgb.astype(np.float32)
    gray = 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
    return gray / 255.0


@torch.no_grad()
def render_images(dynamic, states, device, render_batch_size):
    states_np = states.detach().cpu().numpy()
    chunks = []

    for start in range(0, states_np.shape[0], render_batch_size):
        frames = []
        for state in states_np[start:start + render_batch_size]:
            frames.append(dynamic.render(state))

        gray = rgb_to_gray_01(np.stack(frames, axis=0))
        images = torch.from_numpy(gray[:, None, :, :]).to(device)
        images = F.interpolate(images, size=(96, 96), mode="bilinear", align_corners=False)
        chunks.append(images.clamp(0.0, 1.0))

    return torch.cat(chunks, dim=0)


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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32)


def save_dataset(config, dataset):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for split_name, split_data in dataset.items():
        arrays[f"{split_name}_states"] = to_numpy(split_data["states"])
        arrays[f"{split_name}_actions"] = to_numpy(split_data["actions"])
        arrays[f"{split_name}_next_states"] = to_numpy(split_data["next_states"])

    np.savez_compressed(output_dir / "transition_dataset.npz", **arrays)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return output_dir


def generate_dataset(config):
    device = resolve_device(config.get("device", "auto"))
    controller = load_controller(config, device)

    steps = int(config["rollout_steps"])
    render_batch_size = int(config.get("render_batch_size", 64))

    dynamic = globals()[config["dynamic"]["name"]](**config["dynamic"].get("args", {}))
    print(f"[Dynamic] {config['dynamic']['name']}")

    initial_splits = build_initial_state_splits(config, device)

    dataset = {}
    for split_name, states0 in initial_splits.items():
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

    return save_dataset(config, dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = generate_dataset(config)
    print(f"[Done] transition dataset saved to {output_dir}")


if __name__ == "__main__":
    main()