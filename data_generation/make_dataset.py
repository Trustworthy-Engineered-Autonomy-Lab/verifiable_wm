import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYGLET_HEADLESS", "True")

import gym
import numpy as np
import torch
import torch
import torch.nn.functional as F

from simulation.model import *
from simulation.dynamic import *  


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_models(config, device):
    # Parse model configs
    decoder_config = config["decoder"]
    controller_config = config["controller"]
    decoder_name = decoder_config["name"]
    controller_name = controller_config["name"]
    decoder_cls = globals().get(decoder_name)
    controller_cls = globals().get(controller_name)
    decoder_args = decoder_config.get("args", {})
    controller_args = controller_config.get("args", {})
    # Create model instances
    decoder = decoder_cls(**decoder_args).to(device).eval()
    controller = controller_cls(**controller_args).to(device).eval()
    # Load weights
    decoder.load_state_dict(load_state_dict(decoder_config["weights"], device))
    controller.load_state_dict(load_state_dict(controller_config["weights"], device))
    print(f"[Load] {decoder_name}={decoder_config['weights']}")
    print(f"[Load] {controller_name}={controller_config['weights']}")
    return decoder, controller


def sample_uniform_states(num_samples, state_space, device):
    columns = []
    for dim in state_space:
        low = float(dim["low"])
        high = float(dim["high"])
        columns.append(low + (high - low) * torch.rand(num_samples, 1, device=device))
    return torch.cat(columns, dim=1).float()


def build_initial_state_splits(config, device):
    num_pool = int(config["num_pool"])
    num_train = int(config["num_train"])
    num_val = int(config["num_val"])
    num_test = int(config["num_test"])
    if num_train + num_val != num_pool:
        raise ValueError("num_train + num_val must equal num_pool.")

    set_seed(int(config["seed_pool"]))
    pool_states = sample_uniform_states(num_pool, config["state_space"], device)
    set_seed(int(config["seed_test"]))
    test_states = sample_uniform_states(num_test, config["state_space"], device)

    return {
        "pool_states": pool_states,
        "train_states": pool_states[:num_train].clone(),
        "val_states": pool_states[num_train:].clone(),
        "test_states": test_states,
        "train_indices": torch.arange(0, num_train, device=device),
        "val_indices": torch.arange(num_train, num_pool, device=device),
    }


def rgb_to_gray_01(frames_rgb):
    frames = frames_rgb.astype(np.float32)
    return (
        0.299 * frames[..., 0]
        + 0.587 * frames[..., 1]
        + 0.114 * frames[..., 2]
    ) / 255.0


@torch.no_grad()
def render_images(dynamic, states, device, render_batch_size):
    states_np = states.detach().cpu().numpy()
    chunks = []
    for start in range(0, states_np.shape[0], render_batch_size):
        frames = []
        for state in states_np[start : start + render_batch_size]:
            frames.append(dynamic.render(state))
        gray = rgb_to_gray_01(np.stack(frames, axis=0))
        images = torch.from_numpy(gray[:, None, :, :]).to(device)
        images = F.interpolate(images, size=(96, 96), mode="bilinear", align_corners=False)
        chunks.append(images.clamp(0.0, 1.0))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def rollout_real(states0, steps, controller, dynamic, device, render_batch_size):
    return rollout(
        states0,
        steps,
        controller,
        dynamic,
        device,
        render_batch_size,
        decoder=None,
    )


@torch.no_grad()
def rollout_dwm(states0, steps, decoder, controller, dynamic, device):
    return rollout(
        states0,
        steps,
        controller,
        dynamic,
        device=device,
        render_batch_size=None,
        decoder=decoder,
    )


@torch.no_grad()
def rollout(states0, steps, controller, dynamic, device, render_batch_size, decoder):
    num_samples, state_dim = states0.shape
    trajectories = torch.empty(num_samples, steps + 1, state_dim, device=device)
    actions = torch.empty(num_samples, steps, 1, device=device)

    states = states0.clone()
    trajectories[:, 0, :] = states
    for step in range(steps):
        if decoder is None:
            images = render_images(dynamic, states, device, render_batch_size)
        else:
            images = decoder(states)
        action = controller(images)
        states = dynamic.step(states, action)
        actions[:, step, :] = action
        trajectories[:, step + 1, :] = states
    return trajectories, actions


def to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32)


def save_dataset(config, initial_splits, trajectory_splits):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "initial_states.npz",
        pool_states=to_numpy(initial_splits["pool_states"]),
        train_states=to_numpy(initial_splits["train_states"]),
        val_states=to_numpy(initial_splits["val_states"]),
        test_states=to_numpy(initial_splits["test_states"]),
        train_indices=initial_splits["train_indices"].detach().cpu().numpy().astype(np.int64),
        val_indices=initial_splits["val_indices"].detach().cpu().numpy().astype(np.int64),
    )

    arrays = {}
    for split_name, split_data in trajectory_splits.items():
        for key, value in split_data.items():
            arrays[f"{split_name}_{key}"] = to_numpy(value)
    np.savez_compressed(output_dir / "trajectories.npz", **arrays)

    metadata = {
        **config,
        "dataset_name": config["dataset_name"],
        "seed_pool": config["seed_pool"],
        "seed_test": config["seed_test"],
        "num_pool": config["num_pool"],
        "num_train": config["num_train"],
        "num_val": config["num_val"],
        "num_test": config["num_test"],
        "rollout_steps": config["rollout_steps"],
        "state_space": config["state_space"],
        "files": {
            "initial_states": "initial_states.npz",
            "trajectories": "trajectories.npz",
            "metadata": "metadata.json",
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return output_dir

def generate_dataset(config):
    device = resolve_device(config.get("device", "auto"))
    decoder, controller = load_models(config, device)
    initial_splits = build_initial_state_splits(config, device)
    steps = int(config["rollout_steps"])
    render_batch_size = int(config.get("render_batch_size", 64))

    dynamic = globals()[config["dynamic"]["name"]](**config["dynamic"].get("args", {}))
    print(f"[Dynamic] {config['dynamic']['name']}")
    try:
        trajectory_splits = {}
        for split_name in ("train", "val", "test"):
            states = initial_splits[f"{split_name}_states"]
            print(f"[Rollout] {split_name}: states={tuple(states.shape)}, steps={steps}")
            real_traj, real_actions = rollout_real(
                states,
                steps,
                controller,
                dynamic,
                device,
                render_batch_size,
            )
            dwm_traj, dwm_actions = rollout_dwm(states, steps, decoder, controller, dynamic, device)
            trajectory_splits[split_name] = {
                "real_traj": real_traj,
                "dwm_traj": dwm_traj,
                "real_actions": real_actions,
                "dwm_actions": dwm_actions,
            }
    finally:
        pass

    return save_dataset(config, initial_splits, trajectory_splits)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MountainCar train/val/test trajectory dataset.")
    parser.add_argument("config", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    print(f"[Config] {args.config}")
    output_dir = generate_dataset(config)
    print(f"[Done] dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
