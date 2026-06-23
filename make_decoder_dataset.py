import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYGLET_HEADLESS", "True")

import numpy as np
import torch
import torch.nn.functional as F

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


def sample_uniform_states(num_samples, state_space, device):
    columns = []
    for dim in state_space:
        low = float(dim["low"])
        high = float(dim["high"])
        columns.append(low + (high - low) * torch.rand(num_samples, 1, device=device))
    return torch.cat(columns, dim=1).float()


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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32)


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

def select_decoder_states(states, config):
    indices = config.get("decoder_state_indices", None)
    if indices is None:
        return states
    return states[:, indices]

def generate_dataset(config):
    device = resolve_device(config.get("device", "auto"))
    render_batch_size = int(config.get("render_batch_size", 64))

    dynamic = globals()[config["dynamic"]["name"]](**config["dynamic"].get("args", {}))
    print(f"[Dynamic] {config['dynamic']['name']}")

    state_splits = build_initial_state_splits(config, device)

    dataset = {}
    for split_name in ("train_states", "val_states", "test_states"):
        states = state_splits[split_name]
        images = render_images(dynamic, states, device, render_batch_size)

        prefix = split_name.replace("_states", "")
        decoder_states = select_decoder_states(states, config)

        dataset[f"{prefix}_states"] = decoder_states
        dataset[f"{prefix}_images"] = images

        print(f"[Render] {prefix}: states={tuple(states.shape)}, images={tuple(images.shape)}")

    return save_dataset(config, dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = generate_dataset(config)
    print(f"[Done] decoder dataset saved to {output_dir}")


if __name__ == "__main__":
    main()