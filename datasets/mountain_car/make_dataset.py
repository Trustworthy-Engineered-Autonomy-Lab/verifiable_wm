import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYGLET_HEADLESS", "True")

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "mountain_car.json"

MIN_ACTION, MAX_ACTION = -1.0, 1.0
MIN_POS, MAX_POS = -1.2, 0.6
MIN_SPEED, MAX_SPEED = -0.07, 0.07
POWER = 0.0015
ENV_ID = "MountainCarContinuous-v0"


class MCController(nn.Module):
    def __init__(self):
        super().__init__()
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
        return torch.tanh(x)


class MCDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 3 * 12 * 12)
        self.dec_conv1 = nn.ConvTranspose2d(3, 4, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(4, 8, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, states):
        batch_size = states.size(0)
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(batch_size, 3, 12, 12)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        return torch.sigmoid(self.dec_conv3(x))


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_config(path):
    with resolve_path(path).open("r", encoding="utf-8") as f:
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
    decoder = MCDecoder().to(device).eval()
    controller = MCController().to(device).eval()
    decoder.load_state_dict(load_state_dict(resolve_path(config["weights"]["decoder"]), device))
    controller.load_state_dict(load_state_dict(resolve_path(config["weights"]["controller"]), device))
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


@torch.no_grad()
def one_step(states, actions):
    pos = states[:, 0]
    vel = states[:, 1]
    force = torch.clamp(actions.squeeze(1), MIN_ACTION, MAX_ACTION)

    vel = vel + force * POWER - 0.0025 * torch.cos(3.0 * pos)
    vel = torch.clamp(vel, MIN_SPEED, MAX_SPEED)
    pos = pos + vel
    pos = torch.clamp(pos, MIN_POS, MAX_POS)
    vel = torch.where((pos == MIN_POS) & (vel < 0), torch.zeros_like(vel), vel)
    return torch.stack([pos, vel], dim=1)


def rgb_to_gray_01(frames_rgb):
    frames = frames_rgb.astype(np.float32)
    return (
        0.299 * frames[..., 0]
        + 0.587 * frames[..., 1]
        + 0.114 * frames[..., 2]
    ) / 255.0


@torch.no_grad()
def render_images(env, states, device, render_batch_size):
    states_np = states.detach().cpu().numpy()
    chunks = []
    for start in range(0, states_np.shape[0], render_batch_size):
        frames = []
        for state in states_np[start : start + render_batch_size]:
            env.unwrapped.state = np.array(state, dtype=np.float32)
            frames.append(env.render(mode="rgb_array"))
        gray = rgb_to_gray_01(np.stack(frames, axis=0))
        images = torch.from_numpy(gray[:, None, :, :]).to(device)
        images = F.interpolate(images, size=(96, 96), mode="bilinear", align_corners=False)
        chunks.append(images.clamp(0.0, 1.0))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def rollout_real(states0, steps, controller, env, device, render_batch_size):
    return rollout(states0, steps, controller, env, device, render_batch_size, decoder=None)


@torch.no_grad()
def rollout_dwm(states0, steps, decoder, controller, device):
    return rollout(states0, steps, controller, env=None, device=device, render_batch_size=None, decoder=decoder)


@torch.no_grad()
def rollout(states0, steps, controller, env, device, render_batch_size, decoder):
    num_samples, state_dim = states0.shape
    trajectories = torch.empty(num_samples, steps + 1, state_dim, device=device)
    actions = torch.empty(num_samples, steps, 1, device=device)

    states = states0.clone()
    trajectories[:, 0, :] = states
    for step in range(steps):
        if decoder is None:
            images = render_images(env, states, device, render_batch_size)
        else:
            images = decoder(states)
        action = controller(images)
        states = one_step(states, action)
        actions[:, step, :] = action
        trajectories[:, step + 1, :] = states
    return trajectories, actions


def safe_close_env(env):
    try:
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "viewer"):
            viewer = env.unwrapped.viewer
            if viewer is not None:
                viewer.close()
                env.unwrapped.viewer = None
    except Exception:
        pass
    try:
        env.close()
    except Exception:
        pass


def to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32)


def save_dataset(config, initial_splits, trajectory_splits):
    output_dir = resolve_path(config["output_dir"])
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
        "env": config["env"],
        "dataset_name": config["dataset_name"],
        "seed_pool": config["seed_pool"],
        "seed_test": config["seed_test"],
        "num_pool": config["num_pool"],
        "num_train": config["num_train"],
        "num_val": config["num_val"],
        "num_test": config["num_test"],
        "rollout_steps": config["rollout_steps"],
        "state_space": config["state_space"],
        "weights": config["weights"],
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

    env = gym.make(ENV_ID)
    env.reset()
    try:
        trajectory_splits = {}
        for split_name in ("train", "val", "test"):
            states = initial_splits[f"{split_name}_states"]
            print(f"[Rollout] {split_name}: states={tuple(states.shape)}, steps={steps}")
            real_traj, real_actions = rollout_real(states, steps, controller, env, device, render_batch_size)
            dwm_traj, dwm_actions = rollout_dwm(states, steps, decoder, controller, device)
            trajectory_splits[split_name] = {
                "real_traj": real_traj,
                "dwm_traj": dwm_traj,
                "real_actions": real_actions,
                "dwm_actions": dwm_actions,
            }
    finally:
        safe_close_env(env)

    return save_dataset(config, initial_splits, trajectory_splits)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MountainCar train/val/test trajectory dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    print(f"[Config] {resolve_path(args.config)}")
    print(f"[Env] {config['env']}")
    print(f"[Load] decoder={resolve_path(config['weights']['decoder'])}")
    print(f"[Load] controller={resolve_path(config['weights']['controller'])}")
    output_dir = generate_dataset(config)
    print(f"[Done] dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
