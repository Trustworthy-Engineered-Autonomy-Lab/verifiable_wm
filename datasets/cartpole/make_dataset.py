import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYGLET_HEADLESS", "True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "cartpole.json"


class CartPoleDecoder(nn.Module):
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
        return torch.clamp(self.dec_conv3(x), 0.0, 1.0)


class CartPoleController(nn.Module):
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
        return torch.sigmoid(x)


class DefaultCartPoleEnv:
    # TODO（CartPole）: 当前项目缺少师哥原始 ContinuousCartPoleEnv。
    # 这里先保留 conformal prediction/cartpole/rollout_95.py 的占位渲染风格；
    # 后续拿到真实环境后，需要替换这个类和 step_cartpole。
    def __init__(self):
        self.state = np.zeros(4, dtype=np.float32)

    def reset(self):
        self.state = np.zeros(4, dtype=np.float32)
        return self.state.copy()

    def render(self, mode="rgb_array"):
        return np.ones((400, 600, 3), dtype=np.uint8) * 255

    def close(self):
        pass


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
    decoder = CartPoleDecoder().to(device).eval()
    controller = CartPoleController().to(device).eval()
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

    # TODO（CartPole）: velocity / angular_velocity 的范围是当前 config 的默认假设。
    # Gym CartPole 没有明确有限原始范围，后续需要根据项目定义确认。
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
def step_cartpole(states, actions):
    # TODO（CartPole）: 默认动力学来自 CP 脚本的 fallback 版本，不是最终正式环境。
    x = states[:, 0]
    x_dot = states[:, 1]
    theta = states[:, 2]
    theta_dot = states[:, 3]
    force = actions.squeeze(1) * 20.0 - 10.0

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    temp = (force + 0.5 * theta_dot**2 * sintheta) / 1.1
    theta_acc = (
        9.8 * sintheta - costheta * temp
    ) / (0.5 * (4.0 / 3.0 - 0.1 * costheta**2 / 1.1))
    x_acc = temp - 0.5 * theta_acc * costheta / 1.1

    dt = 0.02
    next_x = x + dt * x_dot
    next_x_dot = x_dot + dt * x_acc
    next_theta = theta + dt * theta_dot
    next_theta_dot = theta_dot + dt * theta_acc
    return torch.stack([next_x, next_x_dot, next_theta, next_theta_dot], dim=1)


def rgb_to_gray_01(frame):
    frame = frame.astype(np.float32)
    return (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]) / 255.0


def resize_rgb_nearest(image, width, height):
    y_idx = np.linspace(0, image.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, width).astype(np.int64)
    return image[y_idx][:, x_idx]


def state_to_image(env, state):
    env.state = np.array(state, dtype=np.float32)
    frame = env.render(mode="rgb_array")
    canvas = np.ones((600, 600, 3), dtype=np.uint8) * 255
    canvas[100:500, :, :] = frame
    resized = resize_rgb_nearest(canvas, 96, 96)
    gray = rgb_to_gray_01(resized) * 255.0
    gray[63:64, :] = 0.0
    return gray.astype(np.float32)


@torch.no_grad()
def render_images(env, states, device):
    states_np = states.detach().cpu().numpy()
    buffer = np.empty((states_np.shape[0], 96, 96), dtype=np.float32)
    for idx, state in enumerate(states_np):
        buffer[idx] = state_to_image(env, state)
    return torch.from_numpy(buffer[:, None, :, :]).to(device)


def decoder_input(states):
    return states[:, [0, 2]]


@torch.no_grad()
def rollout_real(states0, steps, controller, env, device):
    return rollout(states0, steps, controller, env, device, decoder=None)


@torch.no_grad()
def rollout_dwm(states0, steps, decoder, controller, device):
    return rollout(states0, steps, controller, env=None, device=device, decoder=decoder)


@torch.no_grad()
def rollout(states0, steps, controller, env, device, decoder):
    num_samples, state_dim = states0.shape
    trajectories = torch.empty(num_samples, steps + 1, state_dim, device=device)
    actions = torch.empty(num_samples, steps, 1, device=device)

    states = states0.clone()
    trajectories[:, 0, :] = states
    for step in range(steps):
        images = render_images(env, states, device) if decoder is None else decoder(decoder_input(states))
        action = controller(images)
        states = step_cartpole(states, action)
        actions[:, step, :] = action
        trajectories[:, step + 1, :] = states
    return trajectories, actions


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
    env = DefaultCartPoleEnv()
    try:
        trajectory_splits = {}
        for split_name in ("train", "val", "test"):
            states = initial_splits[f"{split_name}_states"]
            print(f"[Rollout] {split_name}: states={tuple(states.shape)}, steps={steps}")
            real_traj, real_actions = rollout_real(states, steps, controller, env, device)
            dwm_traj, dwm_actions = rollout_dwm(states, steps, decoder, controller, device)
            trajectory_splits[split_name] = {
                "real_traj": real_traj,
                "dwm_traj": dwm_traj,
                "real_actions": real_actions,
                "dwm_actions": dwm_actions,
            }
    finally:
        env.close()
    return save_dataset(config, initial_splits, trajectory_splits)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CartPole train/val/test trajectory dataset.")
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
