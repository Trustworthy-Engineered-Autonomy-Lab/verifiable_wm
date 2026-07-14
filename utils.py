import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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


def starv_grid_to_state_space(starv_config):
    dims = starv_config["grid"]["dims"]
    return [
        {
            "name": dim["name"],
            "low": float(dim["start"]),
            "high": float(dim["stop"]),
        }
        for dim in dims
    ]


def sample_starv_state_splits(starv_config, device):
    sampling = starv_config["starv_states"]
    state_space = starv_grid_to_state_space(starv_config)
    splits = {}

    for split in ("train", "val", "test"):
        set_seed(int(sampling[f"seed_{split}"]))
        splits[f"{split}_states"] = sample_uniform_states(
            int(sampling[f"num_{split}"]),
            state_space,
            device,
        )

    return splits


def starv_states_path(starv_config):
    return Path(starv_config["starv_states"]["output_file"])


def save_state_splits(path, splits):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: to_numpy(value) for key, value in splits.items()})


def load_state_splits(path, device):
    with np.load(Path(path), allow_pickle=False) as data:
        required = {"train_states", "val_states", "test_states"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Missing state splits in {path}: {sorted(missing)}")
        return {
            key: torch.from_numpy(np.asarray(data[key], dtype=np.float32)).to(device)
            for key in sorted(required)
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
