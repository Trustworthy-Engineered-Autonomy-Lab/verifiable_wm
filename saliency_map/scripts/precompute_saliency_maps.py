import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from model import Controller
from utils import resolve_device, set_seed


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_controller(config, device):
    controller_cfg = config["controller"]
    name = controller_cfg.get("name", "Controller")
    if name != "Controller":
        raise ValueError(f"Unsupported controller class: {name}")

    controller = Controller(**controller_cfg.get("args", {})).to(device).eval()
    controller.load_state_dict(load_state_dict(controller_cfg["weights"], device))
    return controller


def normalize_per_image(heat):
    flat = heat.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return ((heat - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)


@torch.no_grad()
def occlusion_batch(controller, images, patch, stride, fill):
    base_actions = controller(images)
    heat = torch.zeros_like(images)
    counts = torch.zeros_like(images)
    _, _, height, width = images.shape

    for y in range(0, height - patch + 1, stride):
        for x0 in range(0, width - patch + 1, stride):
            occluded = images.clone()
            occluded[:, :, y : y + patch, x0 : x0 + patch] = fill
            delta = (controller(occluded) - base_actions).abs().view(-1, 1, 1, 1)
            heat[:, :, y : y + patch, x0 : x0 + patch] += delta
            counts[:, :, y : y + patch, x0 : x0 + patch] += 1.0

    return heat / counts.clamp_min(1.0), base_actions


def compute_split(controller, images_np, device, args):
    heat_chunks = []
    action_chunks = []

    for start in range(0, images_np.shape[0], args.batch_size):
        end = min(start + args.batch_size, images_np.shape[0])
        images = torch.from_numpy(images_np[start:end]).float().to(device)

        if args.method == "occlusion":
            heat, actions = occlusion_batch(
                controller,
                images,
                patch=args.occlusion_patch,
                stride=args.occlusion_stride,
                fill=args.occlusion_fill,
            )
        else:
            raise ValueError(f"Unsupported method: {args.method}")

        if args.normalize:
            heat = normalize_per_image(heat)

        heat_chunks.append(heat.cpu().numpy().astype(np.float32))
        action_chunks.append(actions.cpu().numpy().astype(np.float32))
        print(f"[batch] {start}:{end}")

    return np.concatenate(heat_chunks, axis=0), np.concatenate(action_chunks, axis=0)


def default_output_path(dataset_path, method):
    return dataset_path.parent / f"saliency_{method}.npz"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/make_decoder_dataset/cartpole.json"),
        help="Dataset/controller config for the study case.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Defaults to <config output_dir>/states.npz.",
    )
    parser.add_argument("--method", choices=["occlusion"], default="occlusion")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--occlusion-patch", type=int, default=8)
    parser.add_argument("--occlusion-stride", type=int, default=4)
    parser.add_argument("--occlusion-fill", type=float, default=1.0)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <dataset-dir>/saliency_<method>.npz.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    config = load_json(args.config)
    dataset_path = args.dataset or (Path(config["output_dir"]) / "states.npz")
    output_path = args.output or default_output_path(dataset_path, args.method)

    data = np.load(dataset_path)
    controller = build_controller(config, device)

    arrays = {}
    for split in args.splits:
        key = f"{split}_images"
        if key not in data:
            available = ", ".join(data.files)
            raise KeyError(f"Missing key {key}; available keys: {available}")

        print(f"[split] {split}")
        heatmaps, actions = compute_split(controller, data[key], device, args)
        arrays[f"{split}_heatmaps"] = heatmaps
        arrays[f"{split}_actions"] = actions

    arrays["method"] = np.array(args.method)
    arrays["config"] = np.array(str(args.config))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()

