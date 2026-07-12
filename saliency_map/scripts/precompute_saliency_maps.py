import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from utils import resolve_device, set_seed
from saliency_map.methods import (
    build_controller,
    load_json,
    normalize_per_image,
    occlusion,
)


def compute_split(controller, images_np, device, args):
    heat_chunks = []
    action_chunks = []

    for start in range(0, images_np.shape[0], args.batch_size):
        end = min(start + args.batch_size, images_np.shape[0])
        images = torch.from_numpy(images_np[start:end]).float().to(device)

        if args.method == "occlusion":
            heat, actions = occlusion(
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


def parse_args(argv=None):
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
        help="Defaults to <config output_dir>/decoder_states.npz.",
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
    return parser.parse_args(argv)


def run(args):
    set_seed(args.seed)
    device = resolve_device(args.device)

    config = load_json(args.config)
    dataset_path = args.dataset or (Path(config["output_dir"]) / "decoder_states.npz")
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


def main():
    run(parse_args())


if __name__ == "__main__":
    main()

