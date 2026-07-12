"""
One-off metric readout for a decoder checkpoint that wasn't produced by
train_decoder.py itself (e.g. the raw/old decoder weights) so it can be
compared against intensity/saliency test metrics on equal footing.

Reuses train_decoder.py's load_split/compute_weight/evaluate so the numbers
are computed exactly the same way as the ones in metrics.json.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from model import Decoder
from utils import load_config, resolve_device
from train_decoder import load_controller, load_split, compute_weight, evaluate, load_state_dict


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                         help="train_decoder config; supplies dataset_dir/controller/weight settings.")
    parser.add_argument("--weights", type=Path, required=True,
                         help="Decoder checkpoint to evaluate.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--label", default=None)
    return parser.parse_args(argv)


def run(args):
    config = load_config(args.config)
    device = resolve_device(config.get("device", "auto"))

    dataset_dir = Path(config["dataset_dir"])
    saliency_file = config.get("saliency_file", "saliency_occlusion.npz")
    weight_mode = config["weight_mode"]
    weight_cfg = config["weight"]
    lambda_ctrl = config["lambda_ctrl"]
    region_threshold = weight_cfg.get("threshold", 0.7)
    batch_size = config["training"]["batch_size"]

    states, images, heat = load_split(dataset_dir, saliency_file, args.split, "saliency")
    weights = compute_weight(weight_mode, images, heat, weight_cfg)

    controller = load_controller(config, device)
    decoder = Decoder().to(device).eval()
    decoder.load_state_dict(load_state_dict(args.weights, device))

    metrics = evaluate(
        decoder, controller, states, images, heat, weights,
        device, batch_size, region_threshold, lambda_ctrl,
    )

    label = args.label or args.weights.stem
    print(f"[{label}:{args.split}] " + " ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
    return metrics


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
