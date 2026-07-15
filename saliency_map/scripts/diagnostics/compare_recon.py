"""
Decoder reconstruction comparison for a fixed set of test states: real image
vs. several decoder checkpoints side by side.

Replaces the old per-env compare_alpha2_recon.py / compare_pendulum_recon.py,
which were the same script with different checkpoint paths hardcoded in.
`run()` takes the checkpoint set as a plain {label: path} dict so cartpole and
pendulum (or any future env) share one implementation.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import Decoder
from saliency_map.methods import load_state_dict

DEFAULT_CHECKPOINTS = {
    "cartpole": {
        "intensity (paper baseline)": REPO_ROOT / "dwm_weight/now_weight/cartpole/intensity/decoder_last.pth",
        "saliency alpha=2": REPO_ROOT / "dwm_weight/now_weight/cartpole/saliency_alpha_sweep/alpha_2/seed_2025/decoder_last.pth",
        "saliency alpha=8 (mainline)": REPO_ROOT / "dwm_weight/now_weight/cartpole/saliency/decoder_last.pth",
    },
    "pendulum": {
        "raw (old decoder)": REPO_ROOT / "dwm_weight/raw_weight/pendulum/decoder_pen.pth",
        "intensity (paper baseline)": REPO_ROOT / "dwm_weight/now_weight/pendulum/intensity/decoder_last.pth",
        "saliency alpha=8 (mainline)": REPO_ROOT / "dwm_weight/now_weight/pendulum/saliency/decoder_last.pth",
    },
}


def default_dataset_dir(env_name):
    return REPO_ROOT / f"datasets/{env_name}/data/dataset_v1"


def default_output_path(env_name):
    return REPO_ROOT / f"saliency_map/output/previews/{env_name}_recon_compare.png"


def run(env_name, dataset_dir, checkpoints, output_path=None, num_samples=6, seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(Path(dataset_dir) / "decoder_states.npz")
    test_states = torch.from_numpy(data["test_states"]).float().to(device)
    test_images = data["test_images"]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(test_states), size=num_samples, replace=False)
    idx.sort()

    rows = [("real", test_images[idx])]
    for label, checkpoint_path in checkpoints.items():
        decoder = Decoder().to(device).eval()
        decoder.load_state_dict(load_state_dict(checkpoint_path, device))
        with torch.no_grad():
            recon = decoder(test_states[idx]).cpu().numpy()
        rows.append((label, recon))

    fig, axes = plt.subplots(len(rows), len(idx), figsize=(2 * len(idx), 2 * len(rows)))
    for r, (label, imgs) in enumerate(rows):
        for c in range(len(idx)):
            ax = axes[r, c]
            ax.imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            if r == 0:
                ax.set_title(f"test idx {idx[c]}", fontsize=9)

    fig.tight_layout()
    output_path = Path(output_path) if output_path is not None else default_output_path(env_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {output_path}")
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["cartpole", "pendulum"], default="cartpole")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    run(
        env_name=args.env,
        dataset_dir=args.dataset_dir or default_dataset_dir(args.env),
        checkpoints=DEFAULT_CHECKPOINTS[args.env],
        output_path=args.output,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
