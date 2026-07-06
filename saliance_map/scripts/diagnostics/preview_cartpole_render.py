import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dynamic import CartPole
from utils import render_images


def infer_image_key(data, split, image_key):
    if image_key is not None:
        return image_key
    if f"{split}_images" in data:
        return f"{split}_images"
    if "images" in data:
        return "images"
    available = ", ".join(data.files)
    raise KeyError(f"Could not infer image key; available keys: {available}")


def infer_full_states(states_data, traj_data, split, num_images):
    traj_key = f"{split}_traj"
    if traj_data is not None and traj_key in traj_data:
        return torch.from_numpy(traj_data[traj_key][:num_images, 0, :]).float()

    state_key = f"{split}_states"
    if state_key in states_data:
        states = states_data[state_key][:num_images]
        if states.shape[1] == 4:
            return torch.from_numpy(states).float()

    raise KeyError(
        "Could not infer full 4D CartPole states. Provide --traj-npz or a states npz with full 4D states."
    )


def save_preview(saved_images, rerendered_images, output_path):
    rows = saved_images.shape[0]
    cols = 2 if rerendered_images is not None else 1
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.2 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[None, :]
    elif cols == 1:
        axes = axes[:, None]

    for idx in range(rows):
        axes[idx, 0].imshow(saved_images[idx], cmap="gray", vmin=0.0, vmax=1.0)
        axes[idx, 0].set_title("saved npz")
        axes[idx, 0].set_ylabel(f"#{idx}", rotation=0, labelpad=16)

        if rerendered_images is not None:
            axes[idx, 1].imshow(rerendered_images[idx], cmap="gray", vmin=0.0, vmax=1.0)
            axes[idx, 1].set_title("current render")

        for col in range(cols):
            axes[idx, col].set_xticks([])
            axes[idx, col].set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-npz", type=Path, default=Path("datasets/cartpole/data/dataset_v1/states.npz"))
    parser.add_argument(
        "--traj-npz",
        type=Path,
        default=Path("datasets/cartpole/data/dataset_v1/real_trajectories.npz"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--image-key", default=None)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--render-current", action="store_true")
    parser.add_argument("--render-batch-size", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("saliance_map/output/diagnostics/cartpole_render_preview"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    states_data = np.load(args.states_npz)
    traj_data = np.load(args.traj_npz) if args.traj_npz.exists() else None

    image_key = infer_image_key(states_data, args.split, args.image_key)
    saved_images = states_data[image_key][: args.num_images, 0]

    rerendered = None
    if args.render_current:
        full_states = infer_full_states(states_data, traj_data, args.split, args.num_images)
        dynamic = CartPole()
        try:
            rerendered = render_images(dynamic, full_states, torch.device("cpu"), args.render_batch_size)[:, 0].numpy()
        finally:
            env = getattr(dynamic, "env", None)
            if env is not None:
                env.close()

    output_path = args.output_dir / f"{args.split}_preview.png"
    save_preview(saved_images, rerendered, output_path)
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
