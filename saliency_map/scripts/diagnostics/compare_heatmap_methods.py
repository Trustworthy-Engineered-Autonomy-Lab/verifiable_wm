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

from saliency_map.methods import (
    build_controller,
    gradcam,
    integrated_gradients,
    load_json,
    normalize_per_image,
    occlusion,
    smoothgrad,
    vanilla_gradient,
)


def compute_methods(controller, images, args):
    methods = [
        ("vanilla_grad", "Vanilla grad", lambda: vanilla_gradient(controller, images)),
        ("smoothgrad_abs", "SmoothGrad", lambda: smoothgrad(controller, images, args.samples, args.noise_std, square=False)),
        ("smoothgrad_sq", "SmoothGrad^2", lambda: smoothgrad(controller, images, args.samples, args.noise_std, square=True)),
        ("ig_white_sq", "IG white^2", lambda: integrated_gradients(controller, images, args.ig_steps, baseline_value=1.0, square=True)),
        ("gradcam", "Grad-CAM", lambda: gradcam(controller, images)),
        ("occlusion_white", "Occlusion white", lambda: occlusion(controller, images, args.occlusion_patch, args.occlusion_stride, fill=1.0)),
    ]

    outputs = {}
    actions = None
    for key, label, fn in methods:
        heat, method_actions = fn()
        outputs[key] = {
            "label": label,
            "heatmap": normalize_per_image(heat),
            "raw_max": heat.flatten(1).max(dim=1).values.detach().cpu().numpy(),
            "raw_mean": heat.flatten(1).mean(dim=1).detach().cpu().numpy(),
        }
        if actions is None:
            actions = method_actions
    return outputs, actions


def infer_case_name(config_path):
    return config_path.stem


def save_overlay_grid(images, actions, outputs, output_path, alpha, title):
    image_np = images.detach().cpu().numpy()[:, 0]
    actions_np = actions.detach().cpu().numpy().reshape(-1)
    method_keys = list(outputs)
    rows = image_np.shape[0]
    cols = 1 + len(method_keys)

    fig, axes = plt.subplots(rows, cols, figsize=(2.35 * cols, 2.05 * rows))
    if rows == 1:
        axes = axes[None, :]

    for row in range(rows):
        axes[row, 0].imshow(image_np[row], cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_title("image", fontsize=8)
        axes[row, 0].set_ylabel(f"#{row}\na={actions_np[row]:.4f}", rotation=0, labelpad=24, fontsize=8)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        for col, key in enumerate(method_keys, start=1):
            heat = outputs[key]["heatmap"][row, 0].detach().cpu().numpy()
            axes[row, col].imshow(image_np[row], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, col].imshow(heat, cmap="magma", vmin=0.0, vmax=1.0, alpha=alpha)
            axes[row, col].set_title(outputs[key]["label"], fontsize=8)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


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
    parser.add_argument("--image-key", default="train_images")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--noise-std", type=float, default=0.12)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--occlusion-patch", type=int, default=8)
    parser.add_argument("--occlusion-stride", type=int, default=4)
    parser.add_argument("--overlay-alpha", type=float, default=0.6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("saliency_map/output/previews"),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Defaults to <config-stem>_saliency_methods.png.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    config = load_json(args.config)
    dataset_path = args.dataset or (Path(config["output_dir"]) / "decoder_states.npz")
    data = np.load(dataset_path)
    images = torch.from_numpy(data[args.image_key][: args.num_images]).float().to(device)

    controller = build_controller(config, device)

    outputs, actions = compute_methods(controller, images, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_name = infer_case_name(args.config)
    output_name = args.output_name or f"{case_name}_saliency_methods.png"
    output_path = args.output_dir / output_name
    save_overlay_grid(
        images,
        actions,
        outputs,
        output_path,
        args.overlay_alpha,
        title=f"{case_name}: controller saliency method preview",
    )

    print(f"[saved] {output_path}")
    print("[actions]", actions.detach().cpu().numpy().reshape(-1).round(6).tolist())


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
