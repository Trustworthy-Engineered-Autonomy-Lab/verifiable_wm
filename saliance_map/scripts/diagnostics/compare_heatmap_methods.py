import argparse
import json
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
import torch.nn.functional as F

from model import Controller


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


def normalize_per_image(heatmaps):
    flat = heatmaps.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return ((heatmaps - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)


def vanilla_gradient(controller, images):
    x = images.detach().clone().requires_grad_(True)
    actions = controller(x)
    grads = torch.autograd.grad(actions.sum(), x)[0]
    return grads.abs().detach(), actions.detach()


def smoothgrad(controller, images, samples=16, noise_std=0.12, square=True):
    x0 = images.detach()
    heat = torch.zeros_like(x0)
    for _ in range(samples):
        x = (x0 + noise_std * torch.randn_like(x0)).clamp(0.0, 1.0).requires_grad_(True)
        score = controller(x).sum()
        grads = torch.autograd.grad(score, x)[0]
        heat = heat + (grads.square() if square else grads.abs())
    return (heat / samples).detach(), controller(images).detach()


def integrated_gradients(controller, images, steps=32, baseline_value=1.0, square=True):
    x0 = images.detach()
    baseline = torch.full_like(x0, baseline_value)
    diff = x0 - baseline
    total = torch.zeros_like(x0)
    for alpha in torch.linspace(1.0 / steps, 1.0, steps, device=images.device):
        x = (baseline + alpha * diff).requires_grad_(True)
        score = controller(x).sum()
        grads = torch.autograd.grad(score, x)[0]
        total = total + grads
    attr = diff * total / steps
    return (attr.square() if square else attr.abs()).detach(), controller(images).detach()


def gradcam(controller, images, layer_name="conv2"):
    features = {}
    layer = getattr(controller, layer_name)

    def hook(_module, _inputs, output):
        features["activation"] = output

    handle = layer.register_forward_hook(hook)
    try:
        x = images.detach().clone().requires_grad_(True)
        actions = controller(x)
        activation = features["activation"]
        grads = torch.autograd.grad(actions.sum(), activation)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return cam.detach(), actions.detach()
    finally:
        handle.remove()


@torch.no_grad()
def occlusion(controller, images, patch=8, stride=4, fill=1.0):
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
    return (heat / counts.clamp_min(1.0)).detach(), base_actions.detach()


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


def save_overlay_grid(images, actions, outputs, output_path, alpha):
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

    fig.suptitle("Controller heatmap method comparison", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


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
        default=Path("saliance_map/output/diagnostics/heatmap_methods_compare"),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    config = load_json(args.config)
    dataset_path = args.dataset or (Path(config["output_dir"]) / "states.npz")
    data = np.load(dataset_path)
    images = torch.from_numpy(data[args.image_key][: args.num_images]).float().to(device)

    controller = build_controller(config, device)

    outputs, actions = compute_methods(controller, images, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_overlay_grid(images, actions, outputs, args.output_dir / "heatmap_methods_grid.png", args.overlay_alpha)

    print(f"[saved] {args.output_dir / 'heatmap_methods_grid.png'}")
    print("[actions]", actions.detach().cpu().numpy().reshape(-1).round(6).tolist())


if __name__ == "__main__":
    main()
