"""
Shared saliency-map computation library.

Used by both the production script (scripts/precompute_saliency_maps.py, which
only needs `occlusion` to build the training-time weight maps) and the
diagnostic script (scripts/diagnostics/compare_heatmap_methods.py, which runs
all methods side by side for visual comparison). Keeping the controller/
checkpoint loading and the method implementations here means both callers
compute saliency the same way instead of maintaining their own copies.
"""

import json

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


def normalize_per_image(heat):
    flat = heat.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return ((heat - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)


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
