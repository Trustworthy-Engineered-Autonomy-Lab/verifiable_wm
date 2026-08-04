"""
Controller-occlusion saliency, used by scripts/precompute_saliency_maps.py to
build the training-time weight maps H that train_decoder.py turns into the
pixel-wise reconstruction weight w = 1 + alpha * H.
"""

import json

import torch

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
