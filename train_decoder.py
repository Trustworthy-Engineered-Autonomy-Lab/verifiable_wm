import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import Controller, Decoder
from utils import load_config, resolve_device, set_seed


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_controller(config, device):
    controller_config = config["controller"]
    controller = Controller(**controller_config.get("args", {})).to(device).eval()
    controller.load_state_dict(load_state_dict(controller_config["weights"], device))
    for p in controller.parameters():
        p.requires_grad_(False)
    print(f"[Load] Controller={controller_config['weights']}")
    return controller


def load_split(dataset_dir, saliency_file, split, weight_mode):
    data = np.load(dataset_dir / "states.npz")
    states = torch.from_numpy(data[f"{split}_states"]).float()
    images = torch.from_numpy(data[f"{split}_images"]).float()

    heatmaps = None
    if weight_mode in ("saliency", "hybrid"):
        sal = np.load(dataset_dir / saliency_file)
        heatmaps = torch.from_numpy(sal[f"{split}_heatmaps"]).float()
        if heatmaps.shape != images.shape:
            raise ValueError(
                f"{split}: heatmaps {tuple(heatmaps.shape)} != images {tuple(images.shape)}"
            )
    return states, images, heatmaps


def compute_weight(weight_mode, images, heatmaps, weight_cfg):
    # intensity follows the paper exactly (Eq. 7): binary threshold on pixel value
    if weight_mode == "intensity":
        w = torch.where(
            images <= weight_cfg["threshold"],
            torch.full_like(images, weight_cfg["obj_w"]),
            torch.full_like(images, weight_cfg["bg_w"]),
        )
    elif weight_mode == "saliency":
        w = 1.0 + weight_cfg["alpha"] * heatmaps
    elif weight_mode == "hybrid":
        intensity = torch.where(
            images <= weight_cfg["threshold"],
            torch.full_like(images, weight_cfg["obj_w"]),
            torch.full_like(images, weight_cfg["bg_w"]),
        )
        w = intensity + weight_cfg["alpha"] * heatmaps
    else:
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")

    # per-image mean 1 so all modes share the same loss scale and differ
    # only in how the reconstruction budget is distributed spatially
    return w / w.mean(dim=(1, 2, 3), keepdim=True)


def best_checkpoint_name(selection_metric):
    if selection_metric == "total_loss":
        return "decoder_best_total.pth"
    safe_metric = selection_metric.replace("/", "_")
    return f"decoder_best_{safe_metric}.pth"


@torch.no_grad()
def evaluate(
    decoder,
    controller,
    states,
    images,
    heatmaps,
    weights,
    device,
    batch_size,
    region_threshold,
    lambda_ctrl,
):
    decoder.eval()
    sums = {
        "pixel_mse": 0.0,
        "ctrl_mse": 0.0,
        "weighted_rec_loss": 0.0,
        "region_dark_mse": 0.0,
        "region_sal_mse": 0.0,
    }
    counts = {"region_dark": 0.0, "region_sal": 0.0}
    n = states.shape[0]

    for i in range(0, n, batch_size):
        s = states[i : i + batch_size].to(device)
        target = images[i : i + batch_size].to(device)
        w = weights[i : i + batch_size].to(device)
        recon = decoder(s)
        err = (recon - target) ** 2

        sums["pixel_mse"] += err.mean().item() * s.shape[0]
        sums["weighted_rec_loss"] += (w * err).mean().item() * s.shape[0]
        ctrl_err = (controller(recon) - controller(target)) ** 2
        sums["ctrl_mse"] += ctrl_err.mean().item() * s.shape[0]

        dark = (target <= region_threshold).float()
        sums["region_dark_mse"] += (err * dark).sum().item()
        counts["region_dark"] += dark.sum().item()

        if heatmaps is not None:
            sal = (heatmaps[i : i + batch_size].to(device) >= 0.5).float()
            sums["region_sal_mse"] += (err * sal).sum().item()
            counts["region_sal"] += sal.sum().item()

    metrics = {
        "pixel_mse": sums["pixel_mse"] / n,
        "ctrl_mse": sums["ctrl_mse"] / n,
        "weighted_rec_loss": sums["weighted_rec_loss"] / n,
        "region_dark_mse": sums["region_dark_mse"] / max(counts["region_dark"], 1.0),
    }
    metrics["total_loss"] = metrics["weighted_rec_loss"] + lambda_ctrl * metrics["ctrl_mse"]
    if heatmaps is not None:
        metrics["region_sal_mse"] = sums["region_sal_mse"] / max(counts["region_sal"], 1.0)
    return metrics


def train(config, device):
    dataset_dir = Path(config["dataset_dir"])
    weight_mode = config["weight_mode"]
    weight_cfg = config["weight"]
    train_cfg = config["training"]
    saliency_file = config.get("saliency_file", "saliency_occlusion.npz")

    # evaluation always loads heatmaps (if present) so all modes report the
    # same metric set; training only uses them when weight_mode requires it
    eval_mode = "saliency"
    train_states, train_images, train_heat = load_split(
        dataset_dir, saliency_file, "train", eval_mode
    )
    val_states, val_images, val_heat = load_split(dataset_dir, saliency_file, "val", eval_mode)
    test_states, test_images, test_heat = load_split(dataset_dir, saliency_file, "test", eval_mode)

    train_weights = compute_weight(weight_mode, train_images, train_heat, weight_cfg)
    val_weights = compute_weight(weight_mode, val_images, val_heat, weight_cfg)
    test_weights = compute_weight(weight_mode, test_images, test_heat, weight_cfg)

    controller = load_controller(config, device)
    decoder = Decoder().to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=train_cfg["lr"])

    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]
    lambda_ctrl = config["lambda_ctrl"]
    selection_metric = train_cfg.get("selection_metric", "total_loss")
    region_threshold = config["weight"].get("threshold", 0.7)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / best_checkpoint_name(selection_metric)

    best = {"epoch": -1, "value": float("inf")}
    history = []
    n = train_states.shape[0]
    start_time = time.time()

    for epoch in range(epochs):
        decoder.train()
        perm = torch.randperm(n)
        epoch_rec, epoch_ctrl = 0.0, 0.0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            s = train_states[idx].to(device)
            target = train_images[idx].to(device)
            w = train_weights[idx].to(device)

            recon = decoder(s)
            loss_rec = (w * (recon - target) ** 2).mean()
            loss_ctrl = ((controller(recon) - controller(target)) ** 2).mean()
            loss = loss_rec + lambda_ctrl * loss_ctrl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_rec += loss_rec.item() * idx.shape[0]
            epoch_ctrl += loss_ctrl.item() * idx.shape[0]

        val_metrics = evaluate(
            decoder, controller, val_states, val_images, val_heat, val_weights,
            device, batch_size, region_threshold, lambda_ctrl,
        )
        if selection_metric not in val_metrics:
            available = ", ".join(sorted(val_metrics))
            raise KeyError(
                f"Unknown selection_metric={selection_metric!r}; available: {available}"
            )
        record = {
            "epoch": epoch,
            "train_rec": epoch_rec / n,
            "train_ctrl": epoch_ctrl / n,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        if val_metrics[selection_metric] < best["value"]:
            best = {"epoch": epoch, "value": val_metrics[selection_metric]}
            torch.save(decoder.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"[epoch {epoch:3d}] rec={record['train_rec']:.5f} "
                f"ctrl={record['train_ctrl']:.5f} "
                f"val_total={record['val_total_loss']:.5f} "
                f"val_pixel={record['val_pixel_mse']:.5f} "
                f"val_ctrl={record['val_ctrl_mse']:.5f} "
                f"best@{best['epoch']}"
            )

    torch.save(decoder.state_dict(), output_dir / "decoder_last.pth")

    decoder.load_state_dict(load_state_dict(best_path, device))
    test_metrics = evaluate(
        decoder, controller, test_states, test_images, test_heat, test_weights,
        device, batch_size, region_threshold, lambda_ctrl,
    )
    print(
        f"[test:{best_path.name}@{best['epoch']}] "
        + " ".join(f"{k}={v:.5f}" for k, v in test_metrics.items())
    )

    results = {
        "config": config,
        "best_epoch": best["epoch"],
        "best_value": best["value"],
        "best_checkpoint": best_path.name,
        "selection_metric": selection_metric,
        "test": test_metrics,
        "history": history,
        "train_seconds": time.time() - start_time,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--alpha", type=float, default=None, help="Override weight.alpha (sweeps).")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed (sweeps).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output_dir.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.alpha is not None:
        config["weight"]["alpha"] = args.alpha
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)

    set_seed(config["training"]["seed"])
    device = resolve_device(config.get("device", "auto"))
    train(config, device)


if __name__ == "__main__":
    main()
