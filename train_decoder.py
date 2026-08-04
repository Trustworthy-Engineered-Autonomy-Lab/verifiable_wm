import argparse
import copy
import csv
import json
import time
from itertools import product
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


def load_split(dataset_dir, saliency_file, split):
    data = np.load(dataset_dir / "decoder_states.npz")
    states = torch.from_numpy(data[f"{split}_states"]).float()
    images = torch.from_numpy(data[f"{split}_images"]).float()

    sal = np.load(dataset_dir / saliency_file)
    heatmaps = torch.from_numpy(sal[f"{split}_heatmaps"]).float()
    if heatmaps.shape != images.shape:
        raise ValueError(
            f"{split}: heatmaps {tuple(heatmaps.shape)} != images {tuple(images.shape)}"
        )
    return states, images, heatmaps


def compute_weight(heatmaps, weight_cfg):
    # paper Eq. 7: the controller-occlusion saliency map weights the
    # reconstruction loss pixel-wise, w = 1 + alpha * H
    w = 1.0 + weight_cfg["alpha"] * heatmaps
    # per-image mean 1 so the loss scale is independent of alpha, and alpha
    # only changes how the reconstruction budget is distributed spatially
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
    weights,
    device,
    batch_size,
    lambda_ctrl,
):
    decoder.eval()
    sums = {"pixel_mse": 0.0, "ctrl_mse": 0.0, "weighted_rec_loss": 0.0}
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

    metrics = {key: value / n for key, value in sums.items()}
    metrics["total_loss"] = metrics["weighted_rec_loss"] + lambda_ctrl * metrics["ctrl_mse"]
    return metrics


def train(config, device):
    dataset_dir = Path(config["dataset_dir"])
    weight_cfg = config["weight"]
    train_cfg = config["training"]
    saliency_file = config.get("saliency_file", "saliency_occlusion.npz")

    train_states, train_images, train_heat = load_split(dataset_dir, saliency_file, "train")
    val_states, val_images, val_heat = load_split(dataset_dir, saliency_file, "val")
    test_states, test_images, test_heat = load_split(dataset_dir, saliency_file, "test")

    train_weights = compute_weight(train_heat, weight_cfg)
    val_weights = compute_weight(val_heat, weight_cfg)
    test_weights = compute_weight(test_heat, weight_cfg)

    controller = load_controller(config, device)
    output_activation = config.get("output_activation", "sigmoid")
    decoder = Decoder(output_activation=output_activation).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=train_cfg["lr"])

    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]
    lambda_ctrl = config["lambda_ctrl"]
    selection_metric = train_cfg.get("selection_metric", "total_loss")

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
            decoder, controller, val_states, val_images, val_weights,
            device, batch_size, lambda_ctrl,
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
        decoder, controller, test_states, test_images, test_weights,
        device, batch_size, lambda_ctrl,
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
    return results


def format_value(value):
    return format(float(value), ".12g")


def grid_row(alpha, lambda_ctrl, seed, output_dir, results):
    """One summary row: the selected epoch's val metrics and the test metrics."""
    best_epoch = int(results["best_epoch"])
    validation = next(
        record for record in results["history"] if int(record["epoch"]) == best_epoch
    )
    return {
        "alpha": format_value(alpha),
        "lambda_ctrl": format_value(lambda_ctrl),
        "seed": int(seed),
        "best_epoch": best_epoch,
        "val_ctrl_mse": validation["val_ctrl_mse"],
        "val_pixel_mse": validation["val_pixel_mse"],
        "val_total_loss": validation["val_total_loss"],
        "test_ctrl_mse": results["test"]["ctrl_mse"],
        "test_pixel_mse": results["test"]["pixel_mse"],
        "test_total_loss": results["test"]["total_loss"],
        "output_dir": str(output_dir),
    }


def run_grid(config, alphas, lambdas, device):
    """Train one decoder per (alpha, lambda_ctrl) pair and summarize the grid.

    A single pair trains straight into the config's output_dir; a real grid
    nests every run under it so the runs never overwrite each other.
    """
    combinations = list(product(alphas, lambdas))
    base_output = Path(config["output_dir"])
    seed = int(config["training"]["seed"])

    if len(combinations) == 1:
        alpha, lambda_ctrl = combinations[0]
        config["weight"]["alpha"] = alpha
        config["lambda_ctrl"] = lambda_ctrl
        set_seed(seed)
        train(config, device)
        return

    print(
        f"[Grid] {len(combinations)} runs: "
        f"alpha={[format_value(a) for a in alphas]}, "
        f"lambda_ctrl={[format_value(l) for l in lambdas]}, seed={seed}"
    )
    rows = []
    for index, (alpha, lambda_ctrl) in enumerate(combinations, start=1):
        run_config = copy.deepcopy(config)
        run_config["weight"]["alpha"] = alpha
        run_config["lambda_ctrl"] = lambda_ctrl
        output_dir = (
            base_output
            / f"alpha_{format_value(alpha)}"
            / f"lambda_{format_value(lambda_ctrl)}"
        )
        run_config["output_dir"] = str(output_dir)
        print(
            f"\n[Grid {index}/{len(combinations)}] "
            f"alpha={format_value(alpha)} lambda_ctrl={format_value(lambda_ctrl)} "
            f"-> {output_dir}"
        )
        set_seed(seed)
        results = train(run_config, device)
        rows.append(grid_row(alpha, lambda_ctrl, seed, output_dir, results))

    summary_path = base_output / "alpha_lambda_grid.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Grid] summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the DWM decoder. One --alpha/--lambda-ctrl value each trains a "
            "single decoder; several values run the ablation grid over their "
            "cartesian product."
        )
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=None,
        help="Override weight.alpha. Several values run the alpha x lambda grid.",
    )
    parser.add_argument(
        "--lambda-ctrl",
        type=float,
        nargs="+",
        default=None,
        help="Override lambda_ctrl. Several values run the alpha x lambda grid.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output_dir.")
    parser.add_argument(
        "--output-activation",
        choices=["sigmoid", "clamp"],
        default=None,
        help="Override output_activation (decoder's final nonlinearity).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get("weight_mode", "saliency") != "saliency":
        raise ValueError(
            "only weight_mode='saliency' is supported: "
            f"got {config['weight_mode']!r}"
        )
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.output_activation is not None:
        config["output_activation"] = args.output_activation

    alphas = args.alpha if args.alpha is not None else [config["weight"]["alpha"]]
    lambdas = (
        args.lambda_ctrl if args.lambda_ctrl is not None else [config["lambda_ctrl"]]
    )
    device = resolve_device(config.get("device", "auto"))
    run_grid(config, alphas, lambdas, device)


if __name__ == "__main__":
    main()
