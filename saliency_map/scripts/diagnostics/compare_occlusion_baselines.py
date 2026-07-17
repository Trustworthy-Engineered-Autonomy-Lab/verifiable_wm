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

from saliency_map.methods import build_controller, load_json
from saliency_map.scripts.precompute_saliency_maps import build_background_median
from utils import resolve_device, set_seed


def rank_mask(heatmaps, kind, fraction, seed):
    if kind not in {"top", "bottom", "random"}:
        raise ValueError(f"Unsupported mask kind: {kind}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"mask fraction must be in (0, 1], got {fraction}")

    flat_heatmaps = heatmaps.flatten(1)
    count = max(1, round(flat_heatmaps.shape[1] * fraction))
    masks = torch.zeros_like(flat_heatmaps, dtype=torch.bool)
    if kind == "random":
        rng = np.random.default_rng(seed)
        for index in range(flat_heatmaps.shape[0]):
            selected = torch.as_tensor(
                rng.choice(flat_heatmaps.shape[1], size=count, replace=False),
                device=heatmaps.device,
            )
            masks[index, selected] = True
    else:
        selected = torch.topk(
            flat_heatmaps, count, dim=1, largest=(kind == "top")
        ).indices
        masks.scatter_(1, selected, True)
    return masks.view_as(heatmaps)


@torch.no_grad()
def evaluate_ranked_masks(controller, images, heatmaps, background, fraction, seed):
    if heatmaps.shape != images.shape:
        raise ValueError(f"heatmaps {tuple(heatmaps.shape)} != images {tuple(images.shape)}")
    if background.shape != (1, *images.shape[1:]):
        raise ValueError(
            "background must have shape "
            f"{(1, *images.shape[1:])}, got {tuple(background.shape)}"
        )

    base_actions = controller(images).reshape(-1)
    results = {"base_actions": base_actions.detach().cpu().numpy()}
    for kind in ("top", "bottom", "random"):
        masks = rank_mask(heatmaps, kind, fraction, seed)
        masked_images = torch.where(masks, background.expand_as(images), images)
        actions = controller(masked_images).reshape(-1)
        results[f"{kind}_actions"] = actions.detach().cpu().numpy()
        results[f"{kind}_delta"] = (actions - base_actions).abs().detach().cpu().numpy()
    return results


def save_preview(images, white_heatmaps, background_heatmaps, states, indices, output_path):
    rows = [
        ("REAL RENDERER", None),
        ("OCCLUSION WHITE", white_heatmaps),
        ("OCCLUSION BACKGROUND MEDIAN", background_heatmaps),
    ]
    fig, axes = plt.subplots(len(rows), len(indices), figsize=(2.15 * len(indices), 6.4))
    if len(indices) == 1:
        axes = axes[:, None]

    for column, index in enumerate(indices):
        image = images[index, 0]
        state = states[index]
        for row, (label, heatmaps) in enumerate(rows):
            ax = axes[row, column]
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            if heatmaps is not None:
                ax.imshow(heatmaps[index, 0], cmap="magma", vmin=0.0, vmax=1.0, alpha=0.65)
            if row == 0:
                ax.set_title(
                    f"test #{index}\npos={state[0]:+.3f}, vel={state[1]:+.3f}", fontsize=8
                )
            if column == 0:
                ax.set_ylabel(label, rotation=0, labelpad=58, va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("MountainCar: same controller, same images, different occlusion fill", y=0.995)
    fig.tight_layout(rect=(0.06, 0.0, 1.0, 0.98))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/make_decoder_dataset/mountain_car.json"),
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--white-map", type=Path, default=None)
    parser.add_argument("--background-map", type=Path, default=None)
    parser.add_argument("--mask-fraction", type=float, default=0.05)
    parser.add_argument("--num-preview", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("saliency_map/output/previews/mountain_car_white_vs_background_median.png"),
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("saliency_map/output/mountain_car_white_vs_background_median_causal.npz"),
    )
    return parser.parse_args(argv)


def run(args):
    set_seed(args.seed)
    config = load_json(args.config)
    dataset_path = args.dataset or (Path(config["output_dir"]) / "decoder_states.npz")
    white_map_path = args.white_map or dataset_path.parent / "saliency_occlusion.npz"
    background_map_path = (
        args.background_map
        or dataset_path.parent / "saliency_occlusion_background_median.npz"
    )
    data = np.load(dataset_path)
    with np.load(white_map_path) as white_data, np.load(background_map_path) as background_data:
        images_np = data[f"{args.split}_images"]
        states = data[f"{args.split}_states"]
        white_heatmaps_np = white_data[f"{args.split}_heatmaps"]
        background_heatmaps_np = background_data[f"{args.split}_heatmaps"]

    if white_heatmaps_np.shape != images_np.shape:
        raise ValueError(f"white heatmaps {white_heatmaps_np.shape} != images {images_np.shape}")
    if background_heatmaps_np.shape != images_np.shape:
        raise ValueError(
            f"background heatmaps {background_heatmaps_np.shape} != images {images_np.shape}"
        )

    device = resolve_device(args.device)
    controller = build_controller(config, device)
    images = torch.from_numpy(images_np).float().to(device)
    background = torch.from_numpy(build_background_median(data)).float().to(device)
    white_heatmaps = torch.from_numpy(white_heatmaps_np).float().to(device)
    background_heatmaps = torch.from_numpy(background_heatmaps_np).float().to(device)

    white_results = evaluate_ranked_masks(
        controller, images, white_heatmaps, background, args.mask_fraction, args.seed
    )
    background_results = evaluate_ranked_masks(
        controller, images, background_heatmaps, background, args.mask_fraction, args.seed
    )

    rng = np.random.default_rng(args.seed)
    count = min(args.num_preview, images_np.shape[0])
    indices = np.sort(rng.choice(images_np.shape[0], size=count, replace=False))
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    save_preview(
        images_np,
        white_heatmaps_np,
        background_heatmaps_np,
        states,
        indices,
        args.output_image,
    )

    arrays = {
        "split": np.array(args.split),
        "mask_fraction": np.array(args.mask_fraction, dtype=np.float32),
        "causal_fill": np.array("background_median"),
        "preview_indices": indices,
    }
    for name, results in (("white", white_results), ("background_median", background_results)):
        for key, value in results.items():
            arrays[f"{name}_{key}"] = value.astype(np.float32)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_metrics, **arrays)

    for name, results in (("white", white_results), ("background_median", background_results)):
        summary = ", ".join(
            f"{kind}={results[f'{kind}_delta'].mean():.6f}"
            for kind in ("top", "bottom", "random")
        )
        print(f"[{name}] {summary}")
    print(f"[saved] {args.output_image}")
    print(f"[saved] {args.output_metrics}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
