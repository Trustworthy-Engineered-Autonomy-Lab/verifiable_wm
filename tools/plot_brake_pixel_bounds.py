"""Per-pixel reachable bounds for the AEBS decoder and the cGAN baseline.

Lays out one block per model -- lower bound, upper bound, interval width -- in
the style of paper/Figures/pendulum_compare.pdf: hot colormap throughout, 0-255
grey levels, one colorbar per panel.

Both models see the *same* initial state box (one cell of the verification grid
in config/starv_verification/brake_system.json). The cGAN additionally carries
its latent as the interval [-z_range, z_range], which is where its extra width
comes from.

Two ways to resolve the bounds:

  --mode starv     ImageStar reachability, two LPs per pixel. Sound, and what
                   the paper's pixel-interval figures report. Slow.
  --mode sampled   Monte-Carlo over the same box. An under-approximation, so
                   label it as sampled wherever it is used. Seconds.
"""

import argparse
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Checkpoints are large binaries kept outside the repo; dwm_weight/ points
# at wherever they live.
SHARED_WEIGHTS = PROJECT_ROOT / "dwm_weight"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# One cell of the verification grid (6.00..6.40 split into 40 -> 0.01 wide).
STATE_LB = np.array([6.20, 6.20])
STATE_UB = np.array([6.21, 6.21])

DWM_WEIGHTS = SHARED_WEIGHTS / "brake_system/decoder.pth"
CGAN_WEIGHTS = SHARED_WEIGHTS / "brake_system/g_mlp/G_brake.pth"
Z_RANGE = 0.05
NUM_SAMPLES = 4000
OUTPUT_PATH = PROJECT_ROOT / "results/brake_system/brake_pixel_bounds.png"


def bounds_sampled(seed=0):
    """Per-pixel min/max over draws from the state box (and the latent box)."""
    import model

    generator = model.G_MLP(z_range=Z_RANGE)
    generator.load_state_dict(torch.load(CGAN_WEIGHTS, map_location="cpu", weights_only=True))
    decoder = model.Decoder(output_activation="clamp")
    decoder.load_state_dict(torch.load(DWM_WEIGHTS, map_location="cpu"))
    generator.eval()
    decoder.eval()

    rng = np.random.default_rng(seed)
    states = torch.tensor(
        rng.uniform(STATE_LB, STATE_UB, size=(NUM_SAMPLES, 2)), dtype=torch.float32
    )
    torch.manual_seed(seed)
    latent = torch.empty(NUM_SAMPLES, 2).uniform_(-Z_RANGE, Z_RANGE)

    with torch.no_grad():
        dwm = decoder(states).squeeze(1).numpy()
        cgan = generator(states, latent).squeeze(1).numpy()
    return [("DWM", dwm.min(0), dwm.max(0)), ("cGAN Baseline", cgan.min(0), cgan.max(0))]


def bounds_starv(lp_solver):
    """Per-pixel bounds from StarV ImageStar reachability."""
    from starv_verification.model import Decoder, G_MLP

    bound = np.stack([STATE_LB, STATE_UB])
    out = []
    for name, net in (
        ("DWM", Decoder(weights=DWM_WEIGHTS, output_activation="clamp", lp_solver=lp_solver)),
        ("cGAN Baseline", G_MLP(weights=CGAN_WEIGHTS, z_range=Z_RANGE, lp_solver=lp_solver)),
    ):
        started = time.time()
        lb, ub = net(bound).getRanges(lp_solver)
        side = int(round(np.sqrt(lb.size)))
        print(f"[{name}] reach + {2 * lb.size} LPs in {time.time() - started:.1f}s")
        out.append((name, lb.reshape(side, side), ub.reshape(side, side)))
    return out


def plot(blocks, output_path, subtitle):
    """One 1x3 block per model, stacked, matching the paper's figure style."""
    n = len(blocks)
    figure = plt.figure(figsize=(11, 4.6 * n))
    outer = figure.add_gridspec(n, 1, hspace=0.30)

    for i, (name, lb, ub) in enumerate(blocks):
        width = ub - lb
        print(f"{name:<14} width mean={width.mean():.4f} max={width.max():.4f} sum={width.sum():.1f}")

        block = outer[i].subgridspec(1, 3, wspace=0.42)
        for j, (title, data) in enumerate(
            (
                ("Pixel-level Lower Bound", lb),
                ("Pixel-level Upper Bound", ub),
                ("Pixel-level Interval Width", width),
            )
        ):
            axis = figure.add_subplot(block[0, j])
            kwargs = dict(cmap="hot", vmin=0, vmax=255) if j < 2 else dict(cmap="hot", vmin=0)
            handle = axis.imshow(data * 255.0, **kwargs)
            axis.set_title(title, fontsize=12, fontweight="bold")
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)

        anchor = figure.add_subplot(outer[i], frameon=False)
        anchor.set_xticks([])
        anchor.set_yticks([])
        anchor.set_title(name, fontsize=19, fontweight="bold", pad=34)

    figure.suptitle(subtitle, fontsize=11, y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"[saved] {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sampled", "starv"), default="sampled")
    parser.add_argument("--lp-solver", default="gurobi")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    box = (
        f"$d,v\\in[{STATE_LB[0]:.2f}, {STATE_UB[0]:.2f}]$, "
        f"cGAN latent $z\\in[\\pm{Z_RANGE}]$"
    )
    if args.mode == "sampled":
        blocks = bounds_sampled()
        subtitle = f"AEBS, {box} - sampled ({NUM_SAMPLES} draws, under-approximation)"
    else:
        blocks = bounds_starv(args.lp_solver)
        subtitle = f"AEBS, {box} - StarV ImageStar bounds"
    plot(blocks, args.output, subtitle)


if __name__ == "__main__":
    main()
