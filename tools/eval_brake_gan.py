"""Score brake cGAN checkpoints against the two things a usable baseline needs.

A generator that ignores its state input still scores well on naive losses, so
the yardsticks here are explicit:

  test L1        must beat the constant mean-image baseline, otherwise the model
                 is worse than emitting the dataset average and has learned no
                 conditional mapping.
  across-state   per-pixel std over a state sweep. Both earlier brake
  std            checkpoints scored exactly 0.0 here -- generated frames did not
                 move with (d, v) at all -- which is the failure this metric is
                 meant to catch.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import G_MLP

DATASET = PROJECT_ROOT / "datasets/brake_system/data/dataset_v1/decoder_states.npz"


def sweep_states(n_dis: int = 25, n_vel: int = 5) -> torch.Tensor:
    """States spanning the braking system's verification region."""
    dis = np.linspace(0.3, 6.4, n_dis)
    vel = np.linspace(6.0, 6.4, n_vel)
    return torch.tensor([[d, v] for d in dis for v in vel], dtype=torch.float32)


def score(weights: Path, z_range: float, dataset: Path = DATASET) -> dict:
    data = np.load(dataset)
    states = torch.from_numpy(data["test_states"])
    images = torch.from_numpy(data["test_images"])

    generator = G_MLP(z_range=z_range)
    generator.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    generator.eval()

    zeros = torch.zeros(len(states), generator.latent_dim)
    with torch.no_grad():
        pred = generator(states, zeros)
        sweep = generator(sweep_states(), torch.zeros(125, generator.latent_dim)).squeeze(1)

    mean_image = images.mean(0, keepdim=True)
    return {
        "test_l1": float((pred - images).abs().mean()),
        "baseline_l1": float((mean_image - images).abs().mean()),
        "across_state_std": float(sweep.std(0).mean()),
        "target_across_state_std": float(images.std(0).mean()),
        "image_mean": float(sweep.mean()),
        "within_image_std": float(sweep.std(dim=(1, 2)).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path, nargs="+")
    parser.add_argument("--z-range", type=float, default=0.05)
    args = parser.parse_args()

    rows = [(w, score(w, args.z_range)) for w in args.weights]
    baseline = rows[0][1]["baseline_l1"]
    target_std = rows[0][1]["target_across_state_std"]
    print(f"targets: test_l1 < {baseline:.4f} (const mean-image)   across_state_std -> {target_std:.4f}\n")
    print(f"{'run':<24} {'test_l1':>9} {'std(state)':>11} {'img_mean':>9} {'PASS':>6}")
    for weights, s in rows:
        ok = s["test_l1"] < baseline and s["across_state_std"] > 0.5 * target_std
        name = weights.parent.name
        print(
            f"{name:<24} {s['test_l1']:>9.4f} {s['across_state_std']:>11.4f} "
            f"{s['image_mean']:>9.4f} {'yes' if ok else 'no':>6}"
        )


if __name__ == "__main__":
    main()
