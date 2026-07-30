"""Convert the CARLA AEBS capture into the repo's decoder_states.npz format.

make_decoder_dataset.py renders on-line from a gym env, which is not an option
for the braking system: its frames come from CARLA and were captured offline
into aebs_carla/dataset_decoder.npz. That file stores uint8 NHWC frames, while
every other benchmark's decoder_states.npz stores float32 NCHW images already
scaled to [0,1] -- which is what train_gan.py and train_decoder.py assume. The
missing /255 here is exactly what made G_brake.pth collapse to a black image:
the discriminator saw "real" pixels up to 229 against a generator clamped to 1.
"""

import argparse
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "aebs_carla/dataset_decoder.npz"
OUTPUT = PROJECT_ROOT / "datasets/brake_system/data/dataset_v1/decoder_states.npz"


def to_unit_nchw(images: np.ndarray) -> np.ndarray:
    """Scale uint8 NHWC frames to float32 NCHW in [0,1]."""
    if images.ndim != 4 or images.shape[-1] != 1:
        raise ValueError(f"expected (N,H,W,1) images, got {images.shape}")
    scaled = images.astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(scaled, (0, 3, 1, 2)))


def split_indices(n: int, val_frac: float, test_frac: float, seed: int):
    """Return shuffled train/val/test index arrays."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("val_frac + test_frac leave no training data")
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def build(source: Path, output: Path, val_frac: float, test_frac: float, seed: int) -> dict:
    data = np.load(source)
    states = data["states"].astype(np.float32)
    images = to_unit_nchw(data["images"])
    if states.shape[0] != images.shape[0]:
        raise ValueError("states and images disagree on sample count")

    train_idx, val_idx, test_idx = split_indices(states.shape[0], val_frac, test_frac, seed)
    splits = {
        "train_states": states[train_idx],
        "train_images": images[train_idx],
        "val_states": states[val_idx],
        "val_images": images[val_idx],
        "test_states": states[test_idx],
        "test_images": images[test_idx],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **splits)
    return {k: v.shape for k, v in splits.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    shapes = build(args.source, args.output, args.val_frac, args.test_frac, args.seed)
    for name, shape in shapes.items():
        print(f"  {name}: {shape}")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
