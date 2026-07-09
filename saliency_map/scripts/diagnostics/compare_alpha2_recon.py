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

from model import Decoder


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = REPO_ROOT / "datasets/cartpole/data/dataset_v1"
    data = np.load(dataset_dir / "states.npz")
    test_states = torch.from_numpy(data["test_states"]).float().to(device)
    test_images = data["test_images"]

    intensity = Decoder().to(device).eval()
    intensity.load_state_dict(load_state_dict(
        REPO_ROOT / "dwm_weight/now_weight/cartpole/intensity/decoder_last.pth", device
    ))

    saliency_a2 = Decoder().to(device).eval()
    saliency_a2.load_state_dict(load_state_dict(
        REPO_ROOT / "dwm_weight/now_weight/cartpole/saliency_alpha_sweep/alpha_2/seed_2025/decoder_last.pth", device
    ))

    saliency_a8 = Decoder().to(device).eval()
    saliency_a8.load_state_dict(load_state_dict(
        REPO_ROOT / "dwm_weight/now_weight/cartpole/saliency/decoder_last.pth", device
    ))

    rng = np.random.default_rng(0)
    idx = rng.choice(len(test_states), size=6, replace=False)
    idx.sort()

    with torch.no_grad():
        recon_intensity = intensity(test_states[idx]).cpu().numpy()
        recon_a2 = saliency_a2(test_states[idx]).cpu().numpy()
        recon_a8 = saliency_a8(test_states[idx]).cpu().numpy()

    rows = [
        ("real", test_images[idx]),
        ("intensity (paper baseline)", recon_intensity),
        ("saliency alpha=2", recon_a2),
        ("saliency alpha=8 (mainline)", recon_a8),
    ]

    fig, axes = plt.subplots(len(rows), len(idx), figsize=(2 * len(idx), 2 * len(rows)))
    for r, (label, imgs) in enumerate(rows):
        for c in range(len(idx)):
            ax = axes[r, c]
            ax.imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            if r == 0:
                ax.set_title(f"test idx {idx[c]}", fontsize=9)

    fig.tight_layout()
    out_path = REPO_ROOT / "saliency_map/output/diagnostics/previews/cartpole_alpha2_vs_intensity.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
