"""Plot an AEBS DWM/cGAN pixel comparison at a fixed state and latent."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import Decoder, G_MLP


STATE = (6.2, 6.2)
LATENT = (0.0, 0.0)
DWM_WEIGHTS = PROJECT_ROOT / "dwm_weight/brake_system/decoder.pth"
CGAN_WEIGHTS = PROJECT_ROOT / "dwm_weight/brake_system/g_mlp/G_brake.pth"
OUTPUT_PATH = PROJECT_ROOT / "paper/Figures/brake_dwm_cgan_pixel_difference.png"


def absolute_difference(dwm_image: np.ndarray, cgan_image: np.ndarray) -> np.ndarray:
    """Return the elementwise absolute pixel difference between equal-shaped images."""
    if dwm_image.shape != cgan_image.shape:
        raise ValueError("DWM and cGAN images must have the same shape")
    return np.abs(dwm_image - cgan_image)


def load_model(model: torch.nn.Module, weights_path: Path) -> torch.nn.Module:
    """Load a checkpoint and place the model in deterministic evaluation mode."""
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.eval()


def render_images() -> tuple[np.ndarray, np.ndarray]:
    """Generate DWM and cGAN images for the fixed AEBS input."""
    state = torch.tensor([STATE], dtype=torch.float32)
    latent = torch.tensor([LATENT], dtype=torch.float32)
    dwm = load_model(Decoder(output_activation="clamp"), DWM_WEIGHTS)
    cgan = load_model(G_MLP(z_range=0.05), CGAN_WEIGHTS)

    with torch.no_grad():
        dwm_image = dwm(state).squeeze().cpu().numpy()
        cgan_image = cgan(state, latent).squeeze().cpu().numpy()
    return dwm_image, cgan_image


def save_comparison(output_path: Path = OUTPUT_PATH) -> float:
    """Write the three-panel comparison and return its maximum pixel difference."""
    dwm_image, cgan_image = render_images()
    difference = absolute_difference(dwm_image, cgan_image)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(dwm_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("DWM")
    axes[1].imshow(cgan_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("cGAN")
    heatmap = axes[2].imshow(difference, cmap="magma", vmin=0.0)
    axes[2].set_title(r"$|I_{\mathrm{DWM}} - I_{\mathrm{cGAN}}|$")
    figure.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("AEBS: state $(d,v)=(6.2,6.2)$; cGAN latent $z=(0,0)$")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)
    return float(difference.max())


if __name__ == "__main__":
    max_difference = save_comparison()
    print(f"Saved {OUTPUT_PATH}")
    print(f"Maximum absolute pixel difference: {max_difference:.6f}")
