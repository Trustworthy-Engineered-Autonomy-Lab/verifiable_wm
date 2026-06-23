import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import Decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("decoder_result.png"))
    parser.add_argument("--num", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.dataset)
    states = torch.tensor(data["test_states"][:args.num], dtype=torch.float32).to(device)
    true_images = torch.tensor(data["test_images"][:args.num], dtype=torch.float32)

    model = Decoder().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    with torch.no_grad():
        pred_images = model(states).cpu()

    fig, axes = plt.subplots(2, args.num, figsize=(args.num * 2, 4))

    for i in range(args.num):
        axes[0, i].imshow(true_images[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[0, i].set_title("real")

        axes[1, i].imshow(pred_images[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        axes[1, i].set_title("decoder")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    print(f"[Save] result image saved to {args.output}")


if __name__ == "__main__":
    main()