import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import G_MLP
from utils import load_config, resolve_device, set_seed


class Discriminator(nn.Module):
    """Training-only critic: (flat image, state) -> real/fake probability.

    Matches cGAN_mlp/model_2.py's Discriminator. Not needed for rollout or
    StarV verification, so it stays local to this script instead of model.py.
    """

    def __init__(self, img_dim=96 * 96, state_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim + state_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img_flat, state):
        return self.net(torch.cat([img_flat, state], dim=1))


def load_split(dataset_dir, split):
    data = np.load(dataset_dir / "decoder_states.npz")
    states = torch.from_numpy(data[f"{split}_states"]).float()
    images = torch.from_numpy(data[f"{split}_images"]).float()
    return states, images


def sample_latent(batch_size, latent_dim, z_range, device):
    return torch.empty(batch_size, latent_dim, device=device).uniform_(-z_range, z_range)


@torch.no_grad()
def evaluate(generator, discriminator, states, images, z_range, device, batch_size):
    generator.eval()
    discriminator.eval()
    n = states.shape[0]
    d_loss_sum = 0.0

    for i in range(0, n, batch_size):
        s = states[i : i + batch_size].to(device)
        target = images[i : i + batch_size].to(device)
        b = s.shape[0]
        target_flat = target.view(b, -1)

        z = sample_latent(b, generator.latent_dim, z_range, device)
        fake = generator(s, z).view(b, -1)

        real_lbl = torch.ones(b, 1, device=device)
        fake_lbl = torch.zeros(b, 1, device=device)
        loss_real = F.binary_cross_entropy(discriminator(target_flat, s), real_lbl)
        loss_fake = F.binary_cross_entropy(discriminator(fake, s), fake_lbl)

        d_loss_sum += (0.5 * (loss_real + loss_fake)).item() * b

    return {"val_d_loss": d_loss_sum / n}


def train(config, device):
    dataset_dir = Path(config["dataset_dir"])
    train_cfg = config["training"]
    z_range = config["z_range"]

    train_states, train_images = load_split(dataset_dir, "train")
    val_states, val_images = load_split(dataset_dir, "val")
    test_states, test_images = load_split(dataset_dir, "test")

    state_dim = train_states.shape[1]
    generator = G_MLP(state_dim=state_dim, latent_dim=2, z_range=z_range).to(device)
    discriminator = Discriminator(state_dim=state_dim).to(device)

    beta1, beta2 = train_cfg["beta1"], train_cfg["beta2"]
    opt_g = torch.optim.Adam(generator.parameters(), lr=train_cfg["lr_g"], betas=(beta1, beta2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=train_cfg["lr_d"], betas=(beta1, beta2))

    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    g_path = output_dir / config["g_out_name"]
    d_path = output_dir / config["d_out_name"]

    best = {"epoch": -1, "value": float("inf")}
    history = []
    n = train_states.shape[0]
    start_time = time.time()

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        perm = torch.randperm(n)
        d_loss_sum, g_loss_sum = 0.0, 0.0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            s = train_states[idx].to(device)
            target = train_images[idx].to(device)
            b = s.shape[0]
            target_flat = target.view(b, -1)

            real_lbl = torch.ones(b, 1, device=device)
            fake_lbl = torch.zeros(b, 1, device=device)

            # Discriminator: maximize log D(real) + log(1 - D(fake))
            z = sample_latent(b, generator.latent_dim, z_range, device)
            fake_flat = generator(s, z).view(b, -1).detach()
            loss_real = F.binary_cross_entropy(discriminator(target_flat, s), real_lbl)
            loss_fake = F.binary_cross_entropy(discriminator(fake_flat, s), fake_lbl)
            loss_d = 0.5 * (loss_real + loss_fake)

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # Generator: non-saturating loss, encourage D(fake) -> 1
            z = sample_latent(b, generator.latent_dim, z_range, device)
            fake_flat = generator(s, z).view(b, -1)
            loss_g = F.binary_cross_entropy(discriminator(fake_flat, s), real_lbl)

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            d_loss_sum += loss_d.item() * b
            g_loss_sum += loss_g.item() * b

        val_metrics = evaluate(generator, discriminator, val_states, val_images, z_range, device, batch_size)
        record = {
            "epoch": epoch,
            "train_d_loss": d_loss_sum / n,
            "train_g_loss": g_loss_sum / n,
            **val_metrics,
        }
        history.append(record)

        # Best-checkpoint selection follows cGAN_mlp/train_*.ipynb: lowest
        # val D loss. Not a true convergence criterion for a minimax game,
        # but it's the established convention (matches G_brake.pth).
        if val_metrics["val_d_loss"] < best["value"]:
            best = {"epoch": epoch, "value": val_metrics["val_d_loss"]}
            torch.save(generator.state_dict(), g_path)
            torch.save(discriminator.state_dict(), d_path)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"[epoch {epoch:3d}] d_loss={record['train_d_loss']:.4f} "
                f"g_loss={record['train_g_loss']:.4f} "
                f"val_d_loss={record['val_d_loss']:.4f} best@{best['epoch']}"
            )

    generator.load_state_dict(torch.load(g_path, map_location=device, weights_only=True))
    discriminator.load_state_dict(torch.load(d_path, map_location=device, weights_only=True))
    test_metrics = evaluate(generator, discriminator, test_states, test_images, z_range, device, batch_size)
    print(f"[test:{g_path.name}@{best['epoch']}] " + " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))

    results = {
        "config": config,
        "best_epoch": best["epoch"],
        "best_val_d_loss": best["value"],
        "g_checkpoint": g_path.name,
        "d_checkpoint": d_path.name,
        "test": test_metrics,
        "history": history,
        "train_seconds": time.time() - start_time,
    }
    with open(output_dir / "gan_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output_dir.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)

    set_seed(config["training"]["seed"])
    device = resolve_device(config.get("device", "auto"))
    train(config, device)


if __name__ == "__main__":
    main()
