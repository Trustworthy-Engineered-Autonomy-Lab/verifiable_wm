import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import Decoder


def load_dataset(path):
    data = np.load(path)

    train_states = torch.tensor(data["train_states"], dtype=torch.float32)
    train_images = torch.tensor(data["train_images"], dtype=torch.float32)

    val_states = torch.tensor(data["val_states"], dtype=torch.float32)
    val_images = torch.tensor(data["val_images"], dtype=torch.float32)

    return train_states, train_images, val_states, val_images

def weighted_mse_loss(pred, target, foreground_weight=20.0, threshold=0.95):
    foreground = (target < threshold).float()
    weight = 1.0 + foreground_weight * foreground
    return (weight * (pred - target) ** 2).mean()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    train_states, train_images, val_states, val_images = load_dataset(args.dataset)

    print(f"[Train] states={tuple(train_states.shape)}, images={tuple(train_images.shape)}")
    print(f"[Val] states={tuple(val_states.shape)}, images={tuple(val_images.shape)}")

    train_loader = DataLoader(
        TensorDataset(train_states, train_images),
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(val_states, val_images),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = Decoder().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.loss == "mse":
        loss_fn = nn.MSELoss()
    elif args.loss == "weighted_mse":
        loss_fn = lambda pred, target: weighted_mse_loss(
            pred,
            target,
            foreground_weight=args.foreground_weight,
            threshold=args.foreground_threshold,
        )
    else:
        raise ValueError(f"Unsupported loss: {args.loss}")

    best_val_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for states, images in train_loader:
            states = states.to(device)
            images = images.to(device)

            pred_images = model(states)
            loss = loss_fn(pred_images, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * states.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for states, images in val_loader:
                states = states.to(device)
                images = images.to(device)

                pred_images = model(states)
                loss = loss_fn(pred_images, images)

                val_loss += loss.item() * states.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)
            print(f"[Save] best model saved to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "weighted_mse"])
    parser.add_argument("--foreground-weight", type=float, default=20.0)
    parser.add_argument("--foreground-threshold", type=float, default=0.95)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()