#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the point-trajectory Transformer predictor."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_REAL_PATH,
    absolute_path,
    ensure_parent,
    resolve_device,
    set_seed,
)
from data_utils import (
    TrajectoryDataset,
    compute_normalization,
    load_real_trajectories,
    split_fit_selection,
)
from predictor_model import TrajectoryTransformer, trajectory_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Transformer trajectory predictor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--real", type=Path, default=DEFAULT_REAL_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)

    parser.add_argument("--fit-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--terminal-loss-weight", type=float, default=0.2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-8)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.fit_ratio < 1.0:
        raise ValueError("--fit-ratio must be between 0 and 1")
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch-size, and patience must be positive")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("learning-rate and gradient-clip must be positive")

    args.real = absolute_path(args.real)
    args.checkpoint = absolute_path(args.checkpoint)
    if not args.real.exists():
        raise FileNotFoundError(f"real trajectory file does not exist: {args.real}")
    ensure_parent(args.checkpoint)


def model_config(args: argparse.Namespace) -> Dict[str, float | int]:
    return {
        "d_model": int(args.d_model),
        "nhead": int(args.nhead),
        "num_layers": int(args.num_layers),
        "dim_feedforward": int(args.dim_feedforward),
        "dropout": float(args.dropout),
    }


@torch.no_grad()
def evaluate(
    model: TrajectoryTransformer,
    loader: DataLoader,
    device: torch.device,
    terminal_weight: float,
) -> float:
    model.eval()
    total = 0.0
    count = 0

    for initial_states, targets in loader:
        initial_states = initial_states.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        loss = trajectory_loss(model(initial_states), targets, terminal_weight)
        total += float(loss.item()) * len(initial_states)
        count += len(initial_states)

    if count == 0:
        raise ValueError("selection DataLoader is empty")
    return total / count


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    splits = load_real_trajectories(args.real)
    fit_traj, selection_traj = split_fit_selection(
        splits["train_traj"], args.fit_ratio, args.split_seed
    )
    mean, std = compute_normalization(fit_traj)

    fit_loader = DataLoader(
        TrajectoryDataset(fit_traj, mean, std),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    selection_loader = DataLoader(
        TrajectoryDataset(selection_traj, mean, std),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    state_dim = int(fit_traj.shape[2])
    horizon = int(fit_traj.shape[1] - 1)
    config = model_config(args)
    model = TrajectoryTransformer(state_dim, horizon, **config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("========== Predictor training ==========")
    print(f"real file  : {args.real}")
    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {device}")
    print(f"fit        : {fit_traj.shape}")
    print(f"selection  : {selection_traj.shape}")
    print(f"calibration: {splits['val_traj'].shape} (not used during training)")
    print(f"evaluation : {splits['test_traj'].shape} (not used during training)")
    print(f"parameters : {sum(p.numel() for p in model.parameters()):,}")

    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0

        for initial_states, targets in fit_loader:
            initial_states = initial_states.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = trajectory_loss(
                model(initial_states), targets, args.terminal_loss_weight
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("training produced NaN or Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            train_total += float(loss.item()) * len(initial_states)
            train_count += len(initial_states)

        train_loss = train_total / train_count
        selection_loss = evaluate(
            model, selection_loader, device, args.terminal_loss_weight
        )
        improved = selection_loss < best_loss - args.min_delta
        print(
            f"epoch {epoch:04d}/{args.epochs} | "
            f"fit={train_loss:.8f} | selection={selection_loss:.8f}"
            f"{' *' if improved else ''}"
        )

        if improved:
            best_loss = selection_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_mean": torch.from_numpy(mean),
                    "state_std": torch.from_numpy(std),
                    "state_dim": state_dim,
                    "horizon": horizon,
                    "model_config": config,
                    "best_epoch": int(epoch),
                    "best_selection_loss": float(selection_loss),
                    "source_real_trajectories": str(args.real),
                    "training_protocol": {
                        "fit_source": "fit part of train_traj",
                        "selection_source": "held-out part of train_traj",
                        "calibration_source": "val_traj",
                        "evaluation_source": "test_traj",
                        "fit_ratio": float(args.fit_ratio),
                        "split_seed": int(args.split_seed),
                    },
                },
                args.checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"[Early stop] no improvement for {args.patience} epochs")
                break

    if not args.checkpoint.exists():
        raise RuntimeError("training finished without creating a checkpoint")

    print("========== Best predictor ==========")
    print(f"best epoch          : {best_epoch}")
    print(f"best selection loss : {best_loss:.8f}")
    print(f"saved checkpoint    : {args.checkpoint}")


if __name__ == "__main__":
    main()
