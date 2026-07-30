#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train predictor_transformer.pth from real_trajectories.npz.

Edit config.py, then run:

    python train_predictor.py

When train_traj exists, only that split is used to optimize or select the
checkpoint.  For a legacy archive without train_traj, a deterministic part of
val_traj is used for training and the remaining indices stay reserved for
conformal calibration.  test_traj is always reserved for final evaluation.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config
from predictor_model import TrajectoryTransformer, trajectory_loss


TRAJECTORY_KEYS = ("train_traj", "val_traj", "test_traj")
ANGLE_DIM = 0


# =============================================================================
# 1. Runtime and configuration
# =============================================================================


def absolute(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def uses_periodic_angle(environment: str) -> bool:
    return environment.strip().lower() == "pendulum"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config() -> Tuple[str, Path, Path]:
    environment = str(config.ENVIRONMENT).strip().lower()
    if environment not in {
        "pendulum",
        "mountain_car",
        "cartpole",
        "brake_system",
    }:
        raise ValueError(f"unsupported ENVIRONMENT={environment!r}")

    real_path = absolute(config.REAL_TRAJECTORIES)
    checkpoint_path = absolute(config.CHECKPOINT)
    if not real_path.is_file():
        raise FileNotFoundError(
            f"real trajectory input does not exist: {real_path}"
        )
    if checkpoint_path.suffix.lower() != ".pth":
        raise ValueError("CHECKPOINT must end with .pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if int(config.HORIZON) < 1:
        raise ValueError("HORIZON must be positive")
    if not 0.0 < float(config.TRAIN_FIT_RATIO) < 1.0:
        raise ValueError("TRAIN_FIT_RATIO must be between 0 and 1")
    if int(config.TRAIN_EPOCHS) < 1:
        raise ValueError("TRAIN_EPOCHS must be positive")
    if int(config.TRAIN_BATCH_SIZE) < 1:
        raise ValueError("TRAIN_BATCH_SIZE must be positive")
    if int(config.EARLY_STOPPING_PATIENCE) < 1:
        raise ValueError("EARLY_STOPPING_PATIENCE must be positive")
    if float(config.LEARNING_RATE) <= 0.0:
        raise ValueError("LEARNING_RATE must be positive")
    if float(config.GRADIENT_CLIP) <= 0.0:
        raise ValueError("GRADIENT_CLIP must be positive")
    if not 0.0 < float(config.DERIVED_TRAIN_RATIO) < 1.0:
        raise ValueError("DERIVED_TRAIN_RATIO must be between 0 and 1")
    if int(config.MIN_DERIVED_CALIBRATION_COUNT) < 1:
        raise ValueError(
            "MIN_DERIVED_CALIBRATION_COUNT must be positive"
        )

    policy = str(config.MISSING_TRAIN_POLICY).strip().lower()
    if policy not in {"error", "split_val"}:
        raise ValueError(
            "MISSING_TRAIN_POLICY must be 'error' or 'split_val'"
        )
    return environment, real_path, checkpoint_path


# =============================================================================
# 2. Trajectory loading and Pendulum preprocessing
# =============================================================================


def array_fingerprint(values: np.ndarray) -> str:
    """Return a stable identity for the source split used by the checkpoint."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_trajectory(
    key: str,
    trajectories: np.ndarray,
    horizon: int,
) -> np.ndarray:
    values = np.asarray(trajectories, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(
            f"{key} must have shape (N, T+1, state_dim), "
            f"got {values.shape}"
        )
    if values.shape[1] < horizon + 1:
        raise ValueError(
            f"{key} has only {values.shape[1] - 1} transitions, "
            f"but HORIZON={horizon}"
        )
    if len(values) == 0:
        raise ValueError(f"{key} is empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{key} contains NaN or Inf")
    return values[:, : horizon + 1, :].copy()


def load_trajectory_splits(
    path: Path,
    horizon: int,
    policy: str,
    derived_train_ratio: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        for key in ("val_traj", "test_traj"):
            if key not in data.files:
                raise KeyError(f"{path} is missing required key {key}")

        source_val_traj = np.asarray(data["val_traj"])
        val_traj = _validate_trajectory(
            "val_traj", source_val_traj, horizon
        )
        source_val_count = len(val_traj)
        source_val_fingerprint = array_fingerprint(
            source_val_traj[:, : horizon + 1, :]
        )
        test_traj = _validate_trajectory(
            "test_traj", data["test_traj"], horizon
        )

        if "train_traj" in data.files:
            train_traj = _validate_trajectory(
                "train_traj", data["train_traj"], horizon
            )
            split_strategy = "independent_train"
            derived_train_indices = np.empty(0, dtype=np.int64)
            calibration_indices = np.arange(
                source_val_count,
                dtype=np.int64,
            )
        elif policy == "error":
            raise KeyError(
                f"{path} has no train_traj. Training requires train_traj; "
                "use MISSING_TRAIN_POLICY='split_val' only for a legacy "
                "dataset that intentionally lacks it."
            )
        else:
            if len(val_traj) < 2:
                raise ValueError(
                    "val_traj needs at least two trajectories for split_val"
                )
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(val_traj))
            count = int(len(val_traj) * derived_train_ratio)
            count = max(1, min(len(val_traj) - 1, count))
            derived_train_indices = np.sort(
                indices[:count]
            ).astype(np.int64)
            calibration_indices = np.sort(
                indices[count:]
            ).astype(np.int64)
            if (
                len(calibration_indices)
                < int(config.MIN_DERIVED_CALIBRATION_COUNT)
            ):
                raise ValueError(
                    "split_val leaves only "
                    f"{len(calibration_indices)} calibration trajectories; "
                    "lower DERIVED_TRAIN_RATIO or "
                    "MIN_DERIVED_CALIBRATION_COUNT"
                )
            train_traj = val_traj[derived_train_indices].copy()
            val_traj = val_traj[calibration_indices].copy()
            split_strategy = "split_val"

    state_shapes = {
        key: value.shape[1:]
        for key, value in {
            "train_traj": train_traj,
            "val_traj": val_traj,
            "test_traj": test_traj,
        }.items()
    }
    if len(set(state_shapes.values())) != 1:
        raise ValueError(
            f"trajectory split dimensions do not match: {state_shapes}"
        )
    splits = {
        "train_traj": train_traj,
        "val_traj": val_traj,
        "test_traj": test_traj,
    }
    split_protocol = {
        "strategy": split_strategy,
        "source_val_count": int(source_val_count),
        "source_val_fingerprint": source_val_fingerprint,
        "derived_train_indices": derived_train_indices.tolist(),
        "calibration_indices": calibration_indices.tolist(),
    }
    return splits, split_protocol


def unwrap_pendulum(trajectories: np.ndarray) -> np.ndarray:
    values = np.asarray(trajectories, dtype=np.float32).copy()
    values[:, :, ANGLE_DIM] = np.unwrap(
        values[:, :, ANGLE_DIM].astype(np.float64),
        axis=1,
    ).astype(np.float32)
    return values


# =============================================================================
# 3. Fit/selection data and normalization
# =============================================================================


def split_fit_selection(
    train_traj: np.ndarray,
    fit_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(train_traj) < 2:
        raise ValueError("train_traj needs at least two trajectories")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_traj))
    count = int(len(train_traj) * fit_ratio)
    count = max(1, min(len(train_traj) - 1, count))
    return (
        train_traj[indices[:count]].copy(),
        train_traj[indices[count:]].copy(),
    )


def compute_normalization(
    fit_traj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    flat = fit_traj.reshape(-1, fit_traj.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    return mean, std


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        trajectories: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        normalized = (
            trajectories - mean[None, None, :]
        ) / std[None, None, :]
        self.values = torch.from_numpy(
            normalized.astype(np.float32)
        )

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, index: int):
        trajectory = self.values[index]
        return trajectory[0], trajectory


# =============================================================================
# 4. Optimization and checkpoint selection
# =============================================================================


@torch.no_grad()
def evaluate(
    model: TrajectoryTransformer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for initial_states, targets in loader:
        initial_states = initial_states.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        loss = trajectory_loss(
            model(initial_states),
            targets,
            float(config.TERMINAL_LOSS_WEIGHT),
        )
        total += float(loss.item()) * len(initial_states)
        count += len(initial_states)
    if count == 0:
        raise ValueError("checkpoint-selection DataLoader is empty")
    return total / count


def model_config() -> Dict[str, object]:
    return {
        "d_model": int(config.D_MODEL),
        "nhead": int(config.NHEAD),
        "num_layers": int(config.NUM_LAYERS),
        "dim_feedforward": int(config.DIM_FEEDFORWARD),
        "dropout": float(config.DROPOUT),
    }


def main() -> None:
    environment, real_path, checkpoint_path = validate_config()
    seed = int(config.TRAIN_SEED)
    set_seed(seed)
    device = resolve_device(str(config.DEVICE))

    policy = str(config.MISSING_TRAIN_POLICY).strip().lower()
    splits, split_protocol = load_trajectory_splits(
        real_path,
        int(config.HORIZON),
        policy,
        float(config.DERIVED_TRAIN_RATIO),
        seed,
    )
    angle_representation = "linear"
    angle_dim = None
    if uses_periodic_angle(environment):
        splits = {
            key: unwrap_pendulum(values)
            for key, values in splits.items()
        }
        angle_representation = "unwrapped_theta"
        angle_dim = ANGLE_DIM

    fit_traj, selection_traj = split_fit_selection(
        splits["train_traj"],
        float(config.TRAIN_FIT_RATIO),
        seed,
    )
    mean, std = compute_normalization(fit_traj)

    fit_loader = DataLoader(
        TrajectoryDataset(fit_traj, mean, std),
        batch_size=int(config.TRAIN_BATCH_SIZE),
        shuffle=True,
        num_workers=int(config.NUM_WORKERS),
        pin_memory=device.type == "cuda",
    )
    selection_loader = DataLoader(
        TrajectoryDataset(selection_traj, mean, std),
        batch_size=int(config.TRAIN_BATCH_SIZE),
        shuffle=False,
        num_workers=int(config.NUM_WORKERS),
        pin_memory=device.type == "cuda",
    )

    state_dim = int(fit_traj.shape[2])
    horizon = int(fit_traj.shape[1] - 1)
    architecture = model_config()
    model = TrajectoryTransformer(
        state_dim,
        horizon,
        **architecture,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.LEARNING_RATE),
        weight_decay=float(config.WEIGHT_DECAY),
    )

    print("========== Predictor training ==========")
    print(f"environment        : {environment}")
    print(f"device             : {device}")
    print(f"real trajectories  : {real_path}")
    print(f"checkpoint output  : {checkpoint_path}")
    print(f"horizon            : {horizon}")
    print(f"angle representation: {angle_representation}")
    print(f"fit trajectories   : {fit_traj.shape}")
    print(f"selection          : {selection_traj.shape}")
    print(
        f"calibration held out: {splits['val_traj'].shape}"
    )
    if split_protocol["strategy"] == "split_val":
        print(
            "source val split    : "
            f"{split_protocol['source_val_count']} total = "
            f"{len(split_protocol['derived_train_indices'])} training + "
            f"{len(split_protocol['calibration_indices'])} calibration"
        )
    print(f"test held out      : {splits['test_traj'].shape}")
    print(
        f"parameters         : "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    best_loss = math.inf
    best_epoch = -1
    best_state = None
    stale_epochs = 0

    for epoch in range(1, int(config.TRAIN_EPOCHS) + 1):
        model.train()
        train_total = 0.0
        train_count = 0

        for initial_states, targets in fit_loader:
            initial_states = initial_states.to(
                device,
                non_blocking=True,
            )
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = trajectory_loss(
                model(initial_states),
                targets,
                float(config.TERMINAL_LOSS_WEIGHT),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "training produced NaN or Inf"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config.GRADIENT_CLIP),
            )
            optimizer.step()

            train_total += float(loss.item()) * len(initial_states)
            train_count += len(initial_states)

        train_loss = train_total / train_count
        selection_loss = evaluate(
            model,
            selection_loader,
            device,
        )
        improved = (
            selection_loss
            < best_loss - float(config.EARLY_STOPPING_MIN_DELTA)
        )
        print(
            f"epoch {epoch:04d}/{int(config.TRAIN_EPOCHS)} | "
            f"fit={train_loss:.8f} | "
            f"selection={selection_loss:.8f}"
            f"{' *' if improved else ''}"
        )

        if improved:
            best_loss = selection_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(
                config.EARLY_STOPPING_PATIENCE
            ):
                print(
                    "[Early stop] no improvement for "
                    f"{config.EARLY_STOPPING_PATIENCE} epochs"
                )
                break

    if best_state is None:
        raise RuntimeError(
            "training finished without a valid checkpoint"
        )

    checkpoint = {
        "model_state_dict": best_state,
        "state_mean": torch.from_numpy(mean),
        "state_std": torch.from_numpy(std),
        "state_dim": state_dim,
        "horizon": horizon,
        "model_config": architecture,
        "best_epoch": int(best_epoch),
        "best_selection_loss": float(best_loss),
        "source_real_trajectories": str(real_path),
        "environment": environment,
        "angle_representation": angle_representation,
        "angle_dim": angle_dim,
        "data_split": split_protocol,
        "training_protocol": {
            "fit_source": (
                "fit subset of derived val training subset"
                if split_protocol["strategy"] == "split_val"
                else "fit subset of train_traj"
            ),
            "selection_source": (
                "held-out subset of derived val training subset"
                if split_protocol["strategy"] == "split_val"
                else "held-out subset of train_traj"
            ),
            "calibration_source": (
                "disjoint held-out subset of val_traj"
                if split_protocol["strategy"] == "split_val"
                else "val_traj"
            ),
            "evaluation_source": "test_traj",
            "fit_ratio": float(config.TRAIN_FIT_RATIO),
            "split_seed": seed,
            "missing_train_policy": policy,
            "derived_train_ratio": float(
                config.DERIVED_TRAIN_RATIO
            ),
            "test_used_for_training": False,
            "val_used_for_training": (
                split_protocol["strategy"] == "split_val"
            ),
            "pendulum_theta_unwrapped_before_training": (
                angle_representation == "unwrapped_theta"
            ),
        },
    }
    temporary = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".training"
    )
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, checkpoint_path)
    finally:
        temporary.unlink(missing_ok=True)

    print("\n========== Best predictor ==========")
    print(f"best epoch          : {best_epoch}")
    print(f"best selection loss : {best_loss:.8f}")
    print(f"[Saved] {checkpoint_path}")


if __name__ == "__main__":
    main()
