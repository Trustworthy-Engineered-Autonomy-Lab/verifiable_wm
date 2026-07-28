#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transformer model structure, checkpoint loading, and batched inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


class TrajectoryTransformer(nn.Module):
    """Direct trajectory predictor: initial state -> [s0, s1, ..., sT]."""

    def __init__(
        self,
        state_dim: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.state_dim = int(state_dim)
        self.horizon = int(horizon)

        # Encode the known initial state into the Transformer feature space.
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # One learned query for every state time s0...sT.
        self.time_queries = nn.Parameter(
            torch.empty(1, horizon + 1, d_model)
        )
        nn.init.normal_(self.time_queries, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_head = nn.Linear(d_model, state_dim)

    def forward(self, initial_states: torch.Tensor) -> torch.Tensor:
        if initial_states.ndim != 2:
            raise ValueError(
                "initial_states must have shape "
                f"(batch, {self.state_dim}), got {tuple(initial_states.shape)}"
            )
        if initial_states.shape[1] != self.state_dim:
            raise ValueError(
                f"expected state_dim={self.state_dim}, "
                f"got {initial_states.shape[1]}"
            )

        context = self.state_encoder(initial_states).unsqueeze(1)
        tokens = (
            self.time_queries.expand(initial_states.shape[0], -1, -1)
            + context
        )
        features = self.transformer(tokens)

        # Residual prediction around the initial state.
        predicted = self.output_head(features) + initial_states.unsqueeze(1)

        # State zero is known, so return it exactly instead of predicting it.
        return torch.cat(
            [initial_states.unsqueeze(1), predicted[:, 1:, :]],
            dim=1,
        )


def trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    terminal_weight: float,
) -> torch.Tensor:
    """Mean trajectory MSE with an optional extra terminal-state penalty."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match "
            f"target shape {tuple(target.shape)}"
        )
    if terminal_weight < 0.0:
        raise ValueError("terminal_weight must be non-negative")

    trajectory_mse = torch.mean((prediction - target) ** 2)
    terminal_mse = torch.mean(
        (prediction[:, -1, :] - target[:, -1, :]) ** 2
    )
    return trajectory_mse + float(terminal_weight) * terminal_mse


def _as_float32_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def load_predictor_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[
    TrajectoryTransformer,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    """Load the architecture, weights, and normalization from a checkpoint."""

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch releases.
        checkpoint = torch.load(path, map_location=device)

    required = {
        "state_dim",
        "horizon",
        "model_config",
        "model_state_dict",
        "state_mean",
        "state_std",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"checkpoint is missing keys: {missing}")

    model = TrajectoryTransformer(
        state_dim=int(checkpoint["state_dim"]),
        horizon=int(checkpoint["horizon"]),
        **dict(checkpoint["model_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = _as_float32_numpy(checkpoint["state_mean"])
    std = _as_float32_numpy(checkpoint["state_std"])
    if mean.shape != (model.state_dim,) or std.shape != (model.state_dim,):
        raise ValueError(
            "checkpoint normalization shape does not match model state_dim"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("checkpoint normalization contains NaN or Inf")
    if np.any(std <= 0.0):
        raise ValueError("checkpoint state_std must be positive")

    return model, mean, std, checkpoint


@torch.no_grad()
def predict_trajectories(
    model: TrajectoryTransformer,
    initial_states: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predict physical-state trajectories from physical initial states."""

    states = np.asarray(initial_states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != model.state_dim:
        raise ValueError(
            f"initial_states must have shape (N, {model.state_dim}), "
            f"got {states.shape}"
        )
    if len(states) == 0:
        raise ValueError("initial_states is empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    normalized = (states - mean[None, :]) / std[None, :]
    predicted_batches = []
    model.eval()

    for start in range(0, len(normalized), batch_size):
        batch = torch.from_numpy(
            normalized[start : start + batch_size]
        ).to(device)
        predicted_batches.append(model(batch).cpu().numpy())

    normalized_prediction = np.concatenate(predicted_batches, axis=0)
    prediction = (
        normalized_prediction * std[None, None, :]
        + mean[None, None, :]
    ).astype(np.float32)
    prediction[:, 0, :] = states

    if not np.isfinite(prediction).all():
        raise RuntimeError("model prediction contains NaN or Inf")
    return prediction
