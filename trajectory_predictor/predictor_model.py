#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transformer model, checkpoint loading, and batched prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryTransformer(nn.Module):
    """Direct predictor F_theta(s0) -> [s0, s1, ..., sT]."""

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
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if horizon < 1:
            raise ValueError("horizon must be positive")

        self.state_dim = int(state_dim)
        self.horizon = int(horizon)

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.time_queries = nn.Parameter(
            torch.empty(1, horizon + 1, d_model)
        )
        nn.init.normal_(self.time_queries, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_head = nn.Linear(d_model, state_dim)

    def forward(self, initial_states: torch.Tensor) -> torch.Tensor:
        if initial_states.ndim != 2 or initial_states.shape[1] != self.state_dim:
            raise ValueError(
                f"initial_states must have shape (B, {self.state_dim}), "
                f"got {tuple(initial_states.shape)}"
            )

        context = self.state_encoder(initial_states).unsqueeze(1)
        tokens = self.time_queries.expand(len(initial_states), -1, -1) + context
        features = self.transformer(tokens)
        prediction = self.output_head(features) + initial_states.unsqueeze(1)

        # The initial state is known exactly.
        return torch.cat(
            [initial_states.unsqueeze(1), prediction[:, 1:, :]],
            dim=1,
        )


def trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    terminal_weight: float,
) -> torch.Tensor:
    future_mse = F.mse_loss(prediction[:, 1:, :], target[:, 1:, :])
    terminal_mse = F.mse_loss(prediction[:, -1, :], target[:, -1, :])
    return future_mse + float(terminal_weight) * terminal_mse


def tensor_to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return (
            value.detach().cpu().numpy().astype(np.float32, copy=False)
        )
    return np.asarray(value, dtype=np.float32)


def load_predictor_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[TrajectoryTransformer, np.ndarray, np.ndarray, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    model = TrajectoryTransformer(
        state_dim=int(checkpoint["state_dim"]),
        horizon=int(checkpoint["horizon"]),
        **dict(checkpoint["model_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = tensor_to_numpy(checkpoint["state_mean"])
    std = tensor_to_numpy(checkpoint["state_std"])
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
    initial_states = np.asarray(initial_states, dtype=np.float32)
    if initial_states.ndim != 2 or initial_states.shape[1] != model.state_dim:
        raise ValueError(
            f"initial_states must have shape (N, {model.state_dim}), "
            f"got {initial_states.shape}"
        )

    normalized = (initial_states - mean[None, :]) / std[None, :]
    batches = []

    for start in range(0, len(normalized), batch_size):
        batch = torch.from_numpy(normalized[start : start + batch_size]).to(device)
        batches.append(model(batch).cpu().numpy())

    predicted_normalized = np.concatenate(batches, axis=0)
    predicted = (
        predicted_normalized * std[None, None, :] + mean[None, None, :]
    ).astype(np.float32)
    predicted[:, 0, :] = initial_states

    if not np.isfinite(predicted).all():
        raise RuntimeError("prediction contains NaN or Inf")
    return predicted
