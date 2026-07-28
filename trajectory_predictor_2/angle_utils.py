#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Periodic-angle helpers used by the Pendulum predictor."""

from __future__ import annotations

from typing import List

import numpy as np


ANGLE_DIM = 0
PI = float(np.pi)
TWO_PI = float(2.0 * np.pi)
ANGLE_EPS = 1e-7


def uses_periodic_angle(environment: str) -> bool:
    """Return whether the environment uses theta in state dimension zero."""
    return environment.strip().lower() == "pendulum"


def unwrap_angle_trajectories(
    trajectories: np.ndarray,
    angle_dim: int = ANGLE_DIM,
) -> np.ndarray:
    """Make theta continuous along time without changing each initial theta."""
    values = np.asarray(trajectories, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(
            "trajectories must have shape (N, T+1, state_dim), "
            f"got {values.shape}"
        )
    if not 0 <= angle_dim < values.shape[2]:
        raise ValueError(f"invalid angle_dim={angle_dim} for shape {values.shape}")

    result = values.copy()
    result[:, :, angle_dim] = np.unwrap(
        result[:, :, angle_dim].astype(np.float64),
        axis=1,
    ).astype(np.float32)
    return result


def wrap_angles(values: np.ndarray) -> np.ndarray:
    """Map angles to the canonical interval [-pi, pi)."""
    array = np.asarray(values)
    return (np.remainder(array + PI, TWO_PI) - PI).astype(
        array.dtype,
        copy=False,
    )


def wrap_angle_trajectories(
    trajectories: np.ndarray,
    angle_dim: int = ANGLE_DIM,
) -> np.ndarray:
    """Copy trajectories and wrap their theta component to [-pi, pi)."""
    values = np.asarray(trajectories, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError(f"trajectory array needs at least 2 dimensions: {values.shape}")
    if not 0 <= angle_dim < values.shape[-1]:
        raise ValueError(f"invalid angle_dim={angle_dim} for shape {values.shape}")

    result = values.copy()
    result[..., angle_dim] = wrap_angles(result[..., angle_dim])
    return result


def periodic_interval_to_json(lower: float, upper: float) -> List[float]:
    """Convert one unwrapped interval to one or two intervals in [-pi, pi].

    A non-crossing interval is returned as ``[low, high]``.  An interval that
    crosses the branch cut is returned as
    ``[low, pi, -pi, high]``, which compare.py interprets as a union.
    """
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("angle interval contains NaN or Inf")
    if lower > upper:
        raise ValueError(f"angle lower={lower} exceeds upper={upper}")
    if upper - lower >= TWO_PI - ANGLE_EPS:
        return [-PI, PI]

    lower_branch = int(np.floor((lower + PI) / TWO_PI))
    # A closed upper endpoint at +pi belongs to the branch on its left, except
    # for a zero-width interval that should remain a single point.
    upper_for_branch = (
        upper
        if upper == lower
        else float(np.nextafter(upper, -np.inf))
    )
    upper_branch = int(np.floor((upper_for_branch + PI) / TWO_PI))

    wrapped_lower = lower - lower_branch * TWO_PI
    wrapped_upper = upper - upper_branch * TWO_PI
    wrapped_lower = float(np.clip(wrapped_lower, -PI, PI))
    wrapped_upper = float(np.clip(wrapped_upper, -PI, PI))

    if lower_branch == upper_branch:
        return [wrapped_lower, wrapped_upper]
    return [wrapped_lower, PI, -PI, wrapped_upper]
