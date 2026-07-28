#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic corner, center, and Halton cell sampling."""

from __future__ import annotations

import itertools

import numpy as np


EPS = 1e-10


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def sample_cell(bounds: np.ndarray, samples_per_cell: int = 5) -> np.ndarray:
    """Return all active corners, the center, then Halton interior points."""
    bounds = np.asarray(bounds, dtype=np.float32)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must have shape (state_dim, 2), got {bounds.shape}")
    if samples_per_cell < 2:
        raise ValueError("samples_per_cell must be at least 2")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("cell lower bounds must not exceed upper bounds")

    lower = bounds[:, 0]
    upper = bounds[:, 1]
    widths = upper - lower
    active_dims = np.flatnonzero(widths > EPS)
    corner_count = 1 << len(active_dims)
    required = corner_count + 1
    if samples_per_cell < required:
        raise ValueError(
            f"this cell has {len(active_dims)} non-zero width dimensions and "
            f"requires at least {required} samples ({corner_count} corners + "
            f"center), but samples_per_cell={samples_per_cell}"
        )

    points = []
    for corner_bits in itertools.product((0.0, 1.0), repeat=len(active_dims)):
        point = lower.copy()
        for dim, bit in zip(active_dims, corner_bits):
            point[dim] = lower[dim] + np.float32(bit) * widths[dim]
        points.append(point)
    points.append(lower + np.float32(0.5) * widths)

    primes = (2, 3, 5, 7, 11, 13, 17, 19)
    if len(active_dims) > len(primes):
        raise ValueError("Halton sampling supports at most 8 active dimensions")
    halton_index = 1
    while len(points) < samples_per_cell:
        point = lower.copy()
        for sequence_dim, state_dim in enumerate(active_dims):
            fraction = _van_der_corput(halton_index, primes[sequence_dim])
            point[state_dim] = lower[state_dim] + fraction * widths[state_dim]
        halton_index += 1
        if not any(np.allclose(point, existing, atol=1e-7) for existing in points):
            points.append(point)

    return np.asarray(points, dtype=np.float32)
