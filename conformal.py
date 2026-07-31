"""Generic finite-sample conformal quantile utilities."""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile_with_rank(scores, *, alpha):
    """Return the split-conformal quantile and its one-based order rank."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    n = scores.shape[0]
    if n == 0:
        raise ValueError("at least one score is required")

    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return math.inf, rank
    return float(np.sort(scores)[rank - 1]), rank


def conformal_quantile(scores, *, alpha):
    """Return the finite-sample split-conformal quantile."""
    quantile, _ = conformal_quantile_with_rank(scores, alpha=alpha)
    return quantile
