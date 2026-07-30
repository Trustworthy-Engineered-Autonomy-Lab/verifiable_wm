#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate signed_tube_margin-compatible Predictor artifacts.

Usage:

    1. Edit config.py.
    2. Run: python train_predictor.py
    3. Run: python build_predictor.py

Outputs:

    predictor_trajectories.npz
        Predicted trajectories with matching dtypes, actions, provenance,
        order, and exact initial states.  For a legacy split_val checkpoint,
        val contains only the disjoint held-out calibration subset.

    conformal_real_trajectories.npz
        Created only for split_val checkpoints.  This is a non-destructive
        held-out view of the original real NPZ for conformal calibration.

    predictor_tube.json
        grid + cells[].bounds, with one min/max tube per verification cell.

Cell sample trajectories exist only during inference and are never saved.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import config


TRAJECTORY_SPLITS = ("train_traj", "val_traj", "test_traj")
OPTIONAL_EXTERNAL_METADATA = (
    "rollout_steps",
    "starv_config",
    "controller_weights",
)
ANGLE_DIM = 0
PI = float(np.pi)
TWO_PI = float(2.0 * np.pi)
EPS = 1e-10
ANGLE_EPS = 1e-7


# =============================================================================
# 1. Runtime and path checks
# =============================================================================


def absolute(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def uses_periodic_angle(environment: str) -> bool:
    return environment.strip().lower() == "pendulum"


def resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(name)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config() -> Dict[str, Any]:
    environment = str(config.ENVIRONMENT).strip().lower()
    if environment not in {
        "pendulum",
        "mountain_car",
        "cartpole",
        "brake_system",
    }:
        raise ValueError(f"unsupported ENVIRONMENT={environment!r}")
    if int(config.HORIZON) < 1:
        raise ValueError("HORIZON must be positive")
    if int(config.SAMPLES_PER_CELL) < 2:
        raise ValueError("SAMPLES_PER_CELL must be at least 2")
    if int(config.BATCH_SIZE) < 1:
        raise ValueError("BATCH_SIZE must be positive")

    paths = {
        "real": absolute(config.REAL_TRAJECTORIES),
        "grid": absolute(config.GRID_RESULT),
        "checkpoint": absolute(config.CHECKPOINT),
        "trajectories": absolute(config.TRAJECTORY_OUTPUT),
        "tube": absolute(config.TUBE_OUTPUT),
        "conformal_real": absolute(config.CONFORMAL_REAL_OUTPUT),
    }
    for label in ("real", "grid"):
        if not paths[label].is_file():
            raise FileNotFoundError(
                f"{label} input does not exist: {paths[label]}"
            )
    if not paths["checkpoint"].is_file():
        raise FileNotFoundError(
            "checkpoint input does not exist: "
            f"{paths['checkpoint']}\n"
            "Run `python train_predictor.py` first to create it."
        )
    if paths["trajectories"].suffix.lower() != ".npz":
        raise ValueError("TRAJECTORY_OUTPUT must end with .npz")
    if paths["tube"].suffix.lower() != ".json":
        raise ValueError("TUBE_OUTPUT must end with .json")
    if paths["conformal_real"].suffix.lower() != ".npz":
        raise ValueError("CONFORMAL_REAL_OUTPUT must end with .npz")
    if paths["conformal_real"] in {
        paths["real"],
        paths["trajectories"],
    }:
        raise ValueError(
            "CONFORMAL_REAL_OUTPUT must not overwrite the source real NPZ "
            "or predictor output"
        )
    paths["trajectories"].parent.mkdir(parents=True, exist_ok=True)
    paths["tube"].parent.mkdir(parents=True, exist_ok=True)
    paths["conformal_real"].parent.mkdir(parents=True, exist_ok=True)

    return {"environment": environment, **paths}


# =============================================================================
# 2. Pendulum periodic-angle conversion
# =============================================================================


def wrap_angles(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return (np.remainder(array + PI, TWO_PI) - PI).astype(
        array.dtype,
        copy=False,
    )


def wrap_angle_trajectories(trajectories: np.ndarray) -> np.ndarray:
    values = np.asarray(trajectories)
    if values.ndim != 3 or values.shape[2] <= ANGLE_DIM:
        raise ValueError(f"invalid Pendulum trajectory shape: {values.shape}")
    result = values.copy()
    result[:, :, ANGLE_DIM] = wrap_angles(
        result[:, :, ANGLE_DIM]
    )
    return result


def periodic_interval_to_json(
    lower: float,
    upper: float,
) -> List[float]:
    """Map an unwrapped interval to one/two intervals in [-pi, pi]."""

    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("angle interval contains NaN or Inf")
    if lower > upper:
        raise ValueError(
            f"angle lower={lower} exceeds upper={upper}"
        )
    if upper - lower >= TWO_PI - ANGLE_EPS:
        return [-PI, PI]

    lower_branch = int(np.floor((lower + PI) / TWO_PI))
    upper_for_branch = (
        upper
        if upper == lower
        else float(np.nextafter(upper, -np.inf))
    )
    upper_branch = int(np.floor((upper_for_branch + PI) / TWO_PI))
    wrapped_lower = float(
        np.clip(lower - lower_branch * TWO_PI, -PI, PI)
    )
    wrapped_upper = float(
        np.clip(upper - upper_branch * TWO_PI, -PI, PI)
    )

    if lower_branch == upper_branch:
        return [wrapped_lower, wrapped_upper]
    return [wrapped_lower, PI, -PI, wrapped_upper]


# =============================================================================
# 3. Verification grid and deterministic cell sampling
# =============================================================================


@dataclass
class Grid:
    names: List[str]
    starts: np.ndarray
    stops: np.ndarray
    nums: np.ndarray
    steps: np.ndarray

    @property
    def ndim(self) -> int:
        return len(self.names)

    @property
    def total_cells(self) -> int:
        return int(np.prod(self.nums))


def load_grid(
    path: Path,
) -> Tuple[Dict[str, Any], Grid, np.ndarray]:
    with path.open("r", encoding="utf-8") as file:
        source = json.load(file)
    if "grid" not in source or "dims" not in source["grid"]:
        raise ValueError(f"{path} is missing grid.dims")

    dims = source["grid"]["dims"]
    if not dims:
        raise ValueError("grid.dims is empty")
    names = [
        str(spec.get("name", f"dim{index}"))
        for index, spec in enumerate(dims)
    ]
    starts = np.asarray(
        [float(spec["start"]) for spec in dims],
        dtype=np.float64,
    )
    stops = np.asarray(
        [float(spec["stop"]) for spec in dims],
        dtype=np.float64,
    )
    nums = np.asarray(
        [int(spec["num"]) for spec in dims],
        dtype=np.int64,
    )
    if np.any(nums <= 0) or np.any(stops < starts):
        raise ValueError("grid has invalid start/stop/num values")
    steps = np.asarray(
        [
            float(
                spec.get(
                    "step",
                    (float(spec["stop"]) - float(spec["start"]))
                    / int(spec["num"]),
                )
            )
            for spec in dims
        ],
        dtype=np.float64,
    )
    grid = Grid(names, starts, stops, nums, steps)

    # Prefer exact initial bounds already present in the source JSON.
    initial_bounds: List[np.ndarray] = []
    source_cells = source.get("cells", [])
    if len(source_cells) == grid.total_cells:
        for cell in source_cells:
            history = cell.get("bounds", [])
            if not history:
                initial_bounds = []
                break
            bounds = np.asarray(history[0], dtype=np.float64)
            if bounds.shape != (grid.ndim, 2):
                initial_bounds = []
                break
            initial_bounds.append(bounds)

    # Otherwise reconstruct cells in NumPy's row-major ndindex order, which
    # matches the external programs' linear cell-index calculation.
    if not initial_bounds:
        for multi_index in np.ndindex(
            *(int(number) for number in grid.nums)
        ):
            bounds = []
            for dim, index in enumerate(multi_index):
                lower = grid.starts[dim] + index * grid.steps[dim]
                upper = grid.starts[dim] + (index + 1) * grid.steps[dim]
                if index == int(grid.nums[dim]) - 1:
                    upper = grid.stops[dim]
                bounds.append([float(lower), float(upper)])
            initial_bounds.append(
                np.asarray(bounds, dtype=np.float64)
            )

    return source, grid, np.stack(initial_bounds, axis=0)


def sample_cell(
    bounds: np.ndarray,
    samples_per_cell: int,
) -> np.ndarray:
    """Sample lower-to-upper diagonal points inside one cell."""

    box = np.asarray(bounds, dtype=np.float32)
    if box.ndim != 2 or box.shape[1] != 2:
        raise ValueError(f"invalid cell bounds shape: {box.shape}")
    if np.any(box[:, 0] > box[:, 1]):
        raise ValueError("cell lower bound exceeds upper bound")
    if samples_per_cell < 2:
        raise ValueError("samples_per_cell must be at least 2")

    lower = box[:, 0]
    upper = box[:, 1]
    weights = np.linspace(
        0.0,
        1.0,
        samples_per_cell,
        dtype=np.float32,
    )[:, None]
    return np.asarray(
        lower[None, :] + weights * (upper - lower)[None, :],
        dtype=np.float32,
    )


# =============================================================================
# 4. Reference-compatible predictor_trajectories.npz
# =============================================================================


def array_fingerprint(values: np.ndarray) -> str:
    """Return the same stable array identity recorded during training."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def checkpoint_calibration_indices(
    checkpoint: Dict[str, Any],
    source_val_traj: np.ndarray,
) -> Optional[np.ndarray]:
    """Validate and return a disjoint val calibration subset, if recorded."""

    protocol = checkpoint.get("data_split")
    if not isinstance(protocol, dict):
        return None
    strategy = str(protocol.get("strategy", "independent_train"))
    if strategy == "independent_train":
        return None
    if strategy != "split_val":
        raise ValueError(
            f"checkpoint has unsupported data split strategy {strategy!r}"
        )

    source_count = int(protocol.get("source_val_count", -1))
    if source_count != len(source_val_traj):
        raise ValueError(
            "checkpoint/source val count mismatch: "
            f"checkpoint={source_count}, source={len(source_val_traj)}"
        )
    expected_fingerprint = str(
        protocol.get("source_val_fingerprint", "")
    )
    actual_fingerprint = array_fingerprint(source_val_traj)
    if not expected_fingerprint or (
        actual_fingerprint != expected_fingerprint
    ):
        raise ValueError(
            "the source val_traj no longer matches the data used to train "
            "this checkpoint"
        )

    train_indices = np.asarray(
        protocol.get("derived_train_indices", []),
        dtype=np.int64,
    )
    calibration_indices = np.asarray(
        protocol.get("calibration_indices", []),
        dtype=np.int64,
    )
    if len(train_indices) == 0 or len(calibration_indices) == 0:
        raise ValueError(
            "split_val checkpoint must record non-empty training and "
            "calibration indices"
        )
    joined = np.concatenate((train_indices, calibration_indices))
    if (
        np.any(joined < 0)
        or np.any(joined >= source_count)
        or len(joined) != source_count
        or len(np.unique(joined)) != source_count
        or set(joined.tolist()) != set(range(source_count))
    ):
        raise ValueError(
            "checkpoint val indices must be disjoint and cover the source "
            "val split exactly once"
        )
    return calibration_indices


def calibration_payload_view(
    payload: Dict[str, np.ndarray],
    calibration_indices: np.ndarray,
    source_val_count: int,
    source_path: Path,
) -> Dict[str, np.ndarray]:
    """Slice every aligned val_* array without changing the source archive."""

    result: Dict[str, np.ndarray] = {}
    for key, value in payload.items():
        array = np.asarray(value)
        if (
            key.startswith("val_")
            and array.ndim >= 1
            and len(array) == source_val_count
        ):
            result[key] = array[calibration_indices].copy()
        else:
            result[key] = array.copy()
    result["source_real_trajectories"] = np.asarray(str(source_path))
    result["source_val_indices"] = calibration_indices.astype(
        np.int64,
        copy=True,
    )
    result["data_role"] = np.asarray(
        "held_out_conformal_calibration_view"
    )
    return result


def load_reference_npz(
    path: Path,
    horizon: int,
    state_dim: int,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    payload: Dict[str, np.ndarray] = {}
    inference_initial: Dict[str, np.ndarray] = {}
    references: Dict[str, np.ndarray] = {}

    with np.load(path, allow_pickle=False) as source:
        for key in source.files:
            payload[key] = np.asarray(source[key]).copy()

        present = [
            key for key in TRAJECTORY_SPLITS if key in source.files
        ]
        for required in ("val_traj", "test_traj"):
            if required not in present:
                raise KeyError(
                    f"{path} is missing required split {required}"
                )

        for key in present:
            trajectories = np.asarray(source[key])
            if trajectories.ndim != 3:
                raise ValueError(
                    f"{key} must have shape (N, T+1, state_dim), "
                    f"got {trajectories.shape}"
                )
            expected_tail = (horizon + 1, state_dim)
            if trajectories.shape[1:] != expected_tail:
                raise ValueError(
                    f"{key} shape {trajectories.shape} does not match "
                    f"HORIZON={horizon}, state_dim={state_dim}; "
                    "automatic truncation is disabled for compatibility"
                )
            if len(trajectories) == 0:
                raise ValueError(f"{key} is empty")
            if not np.isfinite(trajectories).all():
                raise ValueError(f"{key} contains NaN or Inf")

            references[key] = trajectories.copy()
            inference_initial[key] = np.asarray(
                trajectories[:, 0, :],
                dtype=np.float32,
            )

            action_key = key.replace("_traj", "_actions")
            if action_key not in source.files:
                raise KeyError(
                    f"{path} is missing {action_key}; "
                    "conformal.py requires matching action arrays"
                )
            actions = np.asarray(source[action_key])
            if actions.ndim != 3:
                raise ValueError(
                    f"{action_key} must be 3-D, got {actions.shape}"
                )
            if actions.shape[:2] != (
                trajectories.shape[0],
                horizon,
            ):
                raise ValueError(
                    f"{action_key} shape {actions.shape} does not match "
                    f"({trajectories.shape[0]}, {horizon}, action_dim)"
                )
            if not np.isfinite(actions).all():
                raise ValueError(f"{action_key} contains NaN or Inf")

        # Provenance metadata is optional in the server datasets and in
        # conformal.py.  The complete source payload has already been copied,
        # so every metadata field that is present will be preserved exactly.
        # Only validate rollout_steps when the source actually records it.
        if "rollout_steps" in source.files:
            rollout_steps = np.asarray(source["rollout_steps"])
            if rollout_steps.size != 1:
                raise ValueError(
                    "rollout_steps must contain exactly one value, "
                    f"got shape {rollout_steps.shape}"
                )
            recorded_horizon = int(rollout_steps.item())
            if recorded_horizon != horizon:
                raise ValueError(
                    f"real rollout_steps={recorded_horizon}, "
                    f"configured HORIZON={horizon}"
                )

    return payload, inference_initial, references


def make_compatible_predictions(
    payload: Dict[str, np.ndarray],
    predictions: Dict[str, np.ndarray],
    references: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    result = dict(payload)
    for key, predicted in predictions.items():
        reference = np.asarray(references[key])
        predicted = np.asarray(predicted)
        if predicted.shape != reference.shape:
            raise ValueError(
                f"{key} prediction shape {predicted.shape} does not "
                f"match real shape {reference.shape}"
            )

        # Match the real trajectory dtype, then copy state zero after the cast.
        # Existing conformal.py uses np.array_equal rather than np.allclose.
        compatible = predicted.astype(reference.dtype, copy=True)
        compatible[:, 0, :] = reference[:, 0, :]
        if not np.array_equal(
            compatible[:, 0, :],
            reference[:, 0, :],
        ):
            raise RuntimeError(
                f"failed to preserve exact initial states for {key}"
            )
        result[key] = compatible
    return result


def build_reference_predictions(
    predict_function,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    reference_path: Path,
    checkpoint_path: Path,
    conformal_reference_path: Path,
    checkpoint: Dict[str, Any],
    environment: str,
    horizon: int,
    batch_size: int,
    device,
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    payload, initial_states, references = load_reference_npz(
        reference_path,
        horizon,
        model.state_dim,
    )
    source_val = np.asarray(references["val_traj"])
    calibration_indices = checkpoint_calibration_indices(
        checkpoint,
        source_val,
    )
    calibration_payload: Optional[Dict[str, np.ndarray]] = None
    compatible_reference_path = reference_path
    if calibration_indices is not None:
        payload = calibration_payload_view(
            payload,
            calibration_indices,
            len(source_val),
            reference_path,
        )
        protocol = checkpoint["data_split"]
        payload["derived_training_val_indices"] = np.asarray(
            protocol["derived_train_indices"],
            dtype=np.int64,
        )
        references["val_traj"] = np.asarray(
            payload["val_traj"]
        ).copy()
        initial_states["val_traj"] = np.asarray(
            references["val_traj"][:, 0, :],
            dtype=np.float32,
        )
        calibration_payload = {
            key: np.asarray(value).copy()
            for key, value in payload.items()
        }
        compatible_reference_path = conformal_reference_path
        print(
            "\n========== Legacy split adaptation =========="
        )
        print(
            f"source val          : {len(source_val)}"
        )
        print(
            "used for training   : "
            f"{len(protocol['derived_train_indices'])}"
        )
        print(
            f"held for calibration: {len(calibration_indices)}"
        )
        print("source NPZ modified : no")
    predictions: Dict[str, np.ndarray] = {}

    print("\n========== Predictor trajectory NPZ ==========")
    for split_key, states in initial_states.items():
        values = predict_function(
            model,
            states,
            mean,
            std,
            batch_size,
            device,
        )
        if uses_periodic_angle(environment):
            values = wrap_angle_trajectories(values)
        predictions[split_key] = values
        print(f"{split_key:<18}: {values.shape}")

    result = make_compatible_predictions(
        payload,
        predictions,
        references,
    )
    result["decoder_weights"] = np.asarray(str(checkpoint_path))
    result["environment"] = np.asarray(environment)
    result["trajectory_format"] = np.asarray(
        "real_trajectories_compatible_v3"
    )
    result["predictor_checkpoint"] = np.asarray(
        str(checkpoint_path)
    )
    result["reference_real_trajectories"] = np.asarray(
        str(compatible_reference_path)
    )
    result["action_source"] = np.asarray(
        "copied_from_reference_real_trajectories"
    )
    return result, calibration_payload


# =============================================================================
# 5. In-memory cell tube construction
# =============================================================================


def build_cell_tubes(
    predict_function,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    cell_bounds: np.ndarray,
    samples_per_cell: int,
    horizon: int,
    batch_size: int,
    device,
) -> np.ndarray:
    """Return cell min/max bounds; never retain or save sample trajectories."""

    tubes = np.empty(
        (
            len(cell_bounds),
            horizon + 1,
            model.state_dim,
            2,
        ),
        dtype=np.float32,
    )
    print("\n========== In-memory cell tube ==========")
    print(f"cells              : {len(cell_bounds)}")
    print(f"samples per cell   : {samples_per_cell}")
    print(f"transition steps   : {horizon}")
    print("cell trajectories  : not saved")

    progress_interval = max(1, len(cell_bounds) // 10)
    for cell_index, bounds in enumerate(cell_bounds):
        initial_states = sample_cell(bounds, samples_per_cell)
        predictions = predict_function(
            model,
            initial_states,
            mean,
            std,
            batch_size,
            device,
        )
        tubes[cell_index, :, :, 0] = predictions.min(axis=0)
        tubes[cell_index, :, :, 1] = predictions.max(axis=0)

        completed = cell_index + 1
        if (
            completed % progress_interval == 0
            or completed == len(cell_bounds)
        ):
            print(f"built cells        : {completed}/{len(cell_bounds)}")
    return tubes


def save_tube_json(
    path: Path,
    source_grid: Dict[str, Any],
    grid: Grid,
    cell_bounds: np.ndarray,
    tubes: np.ndarray,
    checkpoint: Dict[str, Any],
    grid_path: Path,
    checkpoint_path: Path,
    samples_per_cell: int,
    environment: str,
) -> None:
    grid_json = dict(source_grid["grid"])
    grid_json["dims"] = [
        {
            **dict(source_grid["grid"]["dims"][index]),
            "name": grid.names[index],
            "start": float(grid.starts[index]),
            "stop": float(grid.stops[index]),
            "num": int(grid.nums[index]),
            "step": float(grid.steps[index]),
        }
        for index in range(grid.ndim)
    ]

    def to_json_bounds(
        time_bounds: np.ndarray,
    ) -> List[List[float]]:
        result: List[List[float]] = []
        for dim, (lower, upper) in enumerate(time_bounds):
            if uses_periodic_angle(environment) and dim == ANGLE_DIM:
                result.append(
                    periodic_interval_to_json(lower, upper)
                )
            else:
                result.append([float(lower), float(upper)])
        return result

    cells = []
    for cell_index in range(grid.total_cells):
        # Use original float64 grid bounds at t=0.  Future bounds come from
        # float32 inference.  This prevents boundary lookup failures.
        history = [to_json_bounds(cell_bounds[cell_index])]
        history.extend(
            to_json_bounds(time_bounds)
            for time_bounds in tubes[cell_index, 1:]
        )
        cells.append({"bounds": history})

    best_loss = checkpoint.get("best_selection_loss")
    payload = {
        "method": "transformer_three_point_minmax_envelope",
        "environment": environment,
        "guarantee_type": "sampled envelope; no formal coverage guarantee",
        "sampling_strategy": "lower_corner_center_upper_corner",
        "cell_trajectories_saved": False,
        "samples_per_cell": int(samples_per_cell),
        "horizon": int(tubes.shape[1] - 1),
        "state_dim": int(tubes.shape[2]),
        "angle_representation": (
            "unwrapped_internal_wrapped_union_json"
            if uses_periodic_angle(environment)
            else "linear"
        ),
        "source_grid_result": str(grid_path),
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "best_selection_loss": (
            None if best_loss is None else float(best_loss)
        ),
        "grid": grid_json,
        "cells": cells,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            separators=(",", ":"),
        )


# =============================================================================
# 6. Compatibility validation
# =============================================================================


def interval_pairs(
    values: Sequence[float],
    label: str,
) -> List[Tuple[float, float]]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2 or array.size % 2:
        raise ValueError(f"{label} must contain low/high pairs")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or Inf")
    pairs = []
    for lower, upper in array.reshape(-1, 2):
        if lower > upper:
            raise ValueError(
                f"{label}: lower={lower} exceeds upper={upper}"
            )
        pairs.append((float(lower), float(upper)))
    return pairs


def value_inside(value: float, intervals: Sequence[float]) -> bool:
    return any(
        lower - EPS <= value <= upper + EPS
        for lower, upper in interval_pairs(
            intervals,
            "initial bounds",
        )
    )


def linear_cell_index(
    point: np.ndarray,
    dims: Sequence[Dict[str, Any]],
) -> int:
    indices = []
    for dim, spec in enumerate(dims):
        start = float(spec["start"])
        stop = float(spec["stop"])
        number = int(spec["num"])
        step = float(
            spec.get("step", (stop - start) / number)
        )
        value = float(point[dim])
        if value < start - EPS or value > stop + EPS:
            raise ValueError(
                f"initial state dim {dim}={value} is outside "
                f"[{start}, {stop}]"
            )
        index = (
            number - 1
            if abs(value - stop) <= EPS
            else int(math.floor((value - start) / step))
        )
        indices.append(max(0, min(number - 1, index)))

    linear = 0
    for index, spec in zip(indices, dims):
        linear = linear * int(spec["num"]) + index
    return linear


def validate_initial_lookup(
    initial_states: Iterable[np.ndarray],
    dims: Sequence[Dict[str, Any]],
    cells: Sequence[Dict[str, Any]],
    split_key: str,
) -> None:
    for trajectory_index, state in enumerate(initial_states):
        cell_index = linear_cell_index(np.asarray(state), dims)
        history = cells[cell_index].get("bounds", [])
        if not history:
            raise ValueError(f"cell {cell_index} has no bounds")
        for dim in range(len(dims)):
            if not value_inside(
                float(state[dim]),
                history[0][dim],
            ):
                raise ValueError(
                    f"{split_key}[{trajectory_index}] does not match "
                    f"tube cell {cell_index}, dimension {dim}"
                )


def validate_outputs(
    real_path: Path,
    predictor_path: Path,
    tube_path: Path,
    expected_calibration_count: Optional[int],
) -> Dict[str, Any]:
    initial_states: Dict[str, np.ndarray] = {}
    split_shapes: Dict[str, Tuple[int, ...]] = {}
    horizon: Optional[int] = None
    state_dim: Optional[int] = None

    with np.load(real_path, allow_pickle=False) as real, np.load(
        predictor_path,
        allow_pickle=False,
    ) as predictor:
        for split_key in ("val_traj", "test_traj"):
            if split_key not in real.files:
                raise KeyError(f"real NPZ is missing {split_key}")
        present = [
            key for key in TRAJECTORY_SPLITS if key in real.files
        ]

        for key in present:
            if key not in predictor.files:
                raise KeyError(f"predictor NPZ is missing {key}")
            real_traj = np.asarray(real[key])
            predicted_traj = np.asarray(predictor[key])
            if predicted_traj.shape != real_traj.shape:
                raise ValueError(f"{key} shape mismatch")
            if predicted_traj.dtype != real_traj.dtype:
                raise ValueError(f"{key} dtype mismatch")
            if not np.isfinite(predicted_traj).all():
                raise ValueError(f"predictor {key} contains NaN/Inf")
            if not np.array_equal(
                predicted_traj[:, 0, :],
                real_traj[:, 0, :],
            ):
                raise ValueError(
                    f"{key} initial states are not exactly equal"
                )

            current_horizon = real_traj.shape[1] - 1
            current_state_dim = real_traj.shape[2]
            if horizon is None:
                horizon = current_horizon
                state_dim = current_state_dim
            elif (horizon, state_dim) != (
                current_horizon,
                current_state_dim,
            ):
                raise ValueError(
                    "trajectory splits use inconsistent dimensions"
                )

            action_key = key.replace("_traj", "_actions")
            if action_key not in predictor.files:
                raise KeyError(
                    f"predictor NPZ is missing {action_key}"
                )
            if not np.array_equal(
                real[action_key],
                predictor[action_key],
            ):
                raise ValueError(
                    f"{action_key} was not copied exactly"
                )
            initial_states[key] = real_traj[:, 0, :].copy()
            split_shapes[key] = tuple(
                int(value) for value in real_traj.shape
            )

        # Compare optional provenance only when it exists in the real NPZ.
        # make_compatible_predictions() starts from a copy of the source
        # payload, so present fields must still survive byte-for-byte.
        for key in OPTIONAL_EXTERNAL_METADATA:
            if key not in real.files:
                continue
            if key not in predictor.files:
                raise KeyError(
                    f"predictor NPZ did not preserve metadata {key}"
                )
            if not np.array_equal(real[key], predictor[key]):
                raise ValueError(f"predictor metadata {key} differs")
        if "decoder_weights" not in predictor.files:
            raise KeyError(
                "predictor NPZ is missing decoder_weights"
            )
        if expected_calibration_count is not None:
            count = int(real["val_traj"].shape[0])
            if count != int(expected_calibration_count):
                raise ValueError(
                    "signed-margin calibration expects "
                    f"{expected_calibration_count} val trajectories, "
                    f"got {count}"
                )

    assert horizon is not None and state_dim is not None
    with tube_path.open("r", encoding="utf-8") as file:
        tube = json.load(file)
    if "grid" not in tube or "cells" not in tube:
        raise ValueError("tube JSON must contain grid and cells")
    dims = tube["grid"].get("dims", [])
    if len(dims) != state_dim:
        raise ValueError(
            f"tube grid dimension={len(dims)}, "
            f"trajectory dimension={state_dim}"
        )
    total_cells = int(
        np.prod([int(spec["num"]) for spec in dims])
    )
    cells = tube["cells"]
    if len(cells) != total_cells:
        raise ValueError(
            f"tube has {len(cells)} cells, expected {total_cells}"
        )

    for cell_index, cell in enumerate(cells):
        history = cell.get("bounds", [])
        if len(history) != horizon + 1:
            raise ValueError(
                f"cell {cell_index} has {len(history)} states, "
                f"expected {horizon + 1}"
            )
        for time_index, state_bounds in enumerate(history):
            if len(state_bounds) != state_dim:
                raise ValueError(
                    f"cell {cell_index}, time {time_index}: "
                    "wrong state dimension"
                )
            for dim, values in enumerate(state_bounds):
                interval_pairs(
                    values,
                    f"cell {cell_index}, time {time_index}, dim {dim}",
                )

    for split_key, states in initial_states.items():
        validate_initial_lookup(
            states,
            dims,
            cells,
            split_key,
        )

    return {
        "trajectory_splits": split_shapes,
        "horizon": horizon,
        "state_dim": state_dim,
        "cells": total_cells,
    }


# =============================================================================
# 7. Main build program
# =============================================================================


def main() -> None:
    settings = validate_config()

    # Torch-dependent code stays in predictor_model.py.  Delaying this import
    # also makes the format/grid helpers inspectable without importing Torch.
    from predictor_model import (
        load_predictor_checkpoint,
        predict_trajectories,
    )

    set_seed(int(config.BUILD_SEED))
    device = resolve_device(str(config.DEVICE))
    model, mean, std, checkpoint = load_predictor_checkpoint(
        settings["checkpoint"],
        device,
    )

    environment = settings["environment"]
    checkpoint_environment = checkpoint.get("environment")
    if (
        checkpoint_environment is not None
        and str(checkpoint_environment).strip().lower() != environment
    ):
        raise ValueError(
            f"checkpoint environment={checkpoint_environment!r}, "
            f"config ENVIRONMENT={environment!r}"
        )
    if int(model.horizon) != int(config.HORIZON):
        raise ValueError(
            f"checkpoint horizon={model.horizon}, "
            f"config HORIZON={config.HORIZON}"
        )
    if (
        uses_periodic_angle(environment)
        and bool(config.REQUIRE_UNWRAPPED_PENDULUM_CHECKPOINT)
        and checkpoint.get("angle_representation")
        != "unwrapped_theta"
    ):
        raise ValueError(
            "Pendulum checkpoint is not marked as unwrapped_theta. "
            "Use a checkpoint trained with continuous unwrapped theta, "
            "or explicitly disable the policy in config.py."
        )

    source_grid, grid, cell_bounds = load_grid(settings["grid"])
    if grid.ndim != model.state_dim:
        raise ValueError(
            f"grid state_dim={grid.ndim}, "
            f"checkpoint state_dim={model.state_dim}"
        )

    print("========== Predictor build ==========")
    print(f"environment        : {environment}")
    print(f"device             : {device}")
    print(f"horizon            : {config.HORIZON}")
    print(f"real trajectories  : {settings['real']}")
    print(f"grid result        : {settings['grid']}")
    print(f"checkpoint         : {settings['checkpoint']}")
    print(f"trajectory output  : {settings['trajectories']}")
    print(f"tube output        : {settings['tube']}")

    # Validate the reference NPZ and build its predictions before the
    # potentially expensive all-cell tube pass.  Input-format errors now fail
    # before thousands of cells are evaluated.
    trajectory_payload, calibration_payload = build_reference_predictions(
        predict_trajectories,
        model,
        mean,
        std,
        settings["real"],
        settings["checkpoint"],
        settings["conformal_real"],
        checkpoint,
        environment,
        int(config.HORIZON),
        int(config.BATCH_SIZE),
        device,
    )
    tubes = build_cell_tubes(
        predict_trajectories,
        model,
        mean,
        std,
        cell_bounds,
        int(config.SAMPLES_PER_CELL),
        int(config.HORIZON),
        int(config.BATCH_SIZE),
        device,
    )

    # Write temporary files, validate them, then atomically replace outputs.
    trajectory_temp = settings["trajectories"].with_name(
        settings["trajectories"].stem + ".building.npz"
    )
    tube_temp = settings["tube"].with_name(
        settings["tube"].stem + ".building.json"
    )
    conformal_real_temp = settings["conformal_real"].with_name(
        settings["conformal_real"].stem + ".building.npz"
    )
    try:
        np.savez_compressed(trajectory_temp, **trajectory_payload)
        if calibration_payload is not None:
            np.savez_compressed(
                conformal_real_temp,
                **calibration_payload,
            )
        save_tube_json(
            tube_temp,
            source_grid,
            grid,
            cell_bounds,
            tubes,
            checkpoint,
            settings["grid"],
            settings["checkpoint"],
            int(config.SAMPLES_PER_CELL),
            environment,
        )
        validation_real = (
            conformal_real_temp
            if calibration_payload is not None
            else settings["real"]
        )
        expected_calibration_count = (
            len(calibration_payload["val_traj"])
            if calibration_payload is not None
            else config.EXPECTED_CALIBRATION_COUNT
        )
        report = validate_outputs(
            validation_real,
            trajectory_temp,
            tube_temp,
            expected_calibration_count,
        )
        os.replace(trajectory_temp, settings["trajectories"])
        os.replace(tube_temp, settings["tube"])
        if calibration_payload is not None:
            os.replace(
                conformal_real_temp,
                settings["conformal_real"],
            )
    finally:
        trajectory_temp.unlink(missing_ok=True)
        tube_temp.unlink(missing_ok=True)
        conformal_real_temp.unlink(missing_ok=True)

    print("\n========== Completed ==========")
    print(f"[Saved] {settings['trajectories']}")
    print(f"[Saved] {settings['tube']}")
    if calibration_payload is not None:
        print(f"[Saved] {settings['conformal_real']}")
    print(
        "[Passed] signed_tube_margin compatibility: "
        f"{report['cells']} cells, "
        f"horizon={report['horizon']}, "
        f"state_dim={report['state_dim']}"
    )


if __name__ == "__main__":
    main()
