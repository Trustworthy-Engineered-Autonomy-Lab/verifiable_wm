"""Construct empirical reachable tubes from three rollouts per grid cell."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGLET_HEADLESS", "True")

import numpy as np
import torch

from dynamic import *
from model import *
from utils import load_config, resolve_device, to_numpy


PROJECT_ROOT = Path(__file__).resolve().parent


_VERIFIER_SYSTEMS = {
    "CartpoleVerifier": {
        "state_dim": 4,
        "decoder_state_indices": [0, 2],
        "dynamic_name": "CartPole",
        "dynamic_defaults": {
            "gravity": 9.8,
            "mass_cart": 1.0,
            "mass_pole": 0.1,
            "length": 0.5,
            "force_mag": 10.0,
            "dt": 0.02,
        },
    },
    "MountainCarVerifier": {
        "state_dim": 2,
        "decoder_state_indices": [0, 1],
        "dynamic_name": "MountainCar",
        "dynamic_defaults": {
            "min_pos": -1.2,
            "max_pos": 0.6,
            "min_speed": -0.07,
            "max_speed": 0.07,
            "min_action": -1.0,
            "max_action": 1.0,
            "power": 0.0015,
        },
    },
    "PendulumVerifier": {
        "state_dim": 2,
        "decoder_state_indices": [0, 1],
        "dynamic_name": "Pendulum",
        # Must stay identical to dynamic.Pendulum.__init__'s defaults; the
        # comparison below is exact, so a stale copy here rejects otherwise
        # valid configs. dt is 0.05 to match gym's Pendulum-v1.
        "dynamic_defaults": {
            "max_speed": 8.0,
            "max_torque": 2.0,
            "dt": 0.05,
            "g": 10.0,
            "m": 1.0,
            "l": 1.0,
        },
    },
    "BrakeVerifier": {
        "state_dim": 2,
        "decoder_state_indices": [0, 1],
        "dynamic_name": "Brake",
        "dynamic_defaults": {
            "dt": 0.1,
            "v_lead": 0.0,
        },
    },
}


def _normalized_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    dimensions = grid.get("dims")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("grid must define at least one dimension")

    normalized = []
    for index, dim in enumerate(dimensions):
        start = float(dim["start"])
        stop = float(dim["stop"])
        count = int(dim["num"])
        if not np.isfinite(start) or not np.isfinite(stop):
            raise ValueError("grid dimension bounds must be finite")
        if start > stop:
            raise ValueError(
                f"grid dimension start exceeds stop at index {index}"
            )
        if count <= 0:
            raise ValueError("grid dimension num must be positive")
        normalized.append(
            {
                "name": str(dim.get("name", f"state_{index}")),
                "start": start,
                "stop": stop,
                "num": count,
            }
        )
    return normalized


def grid_fingerprint(grid: dict[str, Any]) -> str:
    normalized = json.dumps(
        _normalized_grid(grid),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def grid_cell_bounds(grid: dict[str, Any]) -> np.ndarray:
    """Return StarV-ordered cell bounds with shape (cells, dims, 2)."""
    starts = []
    stops = []
    for dim in _normalized_grid(grid):
        edges = np.linspace(
            float(dim["start"]),
            float(dim["stop"]),
            int(dim["num"]) + 1,
        )
        starts.append(edges[:-1])
        stops.append(edges[1:])

    starts_mesh = np.array(np.meshgrid(*starts, indexing="ij"))
    stops_mesh = np.array(np.meshgrid(*stops, indexing="ij"))
    cell_starts = starts_mesh.reshape(starts_mesh.shape[0], -1).T
    cell_stops = stops_mesh.reshape(stops_mesh.shape[0], -1).T
    return np.stack((cell_starts, cell_stops), axis=-1)


def decoder_variant(config: dict[str, Any]) -> str:
    try:
        return str(config["decoder"]["variant"])
    except KeyError:
        raise ValueError("sampling config must define 'decoder.variant'") from None


def _decoder_weights(config: dict[str, Any]) -> str:
    weights = config["decoder"]["weights"]
    if isinstance(weights, dict):
        weights = weights[decoder_variant(config)]
    return Path(str(weights)).as_posix()


def _validate_system_semantics(
    sampling_config: dict[str, Any],
    grid_config: dict[str, Any],
) -> dict[str, Any]:
    verifier_name = str(grid_config["verifier"]["name"])
    system = _VERIFIER_SYSTEMS.get(verifier_name)
    if system is None:
        raise ValueError(
            f"Unsupported verifier for cellwise sampling: {verifier_name}"
        )

    state_dim = len(grid_config["grid"]["dims"])
    if state_dim != int(system["state_dim"]):
        raise ValueError(
            f"{verifier_name} state dimension mismatch: "
            f"grid={state_dim}, expected={system['state_dim']}"
        )

    configured_indices = sampling_config.get(
        "decoder_state_indices",
        sampling_config["decoder"].get("state_indices"),
    )
    if configured_indices is None:
        actual_indices = list(range(state_dim))
    else:
        try:
            actual_indices = [int(index) for index in configured_indices]
        except (TypeError, ValueError) as error:
            raise ValueError(
                "decoder_state_indices must be an ordered integer list"
            ) from error
    expected_indices = list(system["decoder_state_indices"])
    if actual_indices != expected_indices:
        raise ValueError(
            "decoder_state_indices mismatch: "
            f"sampling={actual_indices}, verifier={expected_indices}"
        )

    dynamic = sampling_config["dynamic"]
    actual_dynamic_name = str(dynamic["name"])
    expected_dynamic_name = str(system["dynamic_name"])
    if actual_dynamic_name != expected_dynamic_name:
        raise ValueError(
            "dynamic mismatch: "
            f"sampling={actual_dynamic_name!r}, "
            f"verifier={expected_dynamic_name!r}"
        )

    actual_dynamic_args = copy.deepcopy(dynamic.get("args") or {})
    expected_dynamic_args = copy.deepcopy(system["dynamic_defaults"])
    unknown_args = set(actual_dynamic_args).difference(expected_dynamic_args)
    if unknown_args:
        raise ValueError(
            f"dynamic args are not used by the verifier: {sorted(unknown_args)}"
        )
    effective_dynamic_args = {
        **expected_dynamic_args,
        **actual_dynamic_args,
    }
    if effective_dynamic_args != expected_dynamic_args:
        raise ValueError(
            "dynamic parameters do not match the verifier defaults: "
            f"sampling={effective_dynamic_args}, "
            f"verifier={expected_dynamic_args}"
        )

    return {
        "verifier_name": verifier_name,
        "state_dim": state_dim,
        "decoder_state_indices": actual_indices,
        "dynamic_name": actual_dynamic_name,
        "dynamic_args": actual_dynamic_args,
        "effective_dynamic_args": effective_dynamic_args,
    }


def validate_cellwise_setup(
    sampling_config: dict[str, Any],
    model_starv_config: dict[str, Any],
    grid_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate the unified config that defines one cellwise rollout."""
    samples_per_cell = int(sampling_config["samples_per_cell"])
    if samples_per_cell != 3:
        raise ValueError(
            f"samples_per_cell must be exactly 3, got {samples_per_cell}"
        )
    cell_batch_size = int(sampling_config["cell_batch_size"])
    if cell_batch_size <= 0:
        raise ValueError("cell_batch_size must be positive")

    model_id = decoder_variant(sampling_config)
    if str(sampling_config["model_id"]) != model_id:
        raise ValueError(
            f"model_id mismatch: config={sampling_config['model_id']!r}, "
            f"sampling={model_id!r}"
        )

    if grid_fingerprint(model_starv_config["grid"]) != grid_fingerprint(
        grid_config["grid"]
    ):
        raise ValueError("model StarV grid does not match canonical grid")

    rollout_steps = int(sampling_config["rollout_steps"])
    model_steps = int(
        model_starv_config["verifier"]["kwargs"]["num_steps"]
    )
    grid_steps = int(grid_config["verifier"]["kwargs"]["num_steps"])
    if not (rollout_steps == model_steps == grid_steps):
        raise ValueError(
            "horizon mismatch: "
            f"sampling={rollout_steps}, model_starv={model_steps}, "
            f"grid_config={grid_steps}"
        )
    if rollout_steps <= 0:
        raise ValueError("rollout horizon must be positive")
    if model_starv_config["verifier"] != grid_config["verifier"]:
        raise ValueError("model StarV verifier does not match canonical verifier")
    if bool(
        grid_config["verifier"].get("kwargs", {}).get(
            "early_stop",
            False,
        )
    ):
        raise ValueError(
            "cellwise sampled tubes require verifier early_stop=false"
        )

    system_metadata = _validate_system_semantics(
        sampling_config,
        grid_config,
    )
    decoder = sampling_config["decoder"]
    model_type = str(decoder["name"])
    expected_layer_order = [
        model_type,
        str(sampling_config["controller"]["name"]),
    ]
    actual_layer_order = list(model_starv_config["layers"])
    if actual_layer_order != expected_layer_order:
        raise ValueError(
            "StarV layer order mismatch: "
            f"actual={actual_layer_order}, "
            f"expected={expected_layer_order}"
        )
    model_layer = model_starv_config["layers"].get(model_type)
    if model_layer is None:
        raise ValueError(f"StarV layers do not contain {model_type}")
    starv_model_kwargs = model_layer.get("kwargs") or {}
    actual_weights = _decoder_weights(sampling_config)
    starv_weights = Path(str(starv_model_kwargs["weights"])).as_posix()
    if actual_weights != starv_weights:
        raise ValueError(
            f"decoder weights mismatch: sampling={actual_weights!r}, "
            f"starv={starv_weights!r}"
        )

    controller = sampling_config["controller"]
    controller_layer = model_starv_config["layers"].get(
        str(controller["name"])
    )
    if controller_layer is None:
        raise ValueError("StarV layers do not contain the sampling controller")
    controller_kwargs = controller_layer.get("kwargs") or {}
    actual_controller_weights = Path(
        str(controller["weights"])
    ).as_posix()
    starv_controller_weights = Path(
        str(controller_kwargs["weights"])
    ).as_posix()
    if actual_controller_weights != starv_controller_weights:
        raise ValueError("controller weights mismatch")
    actual_controller_activation = str(
        (controller.get("args") or {}).get("activation", "relu")
    )
    starv_controller_activation = str(
        controller_kwargs.get("activation", "tanh")
    )
    if actual_controller_activation != starv_controller_activation:
        raise ValueError("controller activation mismatch")
    if actual_controller_activation not in {"sigmoid", "tanh"}:
        raise ValueError(
            "controller activation must be sigmoid or tanh for StarV"
        )
    if float(controller_kwargs.get("output_factor", 1.0)) != 1.0:
        raise ValueError(
            "StarV controller output_factor does not match point rollout"
        )

    decoder_args = decoder.get("args") or {}
    metadata = {
        "model_id": model_id,
        "model_type": model_type,
        "decoder_weights": actual_weights,
        "controller_name": str(controller["name"]),
        "controller_weights": actual_controller_weights,
        "controller_activation": actual_controller_activation,
        "dynamic_name": system_metadata["dynamic_name"],
        "dynamic_args": system_metadata["dynamic_args"],
        "effective_dynamic_args": system_metadata[
            "effective_dynamic_args"
        ],
        "decoder_state_indices": system_metadata[
            "decoder_state_indices"
        ],
        "horizon": rollout_steps,
        "state_dim": system_metadata["state_dim"],
        "grid_fingerprint": grid_fingerprint(grid_config["grid"]),
        "formal_semantics_match": model_type == "G_MLP"
        or str(decoder_args.get("output_activation", "sigmoid")) == "clamp",
    }

    if model_type == "G_MLP":
        actual_z_range = float(decoder_args.get("z_range", 0.8))
        starv_z_range = float(starv_model_kwargs.get("z_range", 0.8))
        if actual_z_range != starv_z_range:
            raise ValueError("G_MLP z_range mismatch")
        if not np.isfinite(actual_z_range) or actual_z_range <= 0.0:
            raise ValueError("G_MLP z_range must be finite and positive")
        metadata["z_range"] = actual_z_range
        actual_latent_dim = int(decoder_args.get("latent_dim", 2))
        starv_latent_dim = int(starv_model_kwargs.get("latent_dim", 2))
        if actual_latent_dim != starv_latent_dim:
            raise ValueError("G_MLP latent_dim mismatch")
        if actual_latent_dim <= 0:
            raise ValueError("G_MLP latent_dim must be positive")
        actual_state_dim = int(decoder_args.get("state_dim", 2))
        starv_state_dim = int(starv_model_kwargs.get("state_dim", 2))
        expected_decoder_dim = len(
            system_metadata["decoder_state_indices"]
        )
        if not (
            actual_state_dim
            == starv_state_dim
            == expected_decoder_dim
        ):
            raise ValueError(
                "G_MLP state_dim does not match decoder_state_indices"
            )
        actual_output_dim = int(decoder_args.get("output_dim", 96 * 96))
        starv_output_dim = int(
            starv_model_kwargs.get("output_dim", 96 * 96)
        )
        if actual_output_dim != starv_output_dim:
            raise ValueError("G_MLP output_dim mismatch")
        if actual_output_dim != 96 * 96:
            raise ValueError("G_MLP output_dim must be exactly 96 * 96")
        if "output_activation" in decoder_args:
            raise ValueError(
                "G_MLP does not accept an output_activation argument; "
                "its forward pass always clamps"
            )
        if "output_activation" in starv_model_kwargs:
            raise ValueError(
                "StarV G_MLP does not accept an output_activation "
                "argument; its reachability pass always clamps"
            )
        metadata["latent_dim"] = actual_latent_dim
        metadata["generator_output_activation"] = "clamp"
    elif model_type == "Decoder":
        output_activation = str(
            decoder_args.get("output_activation", "sigmoid")
        )
        if (
            system_metadata["verifier_name"] == "BrakeVerifier"
            and output_activation != "clamp"
        ):
            raise ValueError(
                "Brake Decoder output activation must be clamp"
            )
        if (
            system_metadata["verifier_name"] == "BrakeVerifier"
            and str(
                starv_model_kwargs.get(
                    "output_activation",
                    "sigmoid",
                )
            )
            != "clamp"
        ):
            raise ValueError(
                "Brake StarV Decoder output activation must be clamp"
            )
        metadata["decoder_output_activation"] = output_activation
    else:
        raise ValueError(f"Unsupported sampled tube model: {model_type}")

    return metadata


def sample_cellwise_states(
    cell_bounds: np.ndarray,
    *,
    samples_per_cell: int,
    seed: int,
) -> np.ndarray:
    """Uniformly sample a fixed number of states from each cell."""
    bounds = np.asarray(cell_bounds, dtype=np.float64)
    if bounds.ndim != 3 or bounds.shape[2] != 2:
        raise ValueError(
            "cell bounds must have shape (cell count, state_dim, 2)"
        )
    if not np.all(np.isfinite(bounds)):
        raise ValueError("cell bounds must contain only finite values")
    if np.any(bounds[:, :, 0] > bounds[:, :, 1]):
        raise ValueError("cell lower bounds must not exceed upper bounds")
    if int(samples_per_cell) <= 0:
        raise ValueError("samples_per_cell must be positive")
    num_cells, state_dim, _ = bounds.shape
    states = np.empty(
        (num_cells, int(samples_per_cell), state_dim),
        dtype=np.float32,
    )

    for cell_index, cell in enumerate(bounds):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 0, cell_index])
        )
        unit = rng.random((int(samples_per_cell), state_dim))
        states[cell_index] = (
            cell[:, 0] + (cell[:, 1] - cell[:, 0]) * unit
        ).astype(np.float32)

    return states


def sample_cellwise_latents(
    *,
    num_cells: int,
    samples_per_cell: int,
    num_steps: int,
    latent_dim: int,
    z_range: float,
    seed: int,
) -> np.ndarray:
    """Generate explicit per-cell CGAN latent values."""
    latents = np.empty(
        (
            int(num_cells),
            int(samples_per_cell),
            int(num_steps),
            int(latent_dim),
        ),
        dtype=np.float32,
    )

    for cell_index in range(int(num_cells)):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 1, cell_index])
        )
        latents[cell_index] = rng.uniform(
            -float(z_range),
            float(z_range),
            size=latents.shape[1:],
        ).astype(np.float32)

    return latents


def _shortest_circular_bounds(values: np.ndarray) -> list[float]:
    angles = (
        np.asarray(values, dtype=np.float64) + np.pi
    ) % (2.0 * np.pi) - np.pi
    ordered = np.sort(angles)
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 2.0 * np.pi)))
    gap_index = int(np.argmax(gaps))
    arc_start = float(ordered[(gap_index + 1) % len(ordered)])
    arc_end = float(ordered[gap_index])
    if arc_start <= arc_end:
        return [arc_start, arc_end]
    return [arc_start, float(np.pi), float(-np.pi), arc_end]


def _sampled_cell_result(
    trajectories: np.ndarray,
    verifier: dict[str, Any],
) -> bool:
    name = str(verifier["name"])
    kwargs = verifier.get("kwargs") or {}
    final_states = trajectories[:, -1, :]

    if name == "CartpoleVerifier":
        threshold = float(kwargs["goal_angle_threshold"])
        return bool(np.all(np.abs(final_states[:, 2]) <= threshold))
    if name == "MountainCarVerifier":
        threshold = float(kwargs["goal_position_threshold"])
        return bool(np.all(final_states[:, 0] >= threshold))
    if name == "PendulumVerifier":
        threshold = float(kwargs["goal_angle_threshold"])
        return bool(np.all(np.abs(final_states[:, 0]) <= threshold))
    if name == "BrakeVerifier":
        return bool(np.all(trajectories[:, :, 0] > 0.0))
    raise ValueError(f"Unsupported verifier for sampled tube: {name}")


def sampled_cells_from_trajectories(
    trajectories: np.ndarray,
    cell_bounds: np.ndarray,
    verifier: dict[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate per-cell trajectories into compare-compatible cells."""
    trajectories = np.asarray(trajectories, dtype=np.float32)
    cell_bounds = np.asarray(cell_bounds, dtype=np.float64)
    if trajectories.ndim != 4:
        raise ValueError(
            "trajectories must have shape "
            "(cell count, samples per cell, horizon + 1, state_dim)"
        )
    if cell_bounds.ndim != 3 or cell_bounds.shape[2] != 2:
        raise ValueError(
            "cell bounds must have shape (cell count, state_dim, 2)"
        )
    if trajectories.shape[0] != cell_bounds.shape[0]:
        raise ValueError(
            "trajectory cell count does not match cell bounds: "
            f"{trajectories.shape[0]} != {cell_bounds.shape[0]}"
        )
    if trajectories.shape[1] != 3:
        raise ValueError(
            f"expected 3 samples per cell, got {trajectories.shape[1]}"
        )
    expected_states = int(verifier["kwargs"]["num_steps"]) + 1
    if trajectories.shape[2] != expected_states:
        raise ValueError(
            f"trajectory horizon mismatch: actual={trajectories.shape[2] - 1}, "
            f"expected={expected_states - 1}"
        )
    if trajectories.shape[3] != cell_bounds.shape[1]:
        raise ValueError(
            "trajectory state dimension does not match cell bounds"
        )
    if not np.all(np.isfinite(trajectories)):
        raise ValueError("trajectories must contain only finite values")
    if not np.all(np.isfinite(cell_bounds)):
        raise ValueError("cell bounds must contain only finite values")
    if np.any(cell_bounds[:, :, 0] > cell_bounds[:, :, 1]):
        raise ValueError("cell lower bounds must not exceed upper bounds")

    initial_states = trajectories[:, :, 0, :]
    lower = cell_bounds[:, None, :, 0]
    upper = cell_bounds[:, None, :, 1]
    if np.any(initial_states < lower - 1e-6) or np.any(
        initial_states > upper + 1e-6
    ):
        raise ValueError("initial trajectory state is outside its cell")

    verifier_name = str(verifier["name"])
    is_pendulum = verifier_name == "PendulumVerifier"

    cells = []
    for cell_index, (cell_trajectories, initial_bounds) in enumerate(
        zip(trajectories, cell_bounds)
    ):
        bounds_history = [initial_bounds.tolist()]
        for time_index in range(1, cell_trajectories.shape[1]):
            state_bounds = []
            for dim_index in range(cell_trajectories.shape[2]):
                values = cell_trajectories[:, time_index, dim_index]
                if is_pendulum and dim_index == 0:
                    state_bounds.append(_shortest_circular_bounds(values))
                else:
                    state_bounds.append(
                        [float(np.min(values)), float(np.max(values))]
                    )
            bounds_history.append(state_bounds)

        cells.append(
            {
                "cell_index": cell_index,
                "bounds": bounds_history,
                "result": _sampled_cell_result(cell_trajectories, verifier),
            }
        )
    return cells


def build_sampled_safety_result(
    starv_config: dict[str, Any],
    cells: list[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    grid_config: dict[str, Any],
) -> dict[str, Any]:
    """Return a safety-result-shaped payload for the existing compare tools."""
    payload = copy.deepcopy(starv_config)
    payload["grid"] = copy.deepcopy(grid_config["grid"])
    payload.update(metadata)
    payload["method"] = "cellwise_sampled_trajectory_envelope"
    payload["guarantee_type"] = "empirical_only"
    payload["cells"] = cells
    return payload


def validate_sampled_safety_result(payload: dict[str, Any]) -> None:
    if payload.get("method") != "cellwise_sampled_trajectory_envelope":
        return
    for key in ("layers", "verifier", "grid", "cells"):
        if key not in payload:
            raise ValueError(f"sampled tube is missing top-level field {key!r}")

    canonical_cells = grid_cell_bounds(payload["grid"])
    cells = payload["cells"]
    if len(cells) != len(canonical_cells):
        raise ValueError(
            f"sampled tube cell count mismatch: actual={len(cells)}, "
            f"expected={len(canonical_cells)}"
        )
    horizon = int(payload["verifier"]["kwargs"]["num_steps"])
    state_dim = canonical_cells.shape[1]
    for expected_index, (cell, canonical_bounds) in enumerate(
        zip(cells, canonical_cells)
    ):
        if int(cell.get("cell_index", expected_index)) != expected_index:
            raise ValueError("sampled tube cell_index order mismatch")
        bounds_history = cell.get("bounds")
        if not isinstance(bounds_history, list) or len(bounds_history) != horizon + 1:
            raise ValueError(
                f"sampled tube horizon mismatch in cell {expected_index}: "
                f"actual={0 if bounds_history is None else len(bounds_history)}, "
                f"expected={horizon + 1}"
            )
        if type(cell.get("result")) is not bool:
            raise ValueError(
                f"sampled tube result must be bool in cell {expected_index}"
            )

        initial_bounds = np.asarray(bounds_history[0], dtype=np.float64)
        if initial_bounds.shape != canonical_bounds.shape or not np.allclose(
            initial_bounds, canonical_bounds
        ):
            raise ValueError(
                f"sampled tube initial bounds mismatch in cell {expected_index}"
            )

        for time_index, state_bounds in enumerate(bounds_history):
            if len(state_bounds) != state_dim:
                raise ValueError(
                    "sampled tube state dimension mismatch in "
                    f"cell {expected_index}, time {time_index}"
                )
            for dim_index, dim_bounds in enumerate(state_bounds):
                values = np.asarray(dim_bounds, dtype=np.float64).reshape(-1)
                if values.size < 2 or values.size % 2:
                    raise ValueError(
                        "sampled tube bounds must contain low/high pairs in "
                        f"cell {expected_index}, time {time_index}, dim {dim_index}"
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError("sampled tube bounds must be finite")
                if np.any(values[0::2] > values[1::2]):
                    raise ValueError(
                        "sampled tube interval lower bound exceeds upper bound"
                    )

    if payload.get("guarantee_type") != "empirical_only":
        raise ValueError(
            "sampled tube guarantee_type must be 'empirical_only'"
        )
    if int(payload.get("samples_per_cell", -1)) != 3:
        raise ValueError("sampled tube samples_per_cell must be exactly 3")
    if "seed" not in payload:
        raise ValueError("sampled tube is missing seed metadata")
    if int(payload.get("horizon", -1)) != horizon:
        raise ValueError(
            "sampled tube horizon metadata mismatch: "
            f"actual={payload.get('horizon')}, expected={horizon}"
        )
    if int(payload.get("state_dim", -1)) != state_dim:
        raise ValueError(
            "sampled tube state_dim metadata mismatch: "
            f"actual={payload.get('state_dim')}, expected={state_dim}"
        )
    expected_fingerprint = grid_fingerprint(payload["grid"])
    if payload.get("grid_fingerprint") != expected_fingerprint:
        raise ValueError("sampled tube grid fingerprint mismatch")


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically write a JSON artifact in its destination directory."""
    validate_sampled_safety_result(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(payload, output, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        with temporary_path.open("r", encoding="utf-8") as saved:
            validate_sampled_safety_result(json.load(saved))
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def write_npz_atomic(path: str | Path, **arrays: Any) -> Path:
    """Atomically write an NPZ artifact in its destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_arrays = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(
                f"NPZ field {name!r} has unsupported object dtype"
            )
        normalized_arrays[name] = array

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            np.savez_compressed(output, **normalized_arrays)
            output.flush()
            os.fsync(output.fileno())
        with np.load(temporary_path, allow_pickle=False) as saved:
            if set(saved.files) != set(normalized_arrays):
                raise ValueError("NPZ field list changed during serialization")
            for name, expected in normalized_arrays.items():
                actual = np.asarray(saved[name])
                if actual.shape != expected.shape:
                    raise ValueError(
                        f"NPZ field {name!r} shape changed during serialization"
                    )
                if actual.dtype != expected.dtype:
                    raise ValueError(
                        f"NPZ field {name!r} dtype changed during serialization"
                    )
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def validate_cellwise_rollout_artifact(
    trajectories: np.ndarray,
    actions: np.ndarray,
    *,
    num_cells: int,
    samples_per_cell: int,
    horizon: int,
    state_dim: int,
    initial_states: np.ndarray,
    latents: np.ndarray | None = None,
    latent_dim: int | None = None,
    z_range: float | None = None,
) -> None:
    """Validate rollout arrays before assigning a final artifact name."""
    trajectories = np.asarray(trajectories)
    expected_trajectory_shape = (
        int(num_cells),
        int(samples_per_cell),
        int(horizon) + 1,
        int(state_dim),
    )
    if trajectories.shape != expected_trajectory_shape:
        raise ValueError(
            "trajectory shape mismatch: "
            f"actual={trajectories.shape}, "
            f"expected={expected_trajectory_shape}"
        )
    if not np.issubdtype(trajectories.dtype, np.number) or not np.all(
        np.isfinite(trajectories)
    ):
        raise ValueError("trajectories must contain only finite numbers")

    initial_states = np.asarray(initial_states)
    expected_initial_shape = (
        int(num_cells),
        int(samples_per_cell),
        int(state_dim),
    )
    if initial_states.shape != expected_initial_shape:
        raise ValueError(
            "initial states shape mismatch: "
            f"actual={initial_states.shape}, "
            f"expected={expected_initial_shape}"
        )
    if not np.array_equal(trajectories[:, :, 0, :], initial_states):
        raise ValueError(
            "trajectory initial states do not match the fixed-seed states"
        )

    actions = np.asarray(actions)
    expected_action_prefix = (
        int(num_cells),
        int(samples_per_cell),
        int(horizon),
    )
    if (
        actions.ndim != 4
        or actions.shape[:3] != expected_action_prefix
        or actions.shape[3] <= 0
    ):
        raise ValueError(
            "action shape mismatch: "
            f"actual={actions.shape}, "
            f"expected prefix={expected_action_prefix}"
        )
    if not np.issubdtype(actions.dtype, np.number) or not np.all(
        np.isfinite(actions)
    ):
        raise ValueError("actions must contain only finite numbers")

    if latents is None:
        if latent_dim is not None or z_range is not None:
            raise ValueError(
                "latent metadata was provided without latent values"
            )
        return

    if latent_dim is None or z_range is None:
        raise ValueError("latent_dim and z_range are required with latents")
    latents = np.asarray(latents)
    expected_latent_shape = (
        int(num_cells),
        int(samples_per_cell),
        int(horizon),
        int(latent_dim),
    )
    if latents.shape != expected_latent_shape:
        raise ValueError(
            "latent shape mismatch: "
            f"actual={latents.shape}, expected={expected_latent_shape}"
        )
    if latents.dtype != np.float32:
        raise ValueError(
            f"latents must use float32, got {latents.dtype}"
        )
    if not np.all(np.isfinite(latents)):
        raise ValueError("latents must contain only finite numbers")
    tolerance = np.finfo(np.float32).eps * max(1.0, abs(float(z_range)))
    if np.any(latents < -float(z_range) - tolerance) or np.any(
        latents > float(z_range) + tolerance
    ):
        raise ValueError(
            f"latent values are outside configured range "
            f"[-{z_range}, {z_range}]"
        )


def _load_cellwise_states(
    path: Path,
    cell_bounds: np.ndarray,
    *,
    samples_per_cell: int,
    seed: int,
    grid: dict[str, Any],
    horizon: int,
    source_grid_config: str,
) -> np.ndarray:
    required = {
        "states",
        "cell_bounds",
        "cell_indices",
        "seed",
        "samples_per_cell",
        "grid_fingerprint",
        "grid_json",
        "state_dim",
        "horizon",
        "source_grid_config",
    }
    with np.load(path, allow_pickle=False) as saved:
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(
                f"shared states file is missing fields: {sorted(missing)}"
            )
        actual_seed = int(saved["seed"])
        if actual_seed != int(seed):
            raise ValueError(
                f"shared states seed mismatch: actual={actual_seed}, "
                f"expected={seed}"
            )
        actual_count = int(saved["samples_per_cell"])
        if actual_count != int(samples_per_cell):
            raise ValueError(
                "shared states samples_per_cell mismatch: "
                f"actual={actual_count}, expected={samples_per_cell}"
            )
        expected_fingerprint = grid_fingerprint(grid)
        actual_fingerprint = str(saved["grid_fingerprint"].item())
        if actual_fingerprint != expected_fingerprint:
            raise ValueError("shared states grid fingerprint mismatch")
        if int(saved["horizon"]) != int(horizon):
            raise ValueError("shared states horizon mismatch")
        if int(saved["state_dim"]) != int(cell_bounds.shape[1]):
            raise ValueError("shared states state_dim mismatch")
        cell_indices = np.asarray(saved["cell_indices"], dtype=np.int64)
        expected_indices = np.arange(len(cell_bounds), dtype=np.int64)
        if not np.array_equal(cell_indices, expected_indices):
            raise ValueError("shared states cell_indices mismatch")
        expected_grid_json = json.dumps(
            _normalized_grid(grid),
            sort_keys=True,
            separators=(",", ":"),
        )
        if str(saved["grid_json"].item()) != expected_grid_json:
            raise ValueError("shared states grid_json mismatch")
        actual_source_grid = str(saved["source_grid_config"].item())
        if actual_source_grid != str(source_grid_config):
            raise ValueError(
                "shared states source_grid_config mismatch: "
                f"actual={actual_source_grid!r}, "
                f"expected={str(source_grid_config)!r}"
            )

        states = np.asarray(saved["states"])
        if states.dtype != np.float32:
            raise ValueError(
                f"shared states must use float32, got {states.dtype}"
            )
        saved_bounds = np.asarray(saved["cell_bounds"], dtype=np.float64)

    expected_shape = (
        len(cell_bounds),
        int(samples_per_cell),
        cell_bounds.shape[1],
    )
    if states.shape != expected_shape:
        raise ValueError(
            f"shared states shape mismatch: actual={states.shape}, "
            f"expected={expected_shape}"
        )
    if saved_bounds.shape != cell_bounds.shape or not np.allclose(
        saved_bounds, cell_bounds
    ):
        raise ValueError("shared states cell bounds mismatch")
    if not np.all(np.isfinite(states)):
        raise ValueError("shared states contain NaN or Inf")
    lower = cell_bounds[:, None, :, 0]
    upper = cell_bounds[:, None, :, 1]
    if np.any(states < lower - 1e-6) or np.any(states > upper + 1e-6):
        raise ValueError("shared states contain points outside their cells")
    expected_states = sample_cellwise_states(
        cell_bounds,
        samples_per_cell=samples_per_cell,
        seed=seed,
    )
    if not np.array_equal(states, expected_states):
        raise ValueError(
            "shared states do not match the configured fixed seed"
        )
    return states


def load_or_create_cellwise_states(
    path: str | Path,
    cell_bounds: np.ndarray,
    *,
    samples_per_cell: int,
    seed: int,
    grid: dict[str, Any],
    horizon: int,
    source_grid_config: str,
) -> np.ndarray:
    """Create shared per-cell states once, then validate and reuse them."""
    path = Path(path)
    cell_bounds = np.asarray(cell_bounds, dtype=np.float64)
    if path.exists():
        return _load_cellwise_states(
            path,
            cell_bounds,
            samples_per_cell=samples_per_cell,
            seed=seed,
            grid=grid,
            horizon=horizon,
            source_grid_config=source_grid_config,
        )

    states = sample_cellwise_states(
        cell_bounds,
        samples_per_cell=samples_per_cell,
        seed=seed,
    )
    write_npz_atomic(
        path,
        states=states,
        cell_bounds=cell_bounds,
        cell_indices=np.arange(len(cell_bounds), dtype=np.int64),
        seed=np.array(int(seed)),
        samples_per_cell=np.array(int(samples_per_cell)),
        grid_fingerprint=np.array(grid_fingerprint(grid)),
        grid_json=np.array(
            json.dumps(
                _normalized_grid(grid),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        state_dim=np.array(cell_bounds.shape[1]),
        horizon=np.array(int(horizon)),
        source_grid_config=np.array(str(source_grid_config)),
    )
    return _load_cellwise_states(
        path,
        cell_bounds,
        samples_per_cell=samples_per_cell,
        seed=seed,
        grid=grid,
        horizon=horizon,
        source_grid_config=source_grid_config,
    )


def load_state_dict(path: str | Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_controller(config: dict[str, Any], device: torch.device) -> Any:
    controller_config = config["controller"]
    controller_name = str(controller_config["name"])
    controller_cls = globals()[controller_name]
    controller = controller_cls(**controller_config.get("args", {})).to(device).eval()
    controller.load_state_dict(
        load_state_dict(controller_config["weights"], device)
    )
    print(f"[Load] {controller_name}={controller_config['weights']}")
    return controller


def resolve_decoder_weights(config: dict[str, Any]) -> str:
    weights = config["decoder"]["weights"]
    if isinstance(weights, dict):
        weights = weights[decoder_variant(config)]
    return str(weights)


def load_decoder(config: dict[str, Any], device: torch.device) -> Any:
    decoder_config = config["decoder"]
    decoder_name = str(decoder_config["name"])
    decoder_cls = globals()[decoder_name]
    decoder = decoder_cls(**decoder_config.get("args", {})).to(device).eval()
    weights = resolve_decoder_weights(config)
    decoder.load_state_dict(load_state_dict(weights, device))
    print(f"[Load] {decoder_name}[{decoder_variant(config)}]={weights}")
    return decoder


@torch.no_grad()
def rollout_sampled_trajectories(
    states0: torch.Tensor,
    steps: int,
    decoder: Any,
    controller: Any,
    dynamic: Any,
    device: torch.device,
    decoder_state_indices: list[int] | None = None,
    latents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_samples, state_dim = states0.shape
    trajectories = torch.empty(
        num_samples,
        int(steps) + 1,
        state_dim,
        device=device,
    )
    action_steps = []
    states = states0.clone()
    trajectories[:, 0, :] = states

    for step in range(int(steps)):
        decoder_states = states
        if decoder_state_indices is not None:
            decoder_states = states[:, decoder_state_indices]
        if latents is None:
            images = decoder(decoder_states)
        else:
            images = decoder(decoder_states, latents[:, step, :])
        actions = controller(images)
        states = dynamic.step(states, actions)
        action_steps.append(actions)
        trajectories[:, step + 1, :] = states

    return trajectories, torch.stack(action_steps, dim=1)


@torch.no_grad()
def rollout_cellwise_batches(
    *,
    states: np.ndarray,
    steps: int,
    decoder: Any,
    controller: Any,
    dynamic: Any,
    device: torch.device,
    cell_batch_size: int,
    decoder_state_indices: list[int] | None = None,
    latents: np.ndarray | None = None,
    progress: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 3:
        raise ValueError(
            "cellwise states must have shape (cells, samples, state_dim)"
        )
    if int(cell_batch_size) <= 0:
        raise ValueError("cell_batch_size must be positive")

    latent_array = None
    if latents is not None:
        latent_array = np.asarray(latents, dtype=np.float32)
        expected_prefix = (states.shape[0], states.shape[1], int(steps))
        if latent_array.ndim != 4 or latent_array.shape[:3] != expected_prefix:
            raise ValueError(
                f"cellwise latent shape mismatch: actual={latent_array.shape}, "
                f"expected prefix={expected_prefix}"
            )

    trajectory_chunks = []
    action_chunks = []
    samples_per_cell = states.shape[1]
    for cell_start in range(0, len(states), int(cell_batch_size)):
        cell_stop = min(cell_start + int(cell_batch_size), len(states))
        batch_cells = cell_stop - cell_start
        batch_states = torch.from_numpy(
            states[cell_start:cell_stop].reshape(-1, states.shape[2])
        ).to(device)
        batch_latents = None
        if latent_array is not None:
            batch_latents = torch.from_numpy(
                latent_array[cell_start:cell_stop].reshape(
                    -1,
                    int(steps),
                    latent_array.shape[3],
                )
            ).to(device)

        trajectories, actions = rollout_sampled_trajectories(
            batch_states,
            int(steps),
            decoder,
            controller,
            dynamic,
            device,
            decoder_state_indices,
            batch_latents,
        )
        trajectory_chunks.append(
            to_numpy(trajectories).reshape(
                batch_cells,
                samples_per_cell,
                int(steps) + 1,
                states.shape[2],
            )
        )
        action_chunks.append(
            to_numpy(actions).reshape(
                batch_cells,
                samples_per_cell,
                int(steps),
                actions.shape[2],
            )
        )
        if progress is not None:
            progress(cell_stop, len(states))

    return (
        np.concatenate(trajectory_chunks, axis=0),
        np.concatenate(action_chunks, axis=0),
    )


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def generate_sampled_tube(
    config: dict[str, Any],
    config_path: str | Path,
) -> Path:
    model_starv_config = load_config(_project_path(config["starv_config"]))
    grid_config = load_config(_project_path(config["grid_config"]))
    metadata = validate_cellwise_setup(
        config,
        model_starv_config,
        grid_config,
    )

    device = resolve_device(config.get("device", "auto"))
    controller = load_controller(config, device)
    decoder = load_decoder(config, device)
    dynamic_config = config["dynamic"]
    dynamic = globals()[dynamic_config["name"]](
        **dynamic_config.get("args", {})
    )
    print(f"[Dynamic] {dynamic_config['name']}")

    steps = int(config["rollout_steps"])
    samples_per_cell = int(config["samples_per_cell"])
    seed = int(config["seed"])
    cell_batch_size = int(config["cell_batch_size"])
    cells = grid_cell_bounds(grid_config["grid"])
    print(
        f"[Cellwise Setup] model={metadata['model_id']} "
        f"({metadata['model_type']}), cells={len(cells)}, "
        f"samples_per_cell={samples_per_cell}, "
        f"trajectories={len(cells) * samples_per_cell}, "
        f"horizon={steps}, seed={seed}"
    )

    states_path = _project_path(config["states_file"])
    states = load_or_create_cellwise_states(
        states_path,
        cells,
        samples_per_cell=samples_per_cell,
        seed=seed,
        grid=grid_config["grid"],
        horizon=steps,
        source_grid_config=str(config["grid_config"]),
    )
    print(
        f"[Cellwise States] cells={len(cells)}, "
        f"samples_per_cell={samples_per_cell}, path={states_path}"
    )

    latents = None
    if metadata["model_type"] == "G_MLP":
        latents = sample_cellwise_latents(
            num_cells=len(cells),
            samples_per_cell=samples_per_cell,
            num_steps=steps,
            latent_dim=int(metadata["latent_dim"]),
            z_range=float(metadata["z_range"]),
            seed=seed,
        )

    decoder_state_indices = config.get(
        "decoder_state_indices",
        config["decoder"].get("state_indices"),
    )
    trajectories, actions = rollout_cellwise_batches(
        states=states,
        steps=steps,
        decoder=decoder,
        controller=controller,
        dynamic=dynamic,
        device=device,
        cell_batch_size=cell_batch_size,
        decoder_state_indices=decoder_state_indices,
        latents=latents,
        progress=lambda completed, total: print(
            f"[Cellwise Rollout] {completed}/{total} cells "
            f"({100.0 * completed / total:.1f}%)"
        ),
    )

    latent_dim = None
    z_range = None
    if latents is not None:
        latent_dim = int(metadata["latent_dim"])
        z_range = float(metadata["z_range"])
    validate_cellwise_rollout_artifact(
        trajectories,
        actions,
        num_cells=len(cells),
        samples_per_cell=samples_per_cell,
        horizon=steps,
        state_dim=int(metadata["state_dim"]),
        initial_states=states,
        latents=latents,
        latent_dim=latent_dim,
        z_range=z_range,
    )
    sampled_cells = sampled_cells_from_trajectories(
        trajectories,
        cells,
        model_starv_config["verifier"],
    )

    trajectory_path = _project_path(config["trajectory_file"])
    arrays = {
        "trajectories": trajectories,
        "actions": actions,
        "cell_indices": np.arange(len(cells), dtype=np.int64),
        "seed": np.array(seed),
        "samples_per_cell": np.array(samples_per_cell),
        "grid_fingerprint": np.array(metadata["grid_fingerprint"]),
        "rollout_steps": np.array(steps),
        "model_id": np.array(metadata["model_id"]),
        "model_type": np.array(metadata["model_type"]),
        "state_dim": np.array(int(metadata["state_dim"])),
        "decoder_state_indices": np.asarray(
            metadata["decoder_state_indices"],
            dtype=np.int64,
        ),
        "decoder_weights": np.array(metadata["decoder_weights"]),
        "controller_name": np.array(metadata["controller_name"]),
        "controller_weights": np.array(metadata["controller_weights"]),
        "controller_activation": np.array(
            metadata["controller_activation"]
        ),
        "dynamic_name": np.array(metadata["dynamic_name"]),
        "dynamic_args_json": np.array(
            json.dumps(
                metadata["dynamic_args"],
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "formal_semantics_match": np.array(
            bool(metadata["formal_semantics_match"])
        ),
        "source_sampling_config": np.array(str(config_path)),
        "source_grid_config": np.array(str(config["grid_config"])),
        "source_starv_config": np.array(str(config["starv_config"])),
        "source_states_file": np.array(str(config["states_file"])),
    }
    if latents is not None:
        arrays["latents"] = latents
        arrays["z_range"] = np.array(float(metadata["z_range"]))
        arrays["latent_dim"] = np.array(int(metadata["latent_dim"]))
        arrays["generator_output_activation"] = np.array(
            metadata["generator_output_activation"]
        )
    else:
        arrays["decoder_output_activation"] = np.array(
            metadata["decoder_output_activation"]
        )
    write_npz_atomic(trajectory_path, **arrays)
    print(
        f"[Cellwise Trajectories] shape={trajectories.shape}, "
        f"path={trajectory_path}"
    )

    result_metadata = {
        **metadata,
        "samples_per_cell": samples_per_cell,
        "seed": seed,
        "source_sampling_config": str(config_path),
        "source_grid_config": str(config["grid_config"]),
        "source_starv_config": str(config["starv_config"]),
        "source_states": str(config["states_file"]),
        "source_trajectories": str(config["trajectory_file"]),
    }
    payload = build_sampled_safety_result(
        model_starv_config,
        sampled_cells,
        result_metadata,
        grid_config=grid_config,
    )
    tube_path = _project_path(config["tube_file"])
    write_json_atomic(tube_path, payload)
    print(f"[Cellwise Tube] cells={len(sampled_cells)}, path={tube_path}")
    return tube_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="+",
        type=Path,
        help="Unified sampled-tube config JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for config_path in args.configs:
        print(f"[Config] {config_path}")
        config = load_config(config_path)
        output_path = generate_sampled_tube(config, config_path)
        print(f"[Done] sampled tube saved to {output_path}")


if __name__ == "__main__":
    main()
