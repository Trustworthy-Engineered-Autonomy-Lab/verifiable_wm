#!/usr/bin/env python3
"""Run reproducible repeated evaluations of symbolic and sampled tubes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import copy
import io
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import signed_tube_margin as stm


PROJECT_ROOT = Path(__file__).resolve().parent
# Symbolic tubes and real trajectories are large binary artifacts kept outside
# the repository; safety_results/ points at wherever they live.
SHARED_SAFETY_ROOT = PROJECT_ROOT / "safety_results"
PREDICTOR_ROOT = SHARED_SAFETY_ROOT / "predictor"
DEFAULT_MASTER_SEED = 2025
POOL_SIZE = 800
VAL_SIZE = 600
TEST_SIZE = 200
NUM_REPEATS = 5
STD_DDOF = 0

ENVIRONMENTS: dict[str, dict[str, Any]] = {
    "cartpole": {
        "real": SHARED_SAFETY_ROOT / "cartpole" / "3600cell_real_trajectories.npz",
        "dims": (0, 2),
        "expected_cells": 3600,
        "expected_horizon": 21,
        "symbolic": {
            "dwm": SHARED_SAFETY_ROOT / "cartpole" / "3600cell_dwm_safety_result.json",
            "cgan": SHARED_SAFETY_ROOT / "cartpole" / "3600cell_g_mlp_safety_result.json",
        },
        "sampling_configs": {
            "dwm": PROJECT_ROOT / "config" / "sampling" / "cartpole.json",
            "cgan": PROJECT_ROOT / "config" / "sampling" / "cartpole_g_mlp.json",
        },
    },
    "mountain_car": {
        "real": SHARED_SAFETY_ROOT / "mountain_car" / "6400cell_real_trajectories.npz",
        "dims": (0, 1),
        "expected_cells": 6400,
        "expected_horizon": 21,
        "symbolic": {
            "dwm": SHARED_SAFETY_ROOT / "mountain_car" / "6400cell_dwm_safety_result.json",
            "cgan": SHARED_SAFETY_ROOT / "mountain_car" / "6400cell_g_mlp_safety_result.json",
        },
        "sampling_configs": {
            "dwm": PROJECT_ROOT / "config" / "sampling" / "mountain_car.json",
            "cgan": PROJECT_ROOT / "config" / "sampling" / "mountain_car_g_mlp.json",
        },
    },
    "pendulum": {
        "real": SHARED_SAFETY_ROOT / "pendulum" / "5000cell_real_trajectories.npz",
        "dims": (0, 1),
        "expected_cells": 5000,
        "expected_horizon": 21,
        "symbolic": {
            "dwm": SHARED_SAFETY_ROOT / "pendulum" / "5000cell_dwm_safety_result.json",
            "cgan": SHARED_SAFETY_ROOT / "pendulum" / "5000cell_g_mlp_safety_result.json",
        },
        "sampling_configs": {
            "dwm": PROJECT_ROOT / "config" / "sampling" / "pendulum.json",
            "cgan": PROJECT_ROOT / "config" / "sampling" / "pendulum_g_mlp.json",
        },
    },
    "brake_system": {
        "real": SHARED_SAFETY_ROOT / "brake_system" / "1600cell_real_trajectories.npz",
        "dims": (0, 1),
        "expected_cells": 1600,
        "expected_horizon": 11,
        "symbolic": {
            "dwm": SHARED_SAFETY_ROOT / "brake_system" / "1600cell_dwm_safety_result.json",
            "cgan": SHARED_SAFETY_ROOT / "brake_system" / "1600cell_g_mlp_safety_result.json",
        },
        "sampling_configs": {
            "dwm": PROJECT_ROOT / "config" / "sampling" / "brake_system.json",
            "cgan": PROJECT_ROOT / "config" / "sampling" / "brake_system_g_mlp.json",
        },
    },
}
DECODERS = ("dwm", "cgan")


def predictor_tube_paths(env: str) -> list[Path]:
    """Return the five existing predictor tubes in their fixed file order."""
    paths = [
        PREDICTOR_ROOT / env / f"predictor_tube_seed_{index}.json"
        for index in range(NUM_REPEATS)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing predictor tubes: {missing}")
    return paths


def derive_repeat_seeds(master_seed: int, *, num_repeats: int = NUM_REPEATS) -> list[dict[str, int]]:
    """Derive independent split and tube seeds from one master seed."""
    if num_repeats <= 0:
        raise ValueError("num_repeats must be positive")
    repeat_sequences = np.random.SeedSequence(int(master_seed)).spawn(num_repeats)
    rows = []
    for repeat_index, repeat_sequence in enumerate(repeat_sequences):
        split_sequence, tube_sequence = repeat_sequence.spawn(2)
        rows.append({
            "repeat_index": repeat_index,
            "split_seed": int(split_sequence.generate_state(1, dtype=np.uint32)[0]),
            "tube_seed": int(tube_sequence.generate_state(1, dtype=np.uint32)[0]),
        })
    return rows


def combine_real_pool(val_trajectories: np.ndarray, test_trajectories: np.ndarray) -> np.ndarray:
    """Combine the existing 400 validation and 400 test trajectories."""
    val = np.asarray(val_trajectories)
    test = np.asarray(test_trajectories)
    if val.ndim != 3 or test.ndim != 3:
        raise ValueError("val_traj and test_traj must have shape (N, T+1, state_dim)")
    if len(val) != 400 or len(test) != 400:
        raise ValueError(f"expected 400 val and 400 test trajectories, got {len(val)} and {len(test)}")
    if val.shape[1:] != test.shape[1:]:
        raise ValueError(f"val/test trajectory shape mismatch: {val.shape} versus {test.shape}")
    return np.concatenate([val, test], axis=0)


def split_real_pool(
    pool: np.ndarray,
    *,
    split_seed: int,
    val_size: int = VAL_SIZE,
    test_size: int = TEST_SIZE,
) -> dict[str, Any]:
    """Return one reproducible disjoint split, including arrays and indices."""
    trajectories = np.asarray(pool)
    if trajectories.ndim != 3:
        raise ValueError("real pool must have shape (N, T+1, state_dim)")
    if len(trajectories) != val_size + test_size:
        raise ValueError(
            f"pool size must equal val_size + test_size, got {len(trajectories)} "
            f"versus {val_size} + {test_size}"
        )
    permutation = np.random.default_rng(int(split_seed)).permutation(len(trajectories))
    val_indices = permutation[:val_size].astype(np.int64, copy=False)
    test_indices = permutation[val_size:].astype(np.int64, copy=False)
    return {
        "val_traj": trajectories[val_indices],
        "test_traj": trajectories[test_indices],
        "val_indices": val_indices,
        "test_indices": test_indices,
        "split_seed": int(split_seed),
    }


def write_split_artifact(
    path: Path,
    split: dict[str, Any],
    *,
    repeat_index: int,
    pool_fingerprint: str,
) -> None:
    """Persist a split with enough metadata to validate safe reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        val_traj=np.asarray(split["val_traj"]),
        test_traj=np.asarray(split["test_traj"]),
        val_indices=np.asarray(split["val_indices"], dtype=np.int64),
        test_indices=np.asarray(split["test_indices"], dtype=np.int64),
        split_seed=np.array(int(split["split_seed"]), dtype=np.uint64),
        repeat_index=np.array(int(repeat_index), dtype=np.int64),
        pool_fingerprint=np.array(str(pool_fingerprint)),
    )


def load_split_artifact(
    path: Path,
    *,
    expected_repeat: int,
    expected_seed: int,
    expected_pool_fingerprint: str,
) -> dict[str, Any]:
    """Load and validate a saved 600/200 split."""
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "val_traj", "test_traj", "val_indices", "test_indices",
            "split_seed", "repeat_index", "pool_fingerprint",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"split artifact is missing fields: {sorted(missing)}")
        if int(saved["repeat_index"]) != int(expected_repeat):
            raise ValueError("split repeat index mismatch")
        if int(saved["split_seed"]) != int(expected_seed):
            raise ValueError("split seed mismatch")
        if str(saved["pool_fingerprint"].item()) != str(expected_pool_fingerprint):
            raise ValueError("split pool fingerprint mismatch")
        result = {key: np.asarray(saved[key]).copy() for key in (
            "val_traj", "test_traj", "val_indices", "test_indices"
        )}
        result["split_seed"] = int(saved["split_seed"])
    if result["val_traj"].shape[0] != VAL_SIZE or result["test_traj"].shape[0] != TEST_SIZE:
        raise ValueError("split artifact must contain 600 val and 200 test trajectories")
    val_indices = result["val_indices"]
    test_indices = result["test_indices"]
    if len(np.unique(val_indices)) != VAL_SIZE or len(np.unique(test_indices)) != TEST_SIZE:
        raise ValueError("split indices contain duplicates")
    if np.intersect1d(val_indices, test_indices).size:
        raise ValueError("split val/test indices overlap")
    if not np.array_equal(np.sort(np.concatenate([val_indices, test_indices])), np.arange(POOL_SIZE)):
        raise ValueError("split indices do not partition the 800-trajectory pool")
    return result


def array_fingerprint(array: np.ndarray) -> str:
    """Return a stable fingerprint that includes dtype, shape, and bytes."""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def aggregate_values(values: Sequence[float]) -> dict[str, float | int]:
    """Aggregate repeated values using the requested population std."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("aggregate values must be a nonempty finite one-dimensional sequence")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=STD_DDOF)),
        "count": int(len(array)),
        "ddof": STD_DDOF,
    }


def single_result_summary(*, coverage: float, average_area: float) -> dict[str, Any]:
    """Represent a fixed symbolic raw result without a repeat std."""
    return {
        "coverage_mean": float(coverage),
        "coverage_std": None,
        "area_mean": float(average_area),
        "area_std": None,
        "num_repeats": 1,
    }


def strict_table_metrics(
    trajectories: np.ndarray,
    grid: stm.GridInfo,
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int],
) -> dict[str, Any]:
    """Compute metrics only when every evaluation trajectory is valid."""
    rows = stm.evaluate_set(trajectories, grid, cells, dims)
    invalid = [row for row in rows if row["status"] != "valid"]
    if invalid:
        examples = "; ".join(str(row["error"]) for row in invalid[:3])
        raise ValueError(
            f"every trajectory must match a valid tube cell; invalid={len(invalid)}/{len(rows)}: {examples}"
        )
    return stm.table_metrics(trajectories, grid, cells, dims)


def evaluate_inflated_repeat(
    calibration_trajectories: np.ndarray,
    evaluation_trajectories: np.ndarray,
    grid: stm.GridInfo,
    cells: Sequence[dict[str, Any]],
    dims: Sequence[int],
    *,
    alpha: float,
) -> dict[str, Any]:
    """Calibrate one CP-R gamma and evaluate its inflated tube."""
    gamma, rank_margin, rank = stm.calibration_gamma(
        calibration_trajectories,
        grid,
        cells,
        dims,
        alpha=alpha,
    )
    inflated_cells = stm.inflate_cells(cells, dims, [gamma] * len(dims))
    metrics = strict_table_metrics(
        evaluation_trajectories,
        grid,
        inflated_cells,
        dims,
    )
    return {
        "gamma": float(gamma),
        "signed_margin_at_rank": float(rank_margin),
        "rank": int(rank),
        "alpha": float(alpha),
        "calibration_size": int(len(calibration_trajectories)),
        "evaluation_size": int(len(evaluation_trajectories)),
        "coverage": float(metrics["coverage_rate"]),
        "covered_trajectories": int(metrics["covered_trajectories"]),
        "average_area": float(metrics["tube_area"]["mean"]),
    }


def aggregate_metric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate coverage and average area across repeated experiments."""
    if len(rows) != NUM_REPEATS:
        raise ValueError(f"expected exactly {NUM_REPEATS} repeat rows, got {len(rows)}")
    coverage = aggregate_values([float(row["coverage"]) for row in rows])
    area = aggregate_values([float(row["average_area"]) for row in rows])
    return {
        "coverage_mean": coverage["mean"],
        "coverage_std": coverage["std"],
        "area_mean": area["mean"],
        "area_std": area["std"],
        "num_repeats": NUM_REPEATS,
        "std_ddof": STD_DDOF,
    }


def build_repeat_sampling_config(
    sampling_config: dict[str, Any],
    *,
    env: str,
    decoder: str,
    repeat_index: int,
    tube_seed: int,
    data_root: Path,
    result_root: Path,
) -> dict[str, Any]:
    """Redirect one unified sampling config into a repeat directory."""
    if decoder not in {"dwm", "cgan"}:
        raise ValueError(f"unknown decoder: {decoder}")
    result = copy.deepcopy(sampling_config)
    repeat_name = f"repeat_{int(repeat_index):02d}"
    data_dir = Path(data_root) / env / "sampled" / repeat_name
    result_dir = Path(result_root) / env / "b_sampled" / decoder / repeat_name
    result["seed"] = int(tube_seed)
    result["states_file"] = str(data_dir / "cellwise_states.npz")
    result["trajectory_file"] = str(
        data_dir / f"sampled_trajectories_{result['model_id']}.npz"
    )
    result["tube_file"] = str(result_dir / "sampled_tube.json")
    return result


def apply_experiment_overrides(
    env: str,
    decoder: str,
    sampling_config: dict[str, Any],
    model_starv_config: dict[str, Any],
    canonical_grid_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Apply the approved Brake grid and CartPole cGAN corrections in memory."""
    sampling = copy.deepcopy(sampling_config)
    model_starv = copy.deepcopy(model_starv_config)
    canonical = copy.deepcopy(canonical_grid_config)
    if env == "brake_system":
        for config in (model_starv, canonical):
            velocity_dims = [dim for dim in config["grid"]["dims"] if dim.get("name") == "vel"]
            if len(velocity_dims) != 1:
                raise ValueError("Braking grid must contain exactly one vel dimension")
            velocity_dims[0]["stop"] = 6.4
            velocity_dims[0]["num"] = 40
    if env == "cartpole" and decoder == "cgan":
        sampling["decoder"].setdefault("args", {})["z_range"] = 0.05
        model_starv["layers"]["G_MLP"].setdefault("kwargs", {})["z_range"] = 0.05
    return sampling, model_starv, canonical


def materialize_case_configs(
    env: str,
    decoder: str,
    *,
    result_root: Path,
    overwrite: bool,
) -> Path:
    """Write a self-contained corrected config snapshot for one sampled case."""
    sampling_source = Path(ENVIRONMENTS[env]["sampling_configs"][decoder])
    sampling_config = load_json(sampling_source)
    model_starv = load_json(PROJECT_ROOT / sampling_config["starv_config"])
    canonical = load_json(PROJECT_ROOT / sampling_config["grid_config"])
    sampling_config, model_starv, canonical = apply_experiment_overrides(
        env, decoder, sampling_config, model_starv, canonical
    )
    config_dir = Path(result_root) / "configs" / env / decoder
    model_starv_path = config_dir / "model_starv.json"
    grid_path = Path(result_root) / "configs" / env / "dwm" / "canonical_grid.json"
    sampling_path = config_dir / "sampling.json"
    sampling_config["starv_config"] = str(model_starv_path)
    sampling_config["grid_config"] = str(grid_path)
    write_json_artifact(model_starv_path, model_starv, overwrite=overwrite)
    write_json_artifact(grid_path, canonical, overwrite=overwrite)
    write_json_artifact(sampling_path, sampling_config, overwrite=overwrite)
    return sampling_path


def prepare_environment_splits(
    source_path: Path,
    environment_data_root: Path,
    repeat_seeds: Sequence[dict[str, int]],
    *,
    overwrite: bool,
) -> list[Path]:
    """Create or validate one environment's common pool and five splits."""
    source_path = Path(source_path)
    environment_data_root = Path(environment_data_root)
    with np.load(source_path, allow_pickle=False) as source:
        if "val_traj" not in source or "test_traj" not in source:
            raise KeyError(f"{source_path} must contain val_traj and test_traj")
        pool = combine_real_pool(source["val_traj"], source["test_traj"])
    fingerprint = array_fingerprint(pool)
    pool_path = environment_data_root / "real_pool.npz"
    if pool_path.exists() and not overwrite:
        with np.load(pool_path, allow_pickle=False) as saved:
            if str(saved["pool_fingerprint"].item()) != fingerprint:
                raise ValueError(f"existing pool fingerprint mismatch: {pool_path}")
            if not np.array_equal(saved["pool_traj"], pool):
                raise ValueError(f"existing pool content mismatch: {pool_path}")
    else:
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pool_path,
            pool_traj=pool,
            pool_fingerprint=np.array(fingerprint),
            source_path=np.array(str(source_path.resolve())),
            source_val_key=np.array("val_traj"),
            source_test_key=np.array("test_traj"),
            original_val_count=np.array(400, dtype=np.int64),
            original_test_count=np.array(400, dtype=np.int64),
        )

    split_paths = []
    for row in repeat_seeds:
        repeat_index = int(row["repeat_index"])
        split_seed = int(row["split_seed"])
        split_path = environment_data_root / "splits" / f"repeat_{repeat_index:02d}.npz"
        if split_path.exists() and not overwrite:
            load_split_artifact(
                split_path,
                expected_repeat=repeat_index,
                expected_seed=split_seed,
                expected_pool_fingerprint=fingerprint,
            )
        else:
            split = split_real_pool(pool, split_seed=split_seed)
            write_split_artifact(
                split_path,
                split,
                repeat_index=repeat_index,
                pool_fingerprint=fingerprint,
            )
        split_paths.append(split_path)
    return split_paths


def experiment_roots(master_seed: int) -> tuple[Path, Path]:
    name = f"master_seed_{int(master_seed)}"
    return (
        PROJECT_ROOT / "datasets" / "repeated_tube_evaluation" / name,
        PROJECT_ROOT / "results" / "repeated_tube_evaluation" / name,
    )


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json_artifact(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    """Write deterministic JSON while refusing inconsistent reuse."""
    path = Path(path)
    normalized = _json_ready(payload)
    if path.exists() and not overwrite:
        if load_json(path) != normalized:
            raise FileExistsError(f"existing JSON differs; use --overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_text_artifact(path: Path, content: str, *, overwrite: bool) -> None:
    """Write text idempotently, rejecting an inconsistent existing artifact."""
    path = Path(path)
    if path.exists() and not overwrite:
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"existing text artifact differs; use --overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_seed_manifest(data_root: Path, master_seed: int, *, overwrite: bool) -> list[dict[str, int]]:
    seeds = derive_repeat_seeds(master_seed)
    payload = {
        "master_seed": int(master_seed),
        "num_repeats": NUM_REPEATS,
        "pool_size": POOL_SIZE,
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE,
        "samples_per_cell": 3,
        "alpha": 0.05,
        "std_ddof": STD_DDOF,
        "repeats": seeds,
    }
    write_json_artifact(Path(data_root) / "experiment_seeds.json", payload, overwrite=overwrite)
    return seeds


def prepare_splits(
    environments: Sequence[str],
    *,
    master_seed: int,
    data_root: Path,
    overwrite: bool,
) -> None:
    seeds = write_seed_manifest(data_root, master_seed, overwrite=overwrite)
    for env in environments:
        paths = prepare_environment_splits(
            ENVIRONMENTS[env]["real"],
            Path(data_root) / env,
            seeds,
            overwrite=overwrite,
        )
        print(f"[Splits] {env}: pool=800, repeats={len(paths)}, root={Path(data_root) / env}")


def load_pool_artifact(data_root: Path, env: str) -> tuple[np.ndarray, str]:
    path = Path(data_root) / env / "real_pool.npz"
    with np.load(path, allow_pickle=False) as saved:
        pool = np.asarray(saved["pool_traj"]).copy()
        fingerprint = str(saved["pool_fingerprint"].item())
    if pool.shape[0] != POOL_SIZE or array_fingerprint(pool) != fingerprint:
        raise ValueError(f"invalid real pool artifact: {path}")
    return pool, fingerprint


def load_repeat_splits(
    data_root: Path,
    env: str,
    seeds: Sequence[dict[str, int]],
    pool_fingerprint: str,
) -> list[dict[str, Any]]:
    return [
        load_split_artifact(
            Path(data_root) / env / "splits" / f"repeat_{int(row['repeat_index']):02d}.npz",
            expected_repeat=int(row["repeat_index"]),
            expected_seed=int(row["split_seed"]),
            expected_pool_fingerprint=pool_fingerprint,
        )
        for row in seeds
    ]


def _raw_metric_payload(
    metrics: dict[str, Any],
    *,
    evaluation_size: int,
) -> dict[str, Any]:
    return {
        "coverage": float(metrics["coverage_rate"]),
        "covered_trajectories": int(metrics["covered_trajectories"]),
        "evaluation_size": int(evaluation_size),
        "average_area": float(metrics["tube_area"]["mean"]),
    }


def evaluate_symbolic_case(
    env: str,
    decoder: str,
    *,
    alpha: float,
    data_root: Path,
    result_root: Path,
    repeat_seeds: Sequence[dict[str, int]],
    overwrite: bool,
) -> dict[str, Any]:
    """Evaluate one fixed symbolic tube on the common pool and five splits."""
    spec = ENVIRONMENTS[env]
    safety_path = Path(spec["symbolic"][decoder])
    grid, cells = stm.load_safety_result(safety_path)
    if len(cells) != int(spec["expected_cells"]):
        raise ValueError(f"{env} symbolic cell count mismatch: {len(cells)}")
    pool, pool_fingerprint = load_pool_artifact(data_root, env)
    splits = load_repeat_splits(data_root, env, repeat_seeds, pool_fingerprint)
    dims = tuple(spec["dims"])
    raw_metrics = strict_table_metrics(pool, grid, cells, dims)
    raw = _raw_metric_payload(raw_metrics, evaluation_size=len(pool))
    output_dir = Path(result_root) / env / "a_symbolic" / decoder
    raw_payload = {
        "env": env,
        "decoder": decoder,
        "construction": "symbolic",
        "inflated": False,
        "safety_path": str(safety_path),
        "safety_sha256": file_sha256(safety_path),
        "pool_fingerprint": pool_fingerprint,
        **raw,
        **single_result_summary(
            coverage=raw["coverage"],
            average_area=raw["average_area"],
        ),
    }
    write_json_artifact(output_dir / "raw_metrics.json", raw_payload, overwrite=overwrite)

    repeat_rows = []
    for seed_row, split in zip(repeat_seeds, splits):
        repeat = evaluate_inflated_repeat(
            split["val_traj"],
            split["test_traj"],
            grid,
            cells,
            dims,
            alpha=alpha,
        )
        repeat.update({
            "env": env,
            "decoder": decoder,
            "construction": "symbolic",
            "repeat_index": int(seed_row["repeat_index"]),
            "split_seed": int(seed_row["split_seed"]),
            "pool_fingerprint": pool_fingerprint,
        })
        repeat_rows.append(repeat)
        write_json_artifact(
            output_dir / f"repeat_{int(seed_row['repeat_index']):02d}.json",
            repeat,
            overwrite=overwrite,
        )
    aggregate = {
        "env": env,
        "decoder": decoder,
        "construction": "symbolic",
        "raw": single_result_summary(
            coverage=raw["coverage"], average_area=raw["average_area"]
        ),
        "inflated": aggregate_metric_rows(repeat_rows),
        "repeats": repeat_rows,
    }
    write_json_artifact(output_dir / "aggregate.json", aggregate, overwrite=overwrite)
    print(f"[Symbolic] {env}/{decoder}: {output_dir / 'aggregate.json'}")
    return aggregate


def evaluate_symbolic(
    environments: Sequence[str],
    decoders: Sequence[str],
    *,
    alpha: float,
    data_root: Path,
    result_root: Path,
    master_seed: int,
    overwrite: bool,
) -> None:
    seeds = derive_repeat_seeds(master_seed)
    for env in environments:
        for decoder in decoders:
            evaluate_symbolic_case(
                env,
                decoder,
                alpha=alpha,
                data_root=data_root,
                result_root=result_root,
                repeat_seeds=seeds,
                overwrite=overwrite,
            )


def validate_existing_sampled_outputs(
    sampling_config: dict[str, Any],
    *,
    expected_cells: int,
) -> None:
    from sampling import validate_sampled_safety_result

    for key in ("states_file", "trajectory_file", "tube_file"):
        if not Path(sampling_config[key]).is_file():
            raise FileNotFoundError(
                f"missing existing sampled artifact: {sampling_config[key]}"
            )
    payload = load_json(Path(sampling_config["tube_file"]))
    validate_sampled_safety_result(payload)
    if int(payload["seed"]) != int(sampling_config["seed"]):
        raise ValueError("existing sampled tube seed mismatch")
    if len(payload["cells"]) != int(expected_cells):
        raise ValueError("existing sampled tube cell count mismatch")


def sample_tubes(
    environments: Sequence[str],
    decoders: Sequence[str],
    repeat_indices: Sequence[int],
    *,
    data_root: Path,
    result_root: Path,
    master_seed: int,
    overwrite: bool,
) -> None:
    import sampling

    seeds = derive_repeat_seeds(master_seed)
    for env in environments:
        for repeat_index in repeat_indices:
            seed_row = seeds[repeat_index]
            for decoder in decoders:
                sampling_path = materialize_case_configs(
                    env,
                    decoder,
                    result_root=result_root,
                    overwrite=overwrite,
                )
                sampling_config = build_repeat_sampling_config(
                    load_json(sampling_path),
                    env=env,
                    decoder=decoder,
                    repeat_index=repeat_index,
                    tube_seed=int(seed_row["tube_seed"]),
                    data_root=data_root,
                    result_root=result_root,
                )
                tube_path = Path(sampling_config["tube_file"])
                if tube_path.exists() and not overwrite:
                    validate_existing_sampled_outputs(
                        sampling_config,
                        expected_cells=int(ENVIRONMENTS[env]["expected_cells"]),
                    )
                    print(f"[Sample] reuse {env}/{decoder}/repeat_{repeat_index:02d}: {tube_path}")
                    continue
                sampling.generate_sampled_tube(sampling_config, sampling_path)


def evaluate_sampled_case(
    env: str,
    decoder: str,
    *,
    alpha: float,
    data_root: Path,
    result_root: Path,
    repeat_seeds: Sequence[dict[str, int]],
    overwrite: bool,
) -> dict[str, Any]:
    spec = ENVIRONMENTS[env]
    pool, pool_fingerprint = load_pool_artifact(data_root, env)
    del pool
    splits = load_repeat_splits(data_root, env, repeat_seeds, pool_fingerprint)
    dims = tuple(spec["dims"])
    raw_rows = []
    inflated_rows = []
    for seed_row, split in zip(repeat_seeds, splits):
        repeat_index = int(seed_row["repeat_index"])
        repeat_dir = Path(result_root) / env / "b_sampled" / decoder / f"repeat_{repeat_index:02d}"
        tube_path = repeat_dir / "sampled_tube.json"
        grid, cells = stm.load_safety_result(tube_path)
        if len(cells) != int(spec["expected_cells"]):
            raise ValueError(f"{env} sampled cell count mismatch: {len(cells)}")
        raw_metrics = strict_table_metrics(split["test_traj"], grid, cells, dims)
        raw = _raw_metric_payload(raw_metrics, evaluation_size=TEST_SIZE)
        raw.update({
            "env": env,
            "decoder": decoder,
            "construction": "sampled",
            "repeat_index": repeat_index,
            "split_seed": int(seed_row["split_seed"]),
            "tube_seed": int(seed_row["tube_seed"]),
            "tube_path": str(tube_path),
            "tube_sha256": file_sha256(tube_path),
        })
        inflated = evaluate_inflated_repeat(
            split["val_traj"], split["test_traj"], grid, cells, dims, alpha=alpha
        )
        inflated.update({
            "env": env,
            "decoder": decoder,
            "construction": "sampled",
            "repeat_index": repeat_index,
            "split_seed": int(seed_row["split_seed"]),
            "tube_seed": int(seed_row["tube_seed"]),
            "tube_path": str(tube_path),
        })
        calibration = {
            key: inflated[key]
            for key in (
                "gamma", "signed_margin_at_rank", "rank", "alpha",
                "calibration_size", "split_seed", "tube_seed", "repeat_index",
            )
        }
        raw_rows.append(raw)
        inflated_rows.append(inflated)
        write_json_artifact(repeat_dir / "raw_metrics.json", raw, overwrite=overwrite)
        write_json_artifact(repeat_dir / "calibration.json", calibration, overwrite=overwrite)
        write_json_artifact(repeat_dir / "inflated_metrics.json", inflated, overwrite=overwrite)

    aggregate = {
        "env": env,
        "decoder": decoder,
        "construction": "sampled",
        "raw": aggregate_metric_rows(raw_rows),
        "inflated": aggregate_metric_rows(inflated_rows),
        "raw_repeats": raw_rows,
        "inflated_repeats": inflated_rows,
    }
    output_path = Path(result_root) / env / "b_sampled" / decoder / "aggregate.json"
    write_json_artifact(output_path, aggregate, overwrite=overwrite)
    print(f"[Sampled Evaluation] {env}/{decoder}: {output_path}")
    return aggregate


def evaluate_sampled(
    environments: Sequence[str],
    decoders: Sequence[str],
    *,
    alpha: float,
    data_root: Path,
    result_root: Path,
    master_seed: int,
    overwrite: bool,
) -> None:
    seeds = derive_repeat_seeds(master_seed)
    for env in environments:
        for decoder in decoders:
            evaluate_sampled_case(
                env,
                decoder,
                alpha=alpha,
                data_root=data_root,
                result_root=result_root,
                repeat_seeds=seeds,
                overwrite=overwrite,
            )


def evaluate_predictor_case(
    env: str,
    tube_paths: Sequence[Path],
    *,
    alpha: float,
    data_root: Path,
    result_root: Path,
    repeat_seeds: Sequence[dict[str, int]],
    overwrite: bool,
) -> dict[str, Any]:
    """Evaluate five existing predictor tubes against the five saved splits."""
    if len(tube_paths) != NUM_REPEATS:
        raise ValueError(f"expected exactly {NUM_REPEATS} predictor tubes")
    spec = ENVIRONMENTS[env]
    pool, pool_fingerprint = load_pool_artifact(data_root, env)
    del pool
    splits = load_repeat_splits(data_root, env, repeat_seeds, pool_fingerprint)
    dims = tuple(spec["dims"])
    raw_rows = []
    inflated_rows = []
    for seed_row, split, tube_path in zip(repeat_seeds, splits, tube_paths):
        repeat_index = int(seed_row["repeat_index"])
        tube_path = Path(tube_path)
        payload = load_json(tube_path)
        if payload.get("method") != "transformer_sampled_minmax_envelope":
            raise ValueError(f"unexpected predictor tube method: {tube_path}")
        grid, cells = stm.load_safety_result(tube_path)
        if len(cells) != int(spec["expected_cells"]):
            raise ValueError(f"{env} predictor cell count mismatch: {len(cells)}")
        histories = {len(cell.get("bounds", [])) for cell in cells}
        if histories != {int(spec["expected_horizon"])}:
            raise ValueError(f"{env} predictor horizon mismatch: {sorted(histories)}")

        raw_metrics = strict_table_metrics(split["test_traj"], grid, cells, dims)
        raw = _raw_metric_payload(raw_metrics, evaluation_size=TEST_SIZE)
        common = {
            "env": env,
            "construction": "predictor",
            "repeat_index": repeat_index,
            "split_seed": int(seed_row["split_seed"]),
            "predictor_tube_index": repeat_index,
            "predictor_tube_path": str(tube_path),
            "predictor_tube_sha256": file_sha256(tube_path),
        }
        raw.update(common)
        inflated = evaluate_inflated_repeat(
            split["val_traj"], split["test_traj"], grid, cells, dims, alpha=alpha
        )
        inflated.update(common)
        calibration = {
            key: inflated[key]
            for key in (
                "gamma", "signed_margin_at_rank", "rank", "alpha",
                "calibration_size", "split_seed", "repeat_index",
                "predictor_tube_index", "predictor_tube_path",
            )
        }
        raw_rows.append(raw)
        inflated_rows.append(inflated)
        repeat_dir = Path(result_root) / env / "c_predictor" / f"repeat_{repeat_index:02d}"
        write_json_artifact(repeat_dir / "raw_metrics.json", raw, overwrite=overwrite)
        write_json_artifact(repeat_dir / "calibration.json", calibration, overwrite=overwrite)
        write_json_artifact(repeat_dir / "inflated_metrics.json", inflated, overwrite=overwrite)

    aggregate = {
        "env": env,
        "construction": "predictor",
        "raw": aggregate_metric_rows(raw_rows),
        "inflated": aggregate_metric_rows(inflated_rows),
        "raw_repeats": raw_rows,
        "inflated_repeats": inflated_rows,
    }
    output_path = Path(result_root) / env / "c_predictor" / "aggregate.json"
    write_json_artifact(output_path, aggregate, overwrite=overwrite)
    print(f"[Predictor Evaluation] {env}: {output_path}")
    return aggregate


def evaluate_predictor(
    environments: Sequence[str],
    *,
    alpha: float,
    data_root: Path,
    result_root: Path,
    master_seed: int,
    overwrite: bool,
) -> None:
    repeat_seeds = derive_repeat_seeds(master_seed)
    for env in environments:
        evaluate_predictor_case(
            env,
            predictor_tube_paths(env),
            alpha=alpha,
            data_root=data_root,
            result_root=result_root,
            repeat_seeds=repeat_seeds,
            overwrite=overwrite,
        )
    write_predictor_summary(environments, result_root, overwrite=overwrite)


def write_predictor_summary(
    environments: Sequence[str],
    result_root: Path,
    *,
    overwrite: bool,
) -> None:
    """Write raw and paper-formatted summaries for predictor results."""
    rows = []
    for env in environments:
        aggregate = load_json(Path(result_root) / env / "c_predictor" / "aggregate.json")
        for reachability, metrics in (
            ("TP", aggregate["raw"]),
            ("TP (inflate)", aggregate["inflated"]),
        ):
            rows.append({"env": env, "reachability": reachability, **metrics})

    raw_fields = [
        "env", "reachability", "coverage_mean", "coverage_std",
        "area_mean", "area_std", "num_repeats", "std_ddof",
    ]
    raw_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(raw_buffer, fieldnames=raw_fields)
    writer.writeheader()
    writer.writerows(rows)
    write_text_artifact(
        Path(result_root) / "predictor_metrics_raw.csv",
        raw_buffer.getvalue(),
        overwrite=overwrite,
    )

    formatted_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        formatted_buffer,
        fieldnames=["env", "reachability", "coverage_percent", "area_x1e3"],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "env": row["env"],
            "reachability": row["reachability"],
            "coverage_percent": (
                f"{100.0 * float(row['coverage_mean']):.2f} ± "
                f"{100.0 * float(row['coverage_std']):.2f}"
            ),
            "area_x1e3": (
                f"{1000.0 * float(row['area_mean']):.3f} ± "
                f"{1000.0 * float(row['area_std']):.3f}"
            ),
        })
    write_text_artifact(
        Path(result_root) / "predictor_metrics_formatted.csv",
        formatted_buffer.getvalue(),
        overwrite=overwrite,
    )
    print(f"[Predictor Summary] {Path(result_root) / 'predictor_metrics_formatted.csv'}")


def preflight(environments: Sequence[str], decoders: Sequence[str]) -> dict[str, Any]:
    """Validate fixed inputs and model/grid configs without running sampling."""
    from sampling import grid_cell_bounds, validate_cellwise_setup
    from utils import load_config

    report: dict[str, Any] = {}
    for env in environments:
        spec = ENVIRONMENTS[env]
        real_path = Path(spec["real"])
        if not real_path.is_file():
            raise FileNotFoundError(real_path)
        with np.load(real_path, allow_pickle=False) as data:
            pool = combine_real_pool(data["val_traj"], data["test_traj"])
        if pool.shape[1] != int(spec["expected_horizon"]):
            raise ValueError(f"{env} real trajectory horizon mismatch")
        env_report = {"real": str(real_path), "pool_shape": list(pool.shape), "cases": {}}
        for decoder in decoders:
            safety_path = Path(spec["symbolic"][decoder])
            if not safety_path.is_file():
                raise FileNotFoundError(safety_path)
            grid, cells = stm.load_safety_result(safety_path)
            if len(cells) != int(spec["expected_cells"]):
                raise ValueError(f"{env}/{decoder} symbolic cell count mismatch")
            strict_table_metrics(pool, grid, cells, tuple(spec["dims"]))

            sampling_path = Path(spec["sampling_configs"][decoder])
            sampling_config = load_config(sampling_path)
            model_starv = load_config(
                PROJECT_ROOT / sampling_config["starv_config"]
            )
            canonical = load_config(
                PROJECT_ROOT / sampling_config["grid_config"]
            )
            sampling_config, model_starv, canonical = apply_experiment_overrides(
                env, decoder, sampling_config, model_starv, canonical
            )
            metadata = validate_cellwise_setup(
                sampling_config,
                model_starv,
                canonical,
            )
            cell_count = len(grid_cell_bounds(canonical["grid"]))
            if cell_count != int(spec["expected_cells"]):
                raise ValueError(
                    f"{env}/{decoder} sampled grid has {cell_count} cells; "
                    f"expected {spec['expected_cells']}"
                )
            env_report["cases"][decoder] = {
                "symbolic_safety": str(safety_path),
                "symbolic_sha256": file_sha256(safety_path),
                "cell_count": cell_count,
                "model_id": metadata["model_id"],
                "model_type": metadata["model_type"],
                "decoder_weights": metadata["decoder_weights"],
                "controller_weights": metadata["controller_weights"],
                "z_range": metadata.get("z_range"),
            }
        report[env] = env_report
        print(f"[Preflight] {env}: pool={pool.shape}, cells={spec['expected_cells']}, cases={list(decoders)}")
    return report


def aggregate_results(
    environments: Sequence[str],
    decoders: Sequence[str],
    *,
    master_seed: int,
    result_root: Path,
    overwrite: bool,
) -> None:
    rows = []
    nested: dict[str, Any] = {}
    for env in environments:
        nested[env] = {}
        for decoder in decoders:
            symbolic = load_json(Path(result_root) / env / "a_symbolic" / decoder / "aggregate.json")
            sampled = load_json(Path(result_root) / env / "b_sampled" / decoder / "aggregate.json")
            nested[env][decoder] = {"symbolic": symbolic, "sampled": sampled}
            for section, reachability, metrics in (
                ("a_symbolic", "Sym", symbolic["raw"]),
                ("a_symbolic", "Sym (inflate)", symbolic["inflated"]),
                ("b_sampled", "Smp", sampled["raw"]),
                ("b_sampled", "Smp (inflate)", sampled["inflated"]),
            ):
                rows.append({
                    "section": section,
                    "env": env,
                    "decoder": decoder,
                    "reachability": reachability,
                    **metrics,
                })
    payload = {
        "master_seed": int(master_seed),
        "num_repeats": NUM_REPEATS,
        "std_ddof": STD_DDOF,
        "environments": nested,
    }
    write_json_artifact(Path(result_root) / "metrics.json", payload, overwrite=overwrite)
    raw_fields = [
        "section", "env", "decoder", "reachability", "coverage_mean",
        "coverage_std", "area_mean", "area_std", "num_repeats", "std_ddof",
    ]
    raw_path = Path(result_root) / "metrics_raw.csv"
    formatted_path = Path(result_root) / "metrics_formatted.csv"
    raw_buffer = io.StringIO(newline="")
    raw_writer = csv.DictWriter(raw_buffer, fieldnames=raw_fields)
    raw_writer.writeheader()
    raw_writer.writerows(rows)
    write_text_artifact(raw_path, raw_buffer.getvalue(), overwrite=overwrite)

    formatted_buffer = io.StringIO(newline="")
    formatted_writer = csv.DictWriter(
        formatted_buffer,
        fieldnames=["section", "env", "decoder", "reachability", "coverage_percent", "area_x1e3"],
    )
    formatted_writer.writeheader()
    for row in rows:
        coverage = f"{100.0 * float(row['coverage_mean']):.2f}"
        area = f"{1000.0 * float(row['area_mean']):.3f}"
        if row.get("coverage_std") is not None:
            coverage += f" ± {100.0 * float(row['coverage_std']):.2f}"
        if row.get("area_std") is not None:
            area += f" ± {1000.0 * float(row['area_std']):.3f}"
        formatted_writer.writerow({
            "section": row["section"],
            "env": row["env"],
            "decoder": row["decoder"],
            "reachability": row["reachability"],
            "coverage_percent": coverage,
            "area_x1e3": area,
        })
    write_text_artifact(formatted_path, formatted_buffer.getvalue(), overwrite=overwrite)
    manifest = {
        "master_seed": int(master_seed),
        "num_repeats": NUM_REPEATS,
        "std_ddof": STD_DDOF,
        "shared_safety_root": str(SHARED_SAFETY_ROOT),
        "environments": list(environments),
        "decoders": list(decoders),
        "brake_weight_path_migration": {
            "symbolic_recorded_prefix": "dwm_weight/now_weight/brake_system",
            "current_prefix": "dwm_weight/brake_system",
            "confirmed_same_weights": True,
        },
    }
    write_json_artifact(Path(result_root) / "experiment_manifest.json", manifest, overwrite=overwrite)
    print(f"[Aggregate] {formatted_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preflight", "prepare-splits", "evaluate-symbolic", "sample",
            "evaluate-sampled", "evaluate-predictor", "aggregate", "run-all",
        ),
    )
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--env", nargs="+", choices=tuple(ENVIRONMENTS), default=list(ENVIRONMENTS))
    parser.add_argument("--decoder", nargs="+", choices=DECODERS, default=list(DECODERS))
    parser.add_argument("--repeat", nargs="+", type=int, default=list(range(NUM_REPEATS)))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if any(index < 0 or index >= NUM_REPEATS for index in args.repeat):
        raise ValueError(f"repeat indices must be between 0 and {NUM_REPEATS - 1}")
    default_data_root, default_result_root = experiment_roots(args.master_seed)
    data_root = args.data_root or default_data_root
    result_root = args.result_root or default_result_root

    if args.command in {"preflight", "run-all"}:
        report = preflight(args.env, args.decoder)
        write_json_artifact(
            result_root / "preflight.json",
            report,
            overwrite=args.overwrite,
        )
    if args.command in {"prepare-splits", "run-all"}:
        prepare_splits(
            args.env,
            master_seed=args.master_seed,
            data_root=data_root,
            overwrite=args.overwrite,
        )
    if args.command in {"evaluate-symbolic", "run-all"}:
        evaluate_symbolic(
            args.env,
            args.decoder,
            alpha=args.alpha,
            data_root=data_root,
            result_root=result_root,
            master_seed=args.master_seed,
            overwrite=args.overwrite,
        )
    if args.command in {"sample", "run-all"}:
        sample_tubes(
            args.env,
            args.decoder,
            args.repeat,
            data_root=data_root,
            result_root=result_root,
            master_seed=args.master_seed,
            overwrite=args.overwrite,
        )
    if args.command in {"evaluate-sampled", "run-all"}:
        if sorted(set(args.repeat)) != list(range(NUM_REPEATS)):
            raise ValueError("evaluate-sampled requires all five repeats")
        evaluate_sampled(
            args.env,
            args.decoder,
            alpha=args.alpha,
            data_root=data_root,
            result_root=result_root,
            master_seed=args.master_seed,
            overwrite=args.overwrite,
        )
    if args.command == "evaluate-predictor":
        if sorted(set(args.repeat)) != list(range(NUM_REPEATS)):
            raise ValueError("evaluate-predictor requires all five repeats")
        evaluate_predictor(
            args.env,
            alpha=args.alpha,
            data_root=data_root,
            result_root=result_root,
            master_seed=args.master_seed,
            overwrite=args.overwrite,
        )
    if args.command in {"aggregate", "run-all"}:
        aggregate_results(
            args.env,
            args.decoder,
            master_seed=args.master_seed,
            result_root=result_root,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
