#!/usr/bin/env python3
"""Run any number of raw or inflated tube evaluations and aggregate them."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import evaluate_tube as evaluation
import inflate_tube as inflation


PROJECT_ROOT = Path(__file__).resolve().parent
STD_DDOF = 0
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _default_repeats(env: str, cell_prefix: str) -> list[dict[str, Any]]:
    tube = PROJECT_ROOT / f"safety_results/{env}/{cell_prefix}_dwm_safety_result.json"
    real = PROJECT_ROOT / f"safety_results/{env}/{cell_prefix}_real_trajectories.npz"
    return [
        {
            "name": "repeat_0",
            "group": "raw",
            "inflate": False,
            "tube": tube,
            "evaluation_real": real,
            "evaluation_key": "test_traj",
        },
        {
            "name": "repeat_0",
            "group": "inflated",
            "inflate": True,
            "tube": tube,
            "calibration_real": real,
            "calibration_key": "val_traj",
            "evaluation_real": real,
            "evaluation_key": "test_traj",
            "alpha": 0.05,
        },
    ]


REPEAT_CONFIGS: dict[str, dict[str, Any]] = {
    "cartpole": {
        "check_dims": (0, 2),
        "output_dir": PROJECT_ROOT / "results/repeated_tube_evaluation/cartpole",
        "repeats": _default_repeats("cartpole", "3600cell"),
    },
    "mountain_car": {
        "check_dims": (0, 1),
        "output_dir": PROJECT_ROOT / "results/repeated_tube_evaluation/mountain_car",
        "repeats": _default_repeats("mountain_car", "6400cell"),
    },
    "pendulum": {
        "check_dims": (0, 1),
        "output_dir": PROJECT_ROOT / "results/repeated_tube_evaluation/pendulum",
        "repeats": _default_repeats("pendulum", "5000cell"),
    },
    "brake_system": {
        "check_dims": (0, 1),
        "output_dir": PROJECT_ROOT / "results/repeated_tube_evaluation/brake_system",
        "repeats": _default_repeats("brake_system", "1600cell"),
    },
}

# Select one default environment. Add or remove entries in its `repeats` list.
ACTIVE_ENV = "cartpole"
# ACTIVE_ENV = "mountain_car"
# ACTIVE_ENV = "pendulum"
# ACTIVE_ENV = "brake_system"


def aggregate_values(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(
            "aggregate values must be a nonempty finite one-dimensional sequence"
        )
    return {
        "mean": float(np.mean(array)),
        "variance": float(np.var(array, ddof=STD_DDOF)),
        "std": float(np.std(array, ddof=STD_DDOF)),
        "count": int(len(array)),
        "ddof": STD_DDOF,
    }


def validate_repeats(repeats: Sequence[dict[str, Any]]) -> None:
    if not repeats:
        raise ValueError("at least one repeat is required")
    seen: set[tuple[str, str]] = set()
    for index, repeat in enumerate(repeats):
        required = {"name", "group", "inflate", "tube", "evaluation_real"}
        missing = required.difference(repeat)
        if missing:
            raise ValueError(
                f"repeat {index} is missing fields: {sorted(missing)}"
            )
        name = str(repeat["name"])
        group = str(repeat["group"])
        if not SAFE_NAME.fullmatch(name) or not SAFE_NAME.fullmatch(group):
            raise ValueError(
                "repeat name and group may contain only letters, numbers, '.', '-' and '_'"
            )
        identity = (group, name)
        if identity in seen:
            raise ValueError(f"duplicate repeat in group {group!r}: {name!r}")
        seen.add(identity)
        if not isinstance(repeat["inflate"], bool):
            raise ValueError(f"repeat {group}/{name}: inflate must be bool")
        if repeat["inflate"]:
            missing_inflation = {"calibration_real"}.difference(repeat)
            if missing_inflation:
                raise ValueError(
                    f"repeat {group}/{name} is missing fields: "
                    f"{sorted(missing_inflation)}"
                )
            alpha = float(repeat.get("alpha", 0.05))
            if not 0.0 < alpha < 1.0:
                raise ValueError(
                    f"repeat {group}/{name}: alpha must be between 0 and 1"
                )


def _path(value: Any) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _repeat_dir(output_dir: Path, repeat: dict[str, Any]) -> Path:
    return Path(output_dir) / str(repeat["group"]) / str(repeat["name"])


def _output_paths(
    repeats: Sequence[dict[str, Any]], output_dir: Path
) -> list[Path]:
    paths = [
        Path(output_dir) / "repeat_metrics.csv",
        Path(output_dir) / "aggregate_metrics.csv",
        Path(output_dir) / "repeat_results.json",
    ]
    for repeat in repeats:
        repeat_dir = _repeat_dir(output_dir, repeat)
        paths.append(repeat_dir / "metrics.json")
        if repeat["inflate"]:
            paths.extend((
                repeat_dir / "inflated_tube.json",
                repeat_dir / "calibration.json",
            ))
    return paths


def _require_available_outputs(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing output(s): " + ", ".join(existing)
        )


def _require_inputs(repeats: Sequence[dict[str, Any]]) -> None:
    missing = []
    for repeat in repeats:
        paths = [repeat["tube"], repeat["evaluation_real"]]
        if repeat["inflate"]:
            paths.append(repeat["calibration_real"])
        missing.extend(str(_path(path)) for path in paths if not _path(path).is_file())
    if missing:
        raise FileNotFoundError(
            "missing repeat input file(s): " + ", ".join(sorted(set(missing)))
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_repeat(
    repeat: dict[str, Any],
    *,
    dims: Sequence[int],
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    repeat_dir = _repeat_dir(output_dir, repeat)
    source_tube = _path(repeat["tube"])
    evaluated_tube = source_tube
    calibration_path: Path | None = None
    gamma: float | None = None
    if repeat["inflate"]:
        evaluated_tube = repeat_dir / "inflated_tube.json"
        calibration_path = repeat_dir / "calibration.json"
        calibration = inflation.run_inflation(
            tube_path=source_tube,
            calibration_path=_path(repeat["calibration_real"]),
            calibration_key=str(repeat.get("calibration_key", "val_traj")),
            dims=dims,
            alpha=float(repeat.get("alpha", 0.05)),
            output_path=evaluated_tube,
            calibration_output_path=calibration_path,
            overwrite=overwrite,
        )
        gamma = float(calibration["gamma"])

    _, grid, cells = evaluation.load_tube(evaluated_tube)
    evaluation_real = _path(repeat["evaluation_real"])
    evaluation_key = str(repeat.get("evaluation_key", "test_traj"))
    trajectories = evaluation.load_trajectories(
        evaluation_real, evaluation_key
    )
    metrics, _ = evaluation.calculate_metrics(
        trajectories, grid, cells, dims
    )
    result: dict[str, Any] = {
        "name": str(repeat["name"]),
        "group": str(repeat["group"]),
        "mode": "inflated" if repeat["inflate"] else "raw",
        "coverage": float(metrics["coverage"]),
        "area": float(metrics["area"]),
        "source_tube": str(source_tube),
        "evaluated_tube": str(evaluated_tube),
        "evaluation_real": str(evaluation_real),
        "evaluation_key": evaluation_key,
    }
    if calibration_path is not None:
        result["calibration"] = str(calibration_path)
        result["gamma"] = gamma
    _write_json(repeat_dir / "metrics.json", result)
    return result


def aggregate_groups(
    results: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["group"])].append(result)
    return {
        group: {
            "num_repeats": len(rows),
            "coverage": aggregate_values([row["coverage"] for row in rows]),
            "area": aggregate_values([row["area"] for row in rows]),
        }
        for group, rows in grouped.items()
    }


def write_repeat_table(path: Path, results: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Group", "Repeat", "Mode", "Coverage", "Area", "Evaluated Tube"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "Group": result["group"],
                "Repeat": result["name"],
                "Mode": result["mode"],
                "Coverage": f"{100.0 * result['coverage']:.2f}%",
                "Area": f"{result['area']:.12g}",
                "Evaluated Tube": result["evaluated_tube"],
            })


def write_aggregate_table(
    path: Path, groups: dict[str, dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Group", "Repeats",
        "Coverage Mean", "Coverage Variance (%²)", "Coverage Std",
        "Area Mean", "Area Variance", "Area Std",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for group, metrics in groups.items():
            coverage = metrics["coverage"]
            area = metrics["area"]
            writer.writerow({
                "Group": group,
                "Repeats": metrics["num_repeats"],
                "Coverage Mean": f"{100.0 * coverage['mean']:.2f}%",
                "Coverage Variance (%²)": f"{10000.0 * coverage['variance']:.6f}",
                "Coverage Std": f"{100.0 * coverage['std']:.2f}%",
                "Area Mean": f"{area['mean']:.12g}",
                "Area Variance": f"{area['variance']:.12g}",
                "Area Std": f"{area['std']:.12g}",
            })


def run_repeated_evaluation(
    *,
    repeats: Sequence[dict[str, Any]],
    dims: Sequence[int],
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    repeats = list(repeats)
    dims = tuple(int(dim) for dim in dims)
    if len(dims) != 2:
        raise ValueError("exactly two check dimensions are required")
    validate_repeats(repeats)
    output_dir = Path(output_dir)
    _require_inputs(repeats)
    _require_available_outputs(_output_paths(repeats, output_dir), overwrite)

    results = []
    for repeat in repeats:
        result = evaluate_repeat(
            repeat,
            dims=dims,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        results.append(result)
    groups = aggregate_groups(results)
    payload = {
        "check_dims": list(dims),
        "std_ddof": STD_DDOF,
        "repeats": results,
        "groups": groups,
    }
    write_repeat_table(output_dir / "repeat_metrics.csv", results)
    write_aggregate_table(output_dir / "aggregate_metrics.csv", groups)
    _write_json(output_dir / "repeat_results.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=tuple(REPEAT_CONFIGS), default=ACTIVE_ENV)
    parser.add_argument("--check-dims", type=int, nargs=2, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = REPEAT_CONFIGS[args.env]
    payload = run_repeated_evaluation(
        repeats=config["repeats"],
        dims=tuple(args.check_dims or config["check_dims"]),
        output_dir=args.outdir or config["output_dir"],
        overwrite=args.overwrite,
    )
    for group, metrics in payload["groups"].items():
        coverage = metrics["coverage"]
        area = metrics["area"]
        print(
            f"{group}: coverage={100.0 * coverage['mean']:.2f}% "
            f"± {100.0 * coverage['std']:.2f}%, "
            f"area={area['mean']:.12g} ± {area['std']:.12g}, "
            f"n={metrics['num_repeats']}"
        )


if __name__ == "__main__":
    main()
