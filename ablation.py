from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import train_decoder as td
from utils import load_config, resolve_device, set_seed


DEFAULT_ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DEFAULT_LAMBDAS = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5)
DEFAULT_SEED = 2025
GRID_ROOT = Path("dwm_weight/now_weight")


def format_value(value: float) -> str:
    return format(float(value), ".12g")


@dataclass(frozen=True)
class Experiment:
    env: str
    alpha: float
    lambda_ctrl: float
    seed: int = DEFAULT_SEED
    weight_root: Path = GRID_ROOT
    condition: str = "default"
    base_config_path: Path | None = None

    @property
    def output_dir(self) -> Path:
        root = self.weight_root / self.env / "alpha_lambda_grid"
        if self.condition != "default":
            root = root / self.condition
        return root / f"alpha_{format_value(self.alpha)}" / f"lambda_{format_value(self.lambda_ctrl)}" / f"seed_{self.seed}"

    @property
    def best_checkpoint(self) -> Path:
        return self.output_dir / "decoder_best_total.pth"

    @property
    def last_checkpoint(self) -> Path:
        return self.output_dir / "decoder_last.pth"

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / "metrics.json"

    @property
    def trajectory_path(self) -> Path:
        return self.output_dir / "dwm_trajectories_saliency.npz"

    def as_row(self) -> dict:
        return {
            "env": self.env,
            "alpha": self.alpha,
            "lambda_ctrl": self.lambda_ctrl,
            "seed": self.seed,
            "condition": self.condition,
            "output_dir": self.output_dir.as_posix(),
        }


def build_experiments(
    env: str,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    seed: int = DEFAULT_SEED,
    weight_root: Path = GRID_ROOT,
    condition: str = "default",
    base_config_path: Path | None = None,
) -> list[Experiment]:
    if env not in {"cartpole", "mountain_car", "pendulum"}:
        raise ValueError(f"Unsupported ablation environment: {env}")
    return [
        Experiment(
            env,
            float(alpha),
            float(lambda_ctrl),
            int(seed),
            Path(weight_root),
            condition,
            Path(base_config_path) if base_config_path is not None else None,
        )
        for alpha, lambda_ctrl in product(alphas, lambdas)
    ]


def build_train_config(
    experiment: Experiment,
    base_config: Mapping | None = None,
) -> dict:
    source = base_config
    if source is None:
        source = load_config(
            experiment.base_config_path
            or Path("config/train_decoder") / experiment.env / "saliency.json"
        )
    config = copy.deepcopy(dict(source))
    if config.get("weight_mode") != "saliency":
        raise ValueError("Alpha-lambda grid requires weight_mode='saliency'")
    config["weight"]["alpha"] = experiment.alpha
    config["lambda_ctrl"] = experiment.lambda_ctrl
    config["training"]["seed"] = experiment.seed
    config["output_dir"] = experiment.output_dir.as_posix()
    return config


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def validate_training_artifacts(experiment: Experiment) -> tuple[bool, str]:
    required = [
        experiment.metrics_path,
        experiment.best_checkpoint,
        experiment.last_checkpoint,
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        return False, f"missing: {', '.join(missing)}"

    try:
        metrics = json.loads(experiment.metrics_path.read_text(encoding="utf-8"))
        config = metrics["config"]
        matches = (
            config["weight_mode"] == "saliency"
            and float(config["weight"]["alpha"]) == experiment.alpha
            and float(config["lambda_ctrl"]) == experiment.lambda_ctrl
            and int(config["training"]["seed"]) == experiment.seed
            and _same_path(config["output_dir"], experiment.output_dir)
            and metrics["best_checkpoint"] == experiment.best_checkpoint.name
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return False, f"invalid metrics: {exc}"

    return (True, "complete") if matches else (False, "config mismatch")


def run_training_grid(
    env: str,
    *,
    experiments: Sequence[Experiment] | None = None,
    skip_existing: bool = True,
    continue_on_error: bool = True,
    trainer=None,
) -> pd.DataFrame:
    selected = list(build_experiments(env) if experiments is None else experiments)
    train_fn = trainer or td.train
    rows = []

    for experiment in selected:
        valid, _ = validate_training_artifacts(experiment)
        if skip_existing and valid:
            rows.append({**experiment.as_row(), "status": "skipped", "error": ""})
            continue

        try:
            config = build_train_config(experiment)
            set_seed(experiment.seed)
            device = resolve_device(config.get("device", "auto"))
            train_fn(config, device)
            valid, reason = validate_training_artifacts(experiment)
            if not valid:
                raise RuntimeError(f"training artifact validation failed: {reason}")
            rows.append({**experiment.as_row(), "status": "trained", "error": ""})
        except Exception as exc:
            rows.append(
                {**experiment.as_row(), "status": "failed", "error": str(exc)}
            )
            if not continue_on_error:
                raise

    return pd.DataFrame(rows)


def build_sampling_config(
    experiment: Experiment,
    base_config: Mapping | None = None,
) -> dict:
    source = base_config
    if source is None:
        source = load_config(Path("config/sampling") / f"{experiment.env}.json")
    config = copy.deepcopy(dict(source))
    config["decoder"]["variant"] = "saliency"
    config["decoder"]["weights"] = experiment.best_checkpoint.as_posix()
    config["output_dir"] = experiment.output_dir.as_posix()
    return config


def _starv_states_path_for_env(env: str) -> Path:
    sampling_config = load_config(Path("config/sampling") / f"{env}.json")
    starv_config = load_config(sampling_config["starv_config"])
    return Path(starv_config["starv_states"]["output_file"])


def _validate_trajectory_file(
    trajectory_path: Path,
    *,
    variant: str,
    decoder_weights: str | Path,
    states_path: Path,
) -> tuple[bool, str]:
    if not trajectory_path.exists():
        return False, "missing trajectory"

    try:
        with np.load(states_path, allow_pickle=False) as states, np.load(
            trajectory_path,
            allow_pickle=False,
        ) as trajectory:
            if trajectory["variant"].item() != variant:
                return False, "variant mismatch"
            if not _same_path(
                trajectory["decoder_weights"].item(),
                decoder_weights,
            ):
                return False, "checkpoint mismatch"
            for split in ("train", "val", "test"):
                expected = states[f"{split}_states"]
                actual = trajectory[f"{split}_traj"][:, 0, :]
                if not np.array_equal(expected, actual):
                    return False, f"{split} initial state mismatch"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"invalid trajectory: {exc}"

    return True, "complete"


def validate_rollout_artifact(
    experiment: Experiment,
    *,
    states_path: Path | None = None,
) -> tuple[bool, str]:
    return _validate_trajectory_file(
        experiment.trajectory_path,
        variant="saliency",
        decoder_weights=experiment.best_checkpoint,
        states_path=states_path or _starv_states_path_for_env(experiment.env),
    )


def run_rollout_grid(
    env: str,
    *,
    experiments: Sequence[Experiment] | None = None,
    skip_existing: bool = True,
    continue_on_error: bool = True,
    generator=None,
) -> pd.DataFrame:
    selected = list(build_experiments(env) if experiments is None else experiments)
    if generator is None:
        from sampling import generate_dataset

        generate_fn = generate_dataset
    else:
        generate_fn = generator
    rows = []

    for experiment in selected:
        trained, reason = validate_training_artifacts(experiment)
        if not trained:
            rows.append(
                {**experiment.as_row(), "status": "failed", "error": reason}
            )
            if not continue_on_error:
                raise RuntimeError(reason)
            continue

        valid, _ = validate_rollout_artifact(experiment)
        if skip_existing and valid:
            rows.append({**experiment.as_row(), "status": "skipped", "error": ""})
            continue

        try:
            generate_fn(build_sampling_config(experiment))
            valid, reason = validate_rollout_artifact(experiment)
            if not valid:
                raise RuntimeError(f"rollout artifact validation failed: {reason}")
            rows.append(
                {**experiment.as_row(), "status": "generated", "error": ""}
            )
        except Exception as exc:
            rows.append(
                {**experiment.as_row(), "status": "failed", "error": str(exc)}
            )
            if not continue_on_error:
                raise

    return pd.DataFrame(rows)


def run_mainline_rollouts(
    env: str,
    *,
    variants: Sequence[str] = ("intensity", "saliency"),
    skip_existing: bool = True,
    continue_on_error: bool = True,
    generator=None,
) -> pd.DataFrame:
    base_config = load_config(Path("config/sampling") / f"{env}.json")
    states_path = _starv_states_path_for_env(env)
    if generator is None:
        from sampling import generate_dataset

        generate_fn = generate_dataset
    else:
        generate_fn = generator
    rows = []

    for variant in variants:
        config = copy.deepcopy(base_config)
        weights = config["decoder"]["weights"]
        if not isinstance(weights, Mapping) or variant not in weights:
            raise KeyError(f"sampling config has no decoder weights for {variant!r}")
        checkpoint = weights[variant]
        config["decoder"]["variant"] = variant
        trajectory_path = (
            Path(config["output_dir"]) / f"dwm_trajectories_{variant}.npz"
        )
        valid, _ = _validate_trajectory_file(
            trajectory_path,
            variant=variant,
            decoder_weights=checkpoint,
            states_path=states_path,
        )
        row = {"env": env, "variant": variant, "decoder_weights": checkpoint}
        if skip_existing and valid:
            rows.append({**row, "status": "skipped", "error": ""})
            continue

        try:
            generate_fn(config)
            valid, reason = _validate_trajectory_file(
                trajectory_path,
                variant=variant,
                decoder_weights=checkpoint,
                states_path=states_path,
            )
            if not valid:
                raise RuntimeError(f"rollout artifact validation failed: {reason}")
            rows.append({**row, "status": "generated", "error": ""})
        except Exception as exc:
            rows.append({**row, "status": "failed", "error": str(exc)})
            if not continue_on_error:
                raise

    return pd.DataFrame(rows)


def compute_l2_metrics(
    real: np.ndarray,
    dwm: np.ndarray,
    *,
    circular_dims: Sequence[int] = (),
    period: float = 2 * np.pi,
) -> dict[str, float]:
    real = np.asarray(real, dtype=float)
    dwm = np.asarray(dwm, dtype=float)
    if real.shape != dwm.shape or real.ndim != 3:
        raise ValueError(
            f"trajectory shape mismatch: real={real.shape}, dwm={dwm.shape}"
        )
    if not np.array_equal(real[:, 0, :], dwm[:, 0, :]):
        raise ValueError("real and DWM initial states do not match")

    diff = dwm - real
    if circular_dims:
        diff = diff.copy()
        for dim in circular_dims:
            values = diff[..., dim]
            diff[..., dim] = (values + period / 2) % period - period / 2

    distance = np.linalg.norm(diff, ord=2, axis=-1)
    maximum = distance.max(axis=1)
    return {
        "mean_step_l2": float(distance.mean()),
        "final_l2": float(distance[:, -1].mean()),
        "max_l2_mean": float(maximum.mean()),
        "max_l2_p95": float(np.percentile(maximum, 95)),
    }


def _artifact_missing(experiment: Experiment, path: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"alpha={format_value(experiment.alpha)}, "
        f"lambda_ctrl={format_value(experiment.lambda_ctrl)}: missing {path}"
    )


def collect_training_metrics(
    experiments: Sequence[Experiment],
) -> pd.DataFrame:
    rows = []
    for experiment in experiments:
        for path in (
            experiment.metrics_path,
            experiment.best_checkpoint,
            experiment.last_checkpoint,
        ):
            if not path.exists():
                raise _artifact_missing(experiment, path)

        valid, reason = validate_training_artifacts(experiment)
        if not valid:
            raise ValueError(
                f"alpha={format_value(experiment.alpha)}, "
                f"lambda_ctrl={format_value(experiment.lambda_ctrl)}: {reason}"
            )

        metrics = json.loads(experiment.metrics_path.read_text(encoding="utf-8"))
        best_epoch = int(metrics["best_epoch"])
        try:
            validation = next(
                record
                for record in metrics["history"]
                if int(record["epoch"]) == best_epoch
            )
        except StopIteration as exc:
            raise ValueError(
                f"best epoch {best_epoch} is absent from {experiment.metrics_path}"
            ) from exc

        base = {
            "env": experiment.env,
            "alpha": experiment.alpha,
            "lambda_ctrl": experiment.lambda_ctrl,
            "seed": experiment.seed,
            "condition": experiment.condition,
            "best_epoch": best_epoch,
        }
        rows.append(
            {
                **base,
                "split": "val",
                "ctrl_mse": float(validation["val_ctrl_mse"]),
                "pixel_mse": float(validation["val_pixel_mse"]),
            }
        )
        rows.append(
            {
                **base,
                "split": "test",
                "ctrl_mse": float(metrics["test"]["ctrl_mse"]),
                "pixel_mse": float(metrics["test"]["pixel_mse"]),
            }
        )

    return pd.DataFrame(rows)


def collect_rollout_metrics(
    env: str,
    experiments: Sequence[Experiment],
    *,
    real_path: Path | None = None,
) -> pd.DataFrame:
    real_path = real_path or Path(
        f"datasets/{env}/data/dataset_v1/real_trajectories.npz"
    )
    if not real_path.exists():
        raise FileNotFoundError(f"missing real trajectories: {real_path}")

    circular_dims = (0,) if env == "pendulum" else ()
    rows = []
    with np.load(real_path, allow_pickle=False) as real_data:
        for experiment in experiments:
            if not experiment.trajectory_path.exists():
                raise _artifact_missing(experiment, experiment.trajectory_path)
            with np.load(experiment.trajectory_path, allow_pickle=False) as dwm_data:
                for split in ("val", "test"):
                    metrics = compute_l2_metrics(
                        real_data[f"{split}_traj"],
                        dwm_data[f"{split}_traj"],
                        circular_dims=circular_dims,
                    )
                    action_key = f"{split}_actions"
                    action_mse = float("nan")
                    if action_key in real_data and action_key in dwm_data:
                        action_mse = float(
                            np.mean(
                                (real_data[action_key] - dwm_data[action_key]) ** 2
                            )
                        )
                    rows.append(
                        {
                            "env": experiment.env,
                            "alpha": experiment.alpha,
                            "lambda_ctrl": experiment.lambda_ctrl,
                            "seed": experiment.seed,
                            "condition": experiment.condition,
                            "split": split,
                            "action_mse": action_mse,
                            **metrics,
                        }
                    )

    return pd.DataFrame(rows)


def build_combined_metrics(
    env: str,
    *,
    experiments: Sequence[Experiment] | None = None,
    real_path: Path | None = None,
) -> pd.DataFrame:
    selected = list(build_experiments(env) if experiments is None else experiments)
    training = collect_training_metrics(selected)
    rollout = collect_rollout_metrics(env, selected, real_path=real_path)
    keys = ["env", "condition", "alpha", "lambda_ctrl", "seed", "split"]
    combined = training.merge(rollout, on=keys, how="inner", validate="one_to_one")
    if len(combined) != 2 * len(selected):
        raise ValueError(
            f"incomplete combined table: expected {2 * len(selected)} rows, "
            f"got {len(combined)}"
        )
    columns = [
        "env",
        "condition",
        "alpha",
        "lambda_ctrl",
        "seed",
        "split",
        "best_epoch",
        "ctrl_mse",
        "pixel_mse",
        "action_mse",
        "mean_step_l2",
        "final_l2",
        "max_l2_mean",
        "max_l2_p95",
    ]
    return combined.loc[:, columns].sort_values(
        ["alpha", "lambda_ctrl", "split"]
    ).reset_index(drop=True)


def write_summary_tables(
    env: str,
    *,
    experiments: Sequence[Experiment] | None = None,
    real_path: Path | None = None,
) -> dict[str, pd.DataFrame | dict[str, Path]]:
    selected = list(build_experiments(env) if experiments is None else experiments)
    training = collect_training_metrics(selected)
    rollout = collect_rollout_metrics(env, selected, real_path=real_path)
    combined = build_combined_metrics(
        env,
        experiments=selected,
        real_path=real_path,
    )
    output_dir = GRID_ROOT / env / "alpha_lambda_grid"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "training": output_dir / "training_metrics.csv",
        "rollout": output_dir / "rollout_l2.csv",
        "combined": output_dir / "combined_metrics.csv",
    }
    training.to_csv(paths["training"], index=False)
    rollout.to_csv(paths["rollout"], index=False)
    combined.to_csv(paths["combined"], index=False)
    return {
        "training": training,
        "rollout": rollout,
        "combined": combined,
        "paths": paths,
    }


def pivot_metric(
    frame: pd.DataFrame,
    metric: str,
    *,
    split: str = "val",
) -> pd.DataFrame:
    return (
        frame.loc[frame["split"] == split]
        .pivot(index="alpha", columns="lambda_ctrl", values=metric)
        .sort_index()
        .sort_index(axis=1)
    )


def _rooted(repo_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def compare_with_mainline(
    experiment: Experiment,
    *,
    repo_root: Path = Path("."),
    real_path: Path | None = None,
) -> pd.DataFrame:
    repo_root = Path(repo_root)
    if real_path is None:
        real_path = repo_root / (
            f"datasets/{experiment.env}/data/dataset_v1/real_trajectories.npz"
        )
    selected = build_combined_metrics(
        experiment.env,
        experiments=[experiment],
        real_path=real_path,
    ).assign(source="selected")

    train_config = load_config(
        repo_root
        / "config/train_decoder"
        / experiment.env
        / "saliency.json"
    )
    metrics_path = _rooted(repo_root, train_config["output_dir"]) / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing mainline metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best_epoch = int(metrics["best_epoch"])
    try:
        validation = next(
            record
            for record in metrics["history"]
            if int(record["epoch"]) == best_epoch
        )
    except StopIteration as exc:
        raise ValueError(f"best epoch {best_epoch} is absent from {metrics_path}") from exc

    sampling_config = load_config(
        repo_root / "config/sampling" / f"{experiment.env}.json"
    )
    trajectory_path = (
        _rooted(repo_root, sampling_config["output_dir"])
        / "dwm_trajectories_saliency.npz"
    )
    if not trajectory_path.exists():
        raise FileNotFoundError(f"missing mainline trajectory: {trajectory_path}")

    base = {
        "env": experiment.env,
        "condition": "mainline",
        "alpha": float(train_config["weight"]["alpha"]),
        "lambda_ctrl": float(train_config["lambda_ctrl"]),
        "seed": int(train_config["training"]["seed"]),
        "best_epoch": best_epoch,
        "source": "mainline",
    }
    mainline_rows = []
    circular_dims = (0,) if experiment.env == "pendulum" else ()
    with np.load(real_path, allow_pickle=False) as real_data, np.load(
        trajectory_path,
        allow_pickle=False,
    ) as dwm_data:
        for split in ("val", "test"):
            single_frame = (
                {
                    "ctrl_mse": float(validation["val_ctrl_mse"]),
                    "pixel_mse": float(validation["val_pixel_mse"]),
                }
                if split == "val"
                else {
                    "ctrl_mse": float(metrics["test"]["ctrl_mse"]),
                    "pixel_mse": float(metrics["test"]["pixel_mse"]),
                }
            )
            mainline_rows.append(
                {
                    **base,
                    "split": split,
                    **single_frame,
                    **compute_l2_metrics(
                        real_data[f"{split}_traj"],
                        dwm_data[f"{split}_traj"],
                        circular_dims=circular_dims,
                    ),
                }
            )

    mainline = pd.DataFrame(mainline_rows)
    columns = ["source", *(column for column in selected.columns if column != "source")]
    return pd.concat([selected, mainline], ignore_index=True).loc[:, columns]


def promote_mainline(
    experiment: Experiment,
    *,
    force: bool = False,
    repo_root: Path = Path("."),
    trainer=None,
    mainline_runner=None,
) -> dict:
    if not force:
        raise ValueError("Mainline promotion requires force=True")

    trained, training_reason = validate_training_artifacts(experiment)
    rolled, rollout_reason = validate_rollout_artifact(experiment)
    if not trained or not rolled:
        raise RuntimeError(
            "incomplete source: "
            f"training={training_reason}, rollout={rollout_reason}"
        )

    repo_root = Path(repo_root)
    train_json = (
        repo_root
        / "config/train_decoder"
        / experiment.env
        / "saliency.json"
    )
    config = load_config(train_json)
    changed = (
        float(config["weight"]["alpha"]) != experiment.alpha
        or float(config["lambda_ctrl"]) != experiment.lambda_ctrl
    )
    if changed:
        config["weight"]["alpha"] = experiment.alpha
        config["lambda_ctrl"] = experiment.lambda_ctrl
        train_json.write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )

    canonical_config = copy.deepcopy(config)
    canonical_config["training"]["seed"] = experiment.seed
    canonical_config["output_dir"] = (
        Path("dwm_weight/now_weight") / experiment.env / "saliency"
    ).as_posix()
    set_seed(experiment.seed)
    train_fn = trainer or td.train
    train_fn(
        canonical_config,
        resolve_device(canonical_config.get("device", "auto")),
    )
    rollout_fn = mainline_runner or run_mainline_rollouts
    rollout_frame = rollout_fn(
        experiment.env,
        variants=("saliency",),
        skip_existing=False,
    )
    return {"config_changed": changed, "rollout": rollout_frame}
