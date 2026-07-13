from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

from utils import load_config


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

    @property
    def output_dir(self) -> Path:
        return (
            self.weight_root
            / self.env
            / "alpha_lambda_grid"
            / f"alpha_{format_value(self.alpha)}"
            / f"lambda_{format_value(self.lambda_ctrl)}"
            / f"seed_{self.seed}"
        )

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
            "output_dir": self.output_dir.as_posix(),
        }


def build_experiments(
    env: str,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    seed: int = DEFAULT_SEED,
    weight_root: Path = GRID_ROOT,
) -> list[Experiment]:
    if env not in {"cartpole", "pendulum"}:
        raise ValueError(f"Unsupported ablation environment: {env}")
    return [
        Experiment(env, float(alpha), float(lambda_ctrl), int(seed), Path(weight_root))
        for alpha, lambda_ctrl in product(alphas, lambdas)
    ]


def build_train_config(
    experiment: Experiment,
    base_config: Mapping | None = None,
) -> dict:
    source = base_config
    if source is None:
        source = load_config(
            Path("config/train_decoder") / experiment.env / "saliency.json"
        )
    config = copy.deepcopy(dict(source))
    if config.get("weight_mode") != "saliency":
        raise ValueError("Alpha-lambda grid requires weight_mode='saliency'")
    config["weight"]["alpha"] = experiment.alpha
    config["lambda_ctrl"] = experiment.lambda_ctrl
    config["training"]["seed"] = experiment.seed
    config["output_dir"] = experiment.output_dir.as_posix()
    return config
