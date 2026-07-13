# Alpha-Lambda 消融、DWM Rollout 与 L2 Notebook 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 为 CartPole 和 Pendulum 建立固定 `seed=2025` 的完整 `7×7` saliency `alpha × lambda_ctrl` 消融流程，让 `train_decoder.ipynb` 能完成单模型训练、98 组网格训练、98 组 DWM-only rollout、wrap-aware L2 汇总和显式主线晋升。

**架构：** 新建 `ablation.py` 作为可测试的实验编排层，复用 `train_decoder.train` 和 `sampling.generate_dataset`，不复制训练或 rollout 算法。Notebook 只设置实验参数、调用编排函数并展示 pandas 表格；所有网格点使用独立目录和严格的断点续跑/溯源检查。

**技术栈：** Python 3、PyTorch 2.3、NumPy 1.26、pandas 2.3、Jupyter/nbformat、`unittest`。

## 全局约束

- 环境固定为 `cartpole` 和 `pendulum`。
- `alpha = (0.5, 1, 2, 4, 8, 16, 32)`。
- `lambda_ctrl = (0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5)`。
- 所有网格训练固定 `seed=2025`，每个环境恰好 49 个网格点。
- 每个成功训练的网格点都必须运行 DWM-only rollout，不按 seed 或单帧指标筛选。
- CartPole 使用 full-state L2；Pendulum 的 theta 先取圆周最短差，再计算 full-state L2。
- validation split 用于选择超参数；test split 只用于最终报告和 baseline 对比。
- 不运行 StarV，不实现 conformal inflation，不恢复 `transition_dataset.npz`。
- 主 baseline 路径保持 `dwm_weight/now_weight/<env>/saliency/decoder_best_total.pth`；sampling/StarV JSON 不因晋升改变路径。
- 昂贵 notebook cell 默认不自动执行。
- 保留用户未提交的 `notebooks/generate_dataset.ipynb` 修改，任何提交都不得包含该文件。

## 文件结构

- Create: `ablation.py` — 网格定义、配置生成、断点检查、训练/rollout 编排、L2 汇总、CSV/pivot 和晋升。
- Create: `tests/test_ablation.py` — 不依赖 GPU 的编排与指标单元测试。
- Modify: `notebooks/train_decoder.ipynb` — 薄交互入口，不重复算法。
- Modify: `README.md` — 记录完整消融目录、DWM-only 和 notebook 操作顺序。
- Create: `report/2026-07-13.md` — 记录实现、验证和实际实验运行状态。

---

### Task 1: 实验网格、路径与训练配置

**Files:**
- Create: `ablation.py`
- Create: `tests/test_ablation.py`

**Interfaces:**
- Produces: `Experiment`, `DEFAULT_ALPHAS`, `DEFAULT_LAMBDAS`, `format_value`, `build_experiments`, `build_train_config`。
- Consumes: `utils.load_config`，以及 `config/train_decoder/<env>/saliency.json`。

- [ ] **Step 1: 写失败测试，固定 49 点网格、目录格式和配置不变性**

```python
import copy
import unittest
from pathlib import Path

from ablation import (
    DEFAULT_ALPHAS,
    DEFAULT_LAMBDAS,
    Experiment,
    build_experiments,
    build_train_config,
    format_value,
)


class AblationGridTests(unittest.TestCase):
    def test_complete_grid_has_49_unique_experiments(self):
        experiments = build_experiments("pendulum")
        self.assertEqual(len(experiments), 49)
        self.assertEqual(len(set(experiments)), 49)
        self.assertEqual({e.alpha for e in experiments}, set(DEFAULT_ALPHAS))
        self.assertEqual({e.lambda_ctrl for e in experiments}, set(DEFAULT_LAMBDAS))

    def test_paths_use_compact_decimal_names(self):
        exp = Experiment("cartpole", 0.5, 0.001, 2025)
        self.assertEqual(format_value(8.0), "8")
        self.assertEqual(format_value(0.001), "0.001")
        self.assertEqual(
            exp.output_dir,
            Path("dwm_weight/now_weight/cartpole/alpha_lambda_grid")
            / "alpha_0.5" / "lambda_0.001" / "seed_2025",
        )

    def test_build_train_config_does_not_mutate_base_config(self):
        base = {
            "weight_mode": "saliency",
            "weight": {"alpha": 8.0},
            "lambda_ctrl": 0.1,
            "training": {"seed": 7},
            "output_dir": "old",
        }
        original = copy.deepcopy(base)
        exp = Experiment("pendulum", 2.0, 0.05, 2025)
        actual = build_train_config(exp, base_config=base)
        self.assertEqual(base, original)
        self.assertEqual(actual["weight"]["alpha"], 2.0)
        self.assertEqual(actual["lambda_ctrl"], 0.05)
        self.assertEqual(actual["training"]["seed"], 2025)
        self.assertEqual(actual["output_dir"], exp.output_dir.as_posix())
```

- [ ] **Step 2: 运行测试并确认因模块缺失而失败**

Run:

```bash
python -m unittest discover -s tests -p 'test_ablation.py' -v
```

Expected: FAIL，错误包含 `ModuleNotFoundError: No module named 'ablation'`。

- [ ] **Step 3: 实现最小网格和配置接口**

```python
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
            self.weight_root / self.env / "alpha_lambda_grid"
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
        Experiment(env, float(alpha), float(lam), int(seed), Path(weight_root))
        for alpha, lam in product(alphas, lambdas)
    ]


def build_train_config(
    experiment: Experiment,
    base_config: Mapping | None = None,
) -> dict:
    source = base_config
    if source is None:
        source = load_config(Path("config/train_decoder") / experiment.env / "saliency.json")
    config = copy.deepcopy(dict(source))
    if config.get("weight_mode") != "saliency":
        raise ValueError("Alpha-lambda grid requires weight_mode='saliency'")
    config["weight"]["alpha"] = experiment.alpha
    config["lambda_ctrl"] = experiment.lambda_ctrl
    config["training"]["seed"] = experiment.seed
    config["output_dir"] = experiment.output_dir.as_posix()
    return config
```

- [ ] **Step 4: 运行测试并确认通过**

Run: `python -m unittest discover -s tests -p 'test_ablation.py' -v`

Expected: 3 tests PASS。

- [ ] **Step 5: 提交网格与配置构建器**

```bash
git add ablation.py tests/test_ablation.py
git commit -m "feat: add alpha lambda experiment grid"
```

---

### Task 2: 训练断点检查与 49 点编排

**Files:**
- Modify: `ablation.py`
- Modify: `tests/test_ablation.py`

**Interfaces:**
- Consumes: `Experiment`, `build_experiments`, `build_train_config`。
- Produces: `validate_training_artifacts(experiment) -> tuple[bool, str]` 和 `run_training_grid(env, *, experiments=None, skip_existing=True, continue_on_error=True, trainer=None) -> pandas.DataFrame`。

- [ ] **Step 1: 写失败测试，覆盖完整、部分和配置不匹配的训练产物**

```python
import json
import tempfile
from unittest import mock

import pandas as pd

from ablation import run_training_grid, validate_training_artifacts


class TrainingResumeTests(unittest.TestCase):
    def _experiment(self, root):
        return Experiment("pendulum", 2.0, 0.05, 2025, Path(root))

    def test_complete_matching_training_artifacts_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = self._experiment(tmp)
            exp.output_dir.mkdir(parents=True)
            exp.best_checkpoint.touch()
            exp.last_checkpoint.touch()
            exp.metrics_path.write_text(json.dumps({
                "config": build_train_config(exp, base_config={
                    "weight_mode": "saliency",
                    "weight": {"alpha": 8.0},
                    "lambda_ctrl": 0.1,
                    "training": {"seed": 7},
                    "output_dir": "old",
                }),
                "best_checkpoint": "decoder_best_total.pth",
            }))
            self.assertEqual(validate_training_artifacts(exp), (True, "complete"))

    def test_partial_training_artifacts_are_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = self._experiment(tmp)
            exp.output_dir.mkdir(parents=True)
            exp.best_checkpoint.touch()
            valid, reason = validate_training_artifacts(exp)
            self.assertFalse(valid)
            self.assertIn("missing", reason)

    def test_grid_runner_skips_valid_and_reports_failure(self):
        experiments = [
            Experiment("pendulum", 1.0, 0.0, 2025),
            Experiment("pendulum", 2.0, 0.0, 2025),
        ]
        with mock.patch("ablation.validate_training_artifacts") as validate:
            validate.side_effect = [(True, "complete"), (False, "missing"), (False, "missing")]
            trainer = mock.Mock(side_effect=RuntimeError("boom"))
            frame = run_training_grid(
                "pendulum", experiments=experiments, trainer=trainer,
                skip_existing=True, continue_on_error=True,
            )
        self.assertEqual(frame["status"].tolist(), ["skipped", "failed"])
        self.assertEqual(trainer.call_count, 1)
        self.assertIn("boom", frame.loc[1, "error"])
```

- [ ] **Step 2: 运行测试并确认新接口缺失**

Run: `python -m unittest discover -s tests -p 'test_ablation.py' -v`

Expected: FAIL，错误包含无法导入 `validate_training_artifacts` 或 `run_training_grid`。

- [ ] **Step 3: 实现严格匹配检查与可注入训练器**

实现要点和签名：

```python
import json

import pandas as pd

import train_decoder as td
from utils import resolve_device, set_seed


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
    selected = list(experiments or build_experiments(env))
    train_fn = trainer or td.train
    rows = []
    for experiment in selected:
        valid, reason = validate_training_artifacts(experiment)
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
            rows.append({**experiment.as_row(), "status": "failed", "error": str(exc)})
            if not continue_on_error:
                raise
    return pd.DataFrame(rows)
```

同时给 `Experiment` 增加：

```python
def as_row(self) -> dict:
    return {
        "env": self.env,
        "alpha": self.alpha,
        "lambda_ctrl": self.lambda_ctrl,
        "seed": self.seed,
        "output_dir": self.output_dir.as_posix(),
    }
```

- [ ] **Step 4: 运行训练编排测试和原有测试**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected: 所有测试 PASS；测试不加载 GPU 模型。

- [ ] **Step 5: 提交训练编排**

```bash
git add ablation.py tests/test_ablation.py
git commit -m "feat: add resumable ablation training"
```

---

### Task 3: DWM-only rollout 编排与溯源验证

**Files:**
- Modify: `ablation.py`
- Modify: `tests/test_ablation.py`

**Interfaces:**
- Consumes: `Experiment.best_checkpoint`、`sampling.generate_dataset`、StarV state NPZ。
- Produces: `build_sampling_config`, `validate_rollout_artifact`, `run_rollout_grid`, `run_mainline_rollouts`。

- [ ] **Step 1: 写失败测试，验证 checkpoint、variant 和初始 state**

```python
import numpy as np

from ablation import (
    build_sampling_config,
    run_rollout_grid,
    validate_rollout_artifact,
)


class RolloutResumeTests(unittest.TestCase):
    def _write_rollout_fixture(self, tmp, decoder_weights):
        exp = Experiment("pendulum", 8.0, 0.1, 2025, Path(tmp))
        exp.output_dir.mkdir(parents=True)
        states_path = Path(tmp) / "starv_states.npz"
        states = np.array([[1.0, 4.5]], dtype=np.float32)
        np.savez_compressed(
            states_path,
            train_states=states,
            val_states=states,
            test_states=states,
        )
        np.savez_compressed(
            exp.trajectory_path,
            train_traj=states[:, None],
            train_actions=np.zeros((1, 0, 1)),
            val_traj=states[:, None],
            val_actions=np.zeros((1, 0, 1)),
            test_traj=states[:, None],
            test_actions=np.zeros((1, 0, 1)),
            variant=np.array("saliency"),
            decoder_weights=np.array(str(decoder_weights)),
        )
        return exp, states_path

    def test_sampling_config_uses_direct_grid_checkpoint_and_output_dir(self):
        exp = Experiment("cartpole", 4.0, 0.1, 2025)
        base = {
            "decoder": {"name": "Decoder", "variant": "old", "weights": {"old": "x"}},
            "output_dir": "datasets/cartpole/data/dataset_v1",
            "starv_config": "config/starv_verification/cartpole.json",
        }
        actual = build_sampling_config(exp, base_config=base)
        self.assertEqual(actual["decoder"]["variant"], "saliency")
        self.assertEqual(actual["decoder"]["weights"], exp.best_checkpoint.as_posix())
        self.assertEqual(actual["output_dir"], exp.output_dir.as_posix())

    def test_rollout_validation_rejects_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp, states_path = self._write_rollout_fixture(tmp, "wrong.pth")
            valid, reason = validate_rollout_artifact(exp, states_path=states_path)
            self.assertFalse(valid)
            self.assertIn("checkpoint", reason)

    def test_rollout_validation_accepts_matching_initial_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Experiment("pendulum", 8.0, 0.1, 2025, Path(tmp))
            exp, states_path = self._write_rollout_fixture(tmp, exp.best_checkpoint)
            self.assertEqual(
                validate_rollout_artifact(exp, states_path=states_path),
                (True, "complete"),
            )
```

- [ ] **Step 2: 运行测试并确认新接口缺失**

Run: `python -m unittest discover -s tests -p 'test_ablation.py' -v`

Expected: FAIL，无法导入 rollout 接口。

- [ ] **Step 3: 实现 rollout config、统一验证器和网格运行器**

核心签名：

```python
import numpy as np

import sampling


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


def validate_rollout_artifact(
    experiment: Experiment,
    *,
    states_path: Path | None = None,
) -> tuple[bool, str]:
    if not experiment.trajectory_path.exists():
        return False, "missing trajectory"
    if states_path is None:
        sampling_config = load_config(Path("config/sampling") / f"{experiment.env}.json")
        starv_config = load_config(sampling_config["starv_config"])
        states_path = Path(starv_config["starv_states"]["output_file"])
    try:
        with np.load(states_path, allow_pickle=False) as states, np.load(
            experiment.trajectory_path, allow_pickle=False
        ) as trajectory:
            if trajectory["variant"].item() != "saliency":
                return False, "variant mismatch"
            if not _same_path(trajectory["decoder_weights"].item(), experiment.best_checkpoint):
                return False, "checkpoint mismatch"
            for split in ("train", "val", "test"):
                expected = states[f"{split}_states"]
                actual = trajectory[f"{split}_traj"][:, 0, :]
                if not np.array_equal(expected, actual):
                    return False, f"{split} initial state mismatch"
    except (OSError, ValueError, KeyError) as exc:
        return False, f"invalid trajectory: {exc}"
    return True, "complete"
```

`run_rollout_grid` 与训练运行器采用相同的 `skip_existing`、`continue_on_error` 和依赖注入模式：

```python
def run_rollout_grid(
    env: str,
    *,
    experiments: Sequence[Experiment] | None = None,
    skip_existing: bool = True,
    continue_on_error: bool = True,
    generator=None,
) -> pd.DataFrame:
    selected = list(experiments or build_experiments(env))
    generate_fn = generator or sampling.generate_dataset
    rows = []
    for experiment in selected:
        trained, reason = validate_training_artifacts(experiment)
        if not trained:
            rows.append({**experiment.as_row(), "status": "failed", "error": reason})
            if not continue_on_error:
                raise RuntimeError(reason)
            continue
        valid, reason = validate_rollout_artifact(experiment)
        if skip_existing and valid:
            rows.append({**experiment.as_row(), "status": "skipped", "error": ""})
            continue
        try:
            generate_fn(build_sampling_config(experiment))
            valid, reason = validate_rollout_artifact(experiment)
            if not valid:
                raise RuntimeError(f"rollout artifact validation failed: {reason}")
            rows.append({**experiment.as_row(), "status": "generated", "error": ""})
        except Exception as exc:
            rows.append({**experiment.as_row(), "status": "failed", "error": str(exc)})
            if not continue_on_error:
                raise
    return pd.DataFrame(rows)
```

`run_mainline_rollouts(env: str, *, variants: Sequence[str] = ("intensity", "saliency"), skip_existing: bool = True, generator=None) -> pd.DataFrame` 复用同一个内部 NPZ 验证器，但输出仍写入 `datasets/<env>/data/dataset_v1/`。

- [ ] **Step 4: 运行所有单元测试**

Run: `MPLCONFIGDIR=/tmp/matplotlib python -m unittest discover -s tests -v`

Expected: 全部 PASS；mock generator 证明网格编排没有调用 `render_images` 或 `rollout_transition`。

- [ ] **Step 5: 提交 DWM-only rollout 编排**

```bash
git add ablation.py tests/test_ablation.py
git commit -m "feat: add ablation rollout orchestration"
```

---

### Task 4: Wrap-aware L2、training/rollout 表和 CSV

**Files:**
- Modify: `ablation.py`
- Modify: `tests/test_ablation.py`

**Interfaces:**
- Produces: `compute_l2_metrics`, `collect_training_metrics`, `collect_rollout_metrics`, `build_combined_metrics`, `write_summary_tables`, `pivot_metric`。
- Consumes: 每个网格点的 `metrics.json`、`dwm_trajectories_saliency.npz` 和共享 `real_trajectories.npz`。

- [ ] **Step 1: 写失败测试，锁定普通 L2、圆周差和表连接**

```python
from ablation import build_combined_metrics, compute_l2_metrics, pivot_metric


class L2MetricTests(unittest.TestCase):
    def test_cartpole_full_state_l2(self):
        real = np.zeros((1, 3, 4), dtype=float)
        dwm = real.copy()
        dwm[0, 1, :] = np.array([1.0, 2.0, 2.0, 0.0])
        actual = compute_l2_metrics(real, dwm)
        self.assertAlmostEqual(actual["mean_step_l2"], 1.0)
        self.assertAlmostEqual(actual["max_l2_mean"], 3.0)
        self.assertAlmostEqual(actual["max_l2_p95"], 3.0)

    def test_pendulum_theta_uses_short_circular_difference(self):
        real = np.array([[[np.pi - 0.001, 0.0]]])
        dwm = np.array([[[-np.pi + 0.001, 0.0]]])
        actual = compute_l2_metrics(real, dwm, circular_dims=(0,))
        self.assertAlmostEqual(actual["max_l2_mean"], 0.002, places=6)

    def test_l2_rejects_different_initial_states(self):
        real = np.zeros((1, 2, 2))
        dwm = real.copy()
        dwm[0, 0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "initial states"):
            compute_l2_metrics(real, dwm)

    def test_combined_table_joins_training_and_rollout_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp = Experiment("pendulum", 2.0, 0.05, 2025, root / "weights")
            exp.output_dir.mkdir(parents=True)
            metrics = {
                "config": {
                    "weight_mode": "saliency",
                    "weight": {"alpha": 2.0},
                    "lambda_ctrl": 0.05,
                    "training": {"seed": 2025},
                    "output_dir": exp.output_dir.as_posix(),
                },
                "best_epoch": 1,
                "best_checkpoint": "decoder_best_total.pth",
                "history": [
                    {"epoch": 0, "val_ctrl_mse": 0.3, "val_pixel_mse": 0.4},
                    {"epoch": 1, "val_ctrl_mse": 0.1, "val_pixel_mse": 0.2},
                ],
                "test": {"ctrl_mse": 0.11, "pixel_mse": 0.21},
            }
            exp.metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            exp.best_checkpoint.touch()
            exp.last_checkpoint.touch()

            initial = np.array([[1.0, 4.5]], dtype=np.float32)
            real_val = np.stack([initial, initial + np.array([[0.1, 0.2]])], axis=1)
            real_test = np.stack([initial, initial + np.array([[0.2, 0.3]])], axis=1)
            real_path = root / "real_trajectories.npz"
            np.savez_compressed(real_path, val_traj=real_val, test_traj=real_test)
            np.savez_compressed(
                exp.trajectory_path,
                val_traj=real_val.copy(),
                test_traj=real_test.copy(),
                variant=np.array("saliency"),
                decoder_weights=np.array(exp.best_checkpoint.as_posix()),
            )

            combined = build_combined_metrics(
                "pendulum",
                experiments=[exp],
                real_path=real_path,
            )
            self.assertEqual(len(combined), 2)
            self.assertEqual(set(combined["split"]), {"val", "test"})
            self.assertEqual(set(combined.columns), {
                "env", "alpha", "lambda_ctrl", "seed", "split", "best_epoch",
                "ctrl_mse", "pixel_mse", "mean_step_l2", "final_l2",
                "max_l2_mean", "max_l2_p95",
            })
```

- [ ] **Step 2: 运行测试并确认指标接口缺失**

Run: `python -m unittest discover -s tests -p 'test_ablation.py' -v`

Expected: FAIL，无法导入 `compute_l2_metrics`。

- [ ] **Step 3: 实现 L2 和汇总接口**

```python
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
        raise ValueError(f"trajectory shape mismatch: real={real.shape}, dwm={dwm.shape}")
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
```

`collect_training_metrics` 从 `history[best_epoch]` 读取 validation controller/pixel MSE，从 `test` 读取 test MSE；`collect_rollout_metrics` 为 val/test 各产生一行；`build_combined_metrics` 以 `env, alpha, lambda_ctrl, seed, split` 合并。

接口固定为 `collect_training_metrics(experiments: Sequence[Experiment]) -> pd.DataFrame`、`collect_rollout_metrics(env: str, experiments: Sequence[Experiment], *, real_path: Path | None = None) -> pd.DataFrame` 和 `build_combined_metrics(env: str, *, experiments: Sequence[Experiment] | None = None, real_path: Path | None = None) -> pd.DataFrame`。

以上函数体在实现中必须完整展开：逐个读取 JSON/NPZ，缺失任一网格点时抛出包含 `alpha`、`lambda_ctrl` 和缺失路径的 `FileNotFoundError`，不能返回含 NaN 的“部分成功”表。

`write_summary_tables(env)` 写入：

```text
dwm_weight/now_weight/<env>/alpha_lambda_grid/training_metrics.csv
dwm_weight/now_weight/<env>/alpha_lambda_grid/rollout_l2.csv
dwm_weight/now_weight/<env>/alpha_lambda_grid/combined_metrics.csv
```

`pivot_metric(frame, metric, split="val")` 使用：

```python
return (
    frame.loc[frame["split"] == split]
    .pivot(index="alpha", columns="lambda_ctrl", values=metric)
    .sort_index().sort_index(axis=1)
)
```

- [ ] **Step 4: 运行测试并检查 CSV round trip**

Run: `python -m unittest discover -s tests -v`

Expected: 所有测试 PASS；临时 CSV 重新读取后的行数和列名完全一致。

- [ ] **Step 5: 提交 L2 和汇总表**

```bash
git add ablation.py tests/test_ablation.py
git commit -m "feat: add wrapped rollout L2 summaries"
```

---

### Task 5: 显式主线晋升和固定 JSON 路径

**Files:**
- Modify: `ablation.py`
- Modify: `tests/test_ablation.py`

**Interfaces:**
- Produces: `compare_with_mainline` 和 `promote_mainline`。
- Consumes: 已完成训练和 rollout 的 `Experiment`，以及 `config/train_decoder/<env>/saliency.json`。

- [ ] **Step 1: 写失败测试，保证无 force 不覆盖、路径不变、仅必要时改训练参数**

```python
from ablation import promote_mainline


class PromotionTests(unittest.TestCase):
    def _write_repo_configs(self, root, alpha, lambda_ctrl):
        train_path = root / "config/train_decoder/pendulum/saliency.json"
        sampling_path = root / "config/sampling/pendulum.json"
        starv_path = root / "config/starv_verification/pendulum.json"
        train_path.parent.mkdir(parents=True)
        sampling_path.parent.mkdir(parents=True)
        starv_path.parent.mkdir(parents=True)
        train_path.write_text(json.dumps({
            "weight_mode": "saliency",
            "weight": {"alpha": alpha},
            "lambda_ctrl": lambda_ctrl,
            "training": {"seed": 2025},
            "device": "cpu",
            "output_dir": "dwm_weight/now_weight/pendulum/saliency",
        }, indent=2) + "\n")
        sampling_path.write_text(json.dumps({
            "decoder": {"weights": {
                "saliency": "dwm_weight/now_weight/pendulum/saliency/decoder_best_total.pth"
            }}
        }, indent=2) + "\n")
        starv_path.write_text(json.dumps({
            "layers": {"Decoder": {"kwargs": {
                "weights": "dwm_weight/now_weight/pendulum/saliency/decoder_best_total.pth"
            }}}
        }, indent=2) + "\n")
        return train_path, sampling_path, starv_path

    def test_promotion_requires_force(self):
        exp = Experiment("pendulum", 4.0, 0.05, 2025)
        with self.assertRaisesRegex(ValueError, "force=True"):
            promote_mainline(exp, force=False)

    def test_same_hyperparameters_do_not_rewrite_training_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path, _, _ = self._write_repo_configs(root, 8.0, 0.1)
            before = train_path.read_bytes()
            exp = Experiment("pendulum", 8.0, 0.1, 2025, root / "weights")
            trainer = mock.Mock()
            mainline_runner = mock.Mock(return_value=pd.DataFrame([{"status": "generated"}]))
            with mock.patch("ablation.validate_training_artifacts", return_value=(True, "complete")), \
                 mock.patch("ablation.validate_rollout_artifact", return_value=(True, "complete")):
                result = promote_mainline(
                    exp,
                    force=True,
                    repo_root=root,
                    trainer=trainer,
                    mainline_runner=mainline_runner,
                )
            self.assertFalse(result["config_changed"])
            self.assertEqual(train_path.read_bytes(), before)
            trainer.assert_called_once()
            mainline_runner.assert_called_once()

    def test_changed_hyperparameters_update_only_train_json_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path, sampling_path, starv_path = self._write_repo_configs(root, 8.0, 0.1)
            sampling_before = sampling_path.read_bytes()
            starv_before = starv_path.read_bytes()
            exp = Experiment("pendulum", 4.0, 0.05, 2025, root / "weights")
            with mock.patch("ablation.validate_training_artifacts", return_value=(True, "complete")), \
                 mock.patch("ablation.validate_rollout_artifact", return_value=(True, "complete")):
                result = promote_mainline(
                    exp,
                    force=True,
                    repo_root=root,
                    trainer=mock.Mock(),
                    mainline_runner=mock.Mock(return_value=pd.DataFrame()),
                )
            updated = json.loads(train_path.read_text())
            self.assertTrue(result["config_changed"])
            self.assertEqual(updated["weight"]["alpha"], 4.0)
            self.assertEqual(updated["lambda_ctrl"], 0.05)
            self.assertEqual(
                updated["output_dir"],
                "dwm_weight/now_weight/pendulum/saliency",
            )
            self.assertEqual(sampling_path.read_bytes(), sampling_before)
            self.assertEqual(starv_path.read_bytes(), starv_before)
```

- [ ] **Step 2: 运行测试并确认晋升接口缺失**

Run: `python -m unittest discover -s tests -p 'test_ablation.py' -v`

Expected: FAIL，无法导入 `promote_mainline`。

- [ ] **Step 3: 实现显式晋升**

实现规则：

```python
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
    trained, reason = validate_training_artifacts(experiment)
    rolled, rollout_reason = validate_rollout_artifact(experiment)
    if not trained or not rolled:
        raise RuntimeError(f"incomplete source: training={reason}, rollout={rollout_reason}")

    train_json = repo_root / "config/train_decoder" / experiment.env / "saliency.json"
    config = load_config(train_json)
    changed = (
        float(config["weight"]["alpha"]) != experiment.alpha
        or float(config["lambda_ctrl"]) != experiment.lambda_ctrl
    )
    if changed:
        config["weight"]["alpha"] = experiment.alpha
        config["lambda_ctrl"] = experiment.lambda_ctrl
        train_json.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    canonical_config = copy.deepcopy(config)
    canonical_config["output_dir"] = (
        Path("dwm_weight/now_weight") / experiment.env / "saliency"
    ).as_posix()
    set_seed(experiment.seed)
    train_fn = trainer or td.train
    train_fn(canonical_config, resolve_device(canonical_config.get("device", "auto")))
    rollout_fn = mainline_runner or run_mainline_rollouts
    rollout_frame = rollout_fn(
        experiment.env,
        variants=("saliency",),
        skip_existing=False,
    )
    return {"config_changed": changed, "rollout": rollout_frame}
```

`compare_with_mainline(experiment)` 返回 selected/mainline 的 validation/test 单帧和 L2 并排 DataFrame。Notebook 必须先显示该表，再展示注释状态的 `force=True` 调用。

- [ ] **Step 4: 运行晋升测试和完整测试集**

Run: `MPLCONFIGDIR=/tmp/matplotlib python -m unittest discover -s tests -v`

Expected: 全部 PASS；测试确认 sampling/StarV JSON 未被修改。

- [ ] **Step 5: 提交晋升逻辑**

```bash
git add ablation.py tests/test_ablation.py
git commit -m "feat: add explicit saliency promotion"
```

---

### Task 6: 重构 `train_decoder.ipynb` 为完整交互入口

**Files:**
- Modify: `notebooks/train_decoder.ipynb`

**Interfaces:**
- Consumes: Task 1–5 的公开 `ablation.py` 接口。
- Produces: 用户可按段运行的单训练、完整网格训练、mainline/grid rollout、CSV/L2/pivot 和晋升 cells。

- [ ] **Step 1: 写 notebook 结构验证测试**

在 `tests/test_ablation.py` 增加：

```python
import nbformat


class TrainNotebookTests(unittest.TestCase):
    def test_notebook_contains_ablation_workflow_cells(self):
        nb = nbformat.read("notebooks/train_decoder.ipynb", as_version=4)
        ids = {cell.get("id") for cell in nb.cells}
        required = {
            "nb-setup", "nb-single-train", "nb-grid-config", "nb-grid-train",
            "nb-training-summary", "nb-mainline-rollout", "nb-grid-rollout",
            "nb-l2-summary", "nb-pivots", "nb-promotion",
        }
        self.assertTrue(required.issubset(ids))
        expensive = {"nb-grid-train", "nb-mainline-rollout", "nb-grid-rollout", "nb-promotion"}
        for cell in nb.cells:
            if cell.get("id") in expensive:
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs", []), [])
```

- [ ] **Step 2: 运行测试并确认缺少新 cell id**

Run: `python -m unittest discover -s tests -p 'test_ablation.py' -v`

Expected: FAIL，提示 required cell IDs 不完整。

- [ ] **Step 3: 用 nbformat 机械重建 notebook cell 顺序**

保留中文说明，清除旧运行输出。核心 cells 内容如下：

```python
# nb-grid-config
from ablation import DEFAULT_ALPHAS, DEFAULT_LAMBDAS, build_experiments

ENVS = ("cartpole", "pendulum")
ALPHAS = DEFAULT_ALPHAS
LAMBDAS = DEFAULT_LAMBDAS
SEED = 2025

for env_name in ENVS:
    experiments = build_experiments(env_name, ALPHAS, LAMBDAS, SEED)
    print(env_name, len(experiments), experiments[0].output_dir)
```

```python
# nb-grid-train（昂贵，默认注释）
import ablation as abl

# pendulum_training = abl.run_training_grid("pendulum")
# display(pendulum_training)
# cartpole_training = abl.run_training_grid("cartpole")
# display(cartpole_training)
```

```python
# nb-mainline-rollout（默认注释）
# for env_name in ENVS:
#     display(abl.run_mainline_rollouts(env_name, variants=("intensity", "saliency")))
```

```python
# nb-grid-rollout（昂贵，默认注释）
# pendulum_rollouts = abl.run_rollout_grid("pendulum")
# display(pendulum_rollouts)
# cartpole_rollouts = abl.run_rollout_grid("cartpole")
# display(cartpole_rollouts)
```

```python
# nb-l2-summary
tables = {}
for env_name in ENVS:
    tables[env_name] = abl.write_summary_tables(env_name)
    display(tables[env_name]["combined"])
```

```python
# nb-pivots
for env_name, result in tables.items():
    print(f"[{env_name}] validation max_l2_p95")
    display(abl.pivot_metric(result["combined"], "max_l2_p95", split="val"))
    print(f"[{env_name}] validation ctrl_mse")
    display(abl.pivot_metric(result["combined"], "ctrl_mse", split="val"))
```

```python
# nb-promotion（默认注释，必须人工填写并 force）
# 示例参数只用于展示调用格式；实际晋升前替换为 validation 表中确认的组合。
# candidate = abl.Experiment("pendulum", alpha=4.0, lambda_ctrl=0.05, seed=2025)
# display(abl.compare_with_mainline(candidate))
# abl.promote_mainline(candidate, force=True)
```

单模型训练 cell 继续调用已有 `train_decoder.train`；Notebook 不复制训练循环或 sampling loop。

- [ ] **Step 4: 验证 notebook 格式、cell ids 和无保存错误**

Run:

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python - <<'PY'
import nbformat
nb = nbformat.read("notebooks/train_decoder.ipynb", as_version=4)
errors = [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
print(f"cells={len(nb.cells)} saved_errors={len(errors)}")
assert not errors
PY
python -m unittest discover -s tests -v
```

Expected: notebook 可解析，`saved_errors=0`，所有测试 PASS。

- [ ] **Step 5: 提交 notebook**

```bash
git add notebooks/train_decoder.ipynb tests/test_ablation.py
git commit -m "feat: add complete ablation notebook workflow"
```

---

### Task 7: README、当日报告与全量静态验证

**Files:**
- Modify: `README.md`
- Create: `report/2026-07-13.md`

**Interfaces:**
- Documents: 新网格目录、DWM-only 性能收益、L2 语义、断点续跑和 notebook 操作顺序。

- [ ] **Step 1: 更新 README 的训练与 rollout 流程**

加入以下事实：

```text
train_decoder.ipynb 支持单模型和完整 alpha-lambda 网格。
网格固定 seed=2025，每个环境 49 个训练和 49 个 DWM-only rollout。
sampling 不生成 transition_dataset.npz，不调用真实 renderer；real trajectory 只生成一次并复用。
Pendulum L2 使用圆周 theta 差。
网格结果位于 dwm_weight/now_weight/<env>/alpha_lambda_grid/。
```

- [ ] **Step 2: 创建中文 7/13 工作记录**

记录：

- 设计与实施文件；
- 98 组训练/rollout 的计划和实际完成数；
- 测试命令及结果；
- 尚未运行的 StarV 明确不属于本轮；
- 若完整实验仍在运行，写准确状态，不写“已完成”。

- [ ] **Step 3: 运行全量验证**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib python -m unittest discover -s tests -v
python -m py_compile ablation.py train_decoder.py sampling.py
/home/tealab_shared/starv/env/starv_shared/bin/python - <<'PY'
from pathlib import Path
import nbformat
for path in sorted(Path("notebooks").glob("*.ipynb")):
    nbformat.read(path, as_version=4)
    print("valid", path)
PY
git diff --check
```

Expected: 所有测试 PASS、py_compile exit 0、4 个 notebook 全部 valid、`git diff --check` 无输出。

- [ ] **Step 4: 确认没有混入用户的 notebook 改动**

Run:

```bash
git status --short
git diff -- notebooks/generate_dataset.ipynb
```

Expected: `generate_dataset.ipynb` 仍是用户的未提交修改；本任务提交列表不包含它。

- [ ] **Step 5: 提交文档**

```bash
git add README.md report/2026-07-13.md
git commit -m "docs: describe alpha lambda experiment workflow"
```

---

### Task 8: 运行主 baseline、完整 98 组训练和 98 组 DWM-only rollout

**Files:**
- Generated/ignored: `dwm_weight/now_weight/<env>/alpha_lambda_grid/**`
- Generated/ignored: `datasets/<env>/data/dataset_v1/dwm_trajectories_{intensity,saliency}.npz`

**Interfaces:**
- Consumes: Tasks 1–7 的稳定 API。
- Produces: 两个环境完整的 `combined_metrics.csv` 和 notebook L2/pivot 表。

- [ ] **Step 1: 运行主 baseline DWM-only rollout**

Run in `starv_shared`：

```python
import ablation as abl
for env_name in ("cartpole", "pendulum"):
    print(abl.run_mainline_rollouts(env_name, variants=("intensity", "saliency")))
```

Expected: 每个环境得到 intensity/saliency 两个 NPZ；内部 `variant`、`decoder_weights` 与初始 state 验证通过。

- [ ] **Step 2: 先运行 Pendulum 49 点训练，再运行 CartPole 49 点训练**

```python
for env_name in ("pendulum", "cartpole"):
    frame = abl.run_training_grid(env_name, skip_existing=True, continue_on_error=True)
    print(frame["status"].value_counts())
    failures = frame.loc[frame["status"] == "failed"]
    if not failures.empty:
        raise RuntimeError(f"{env_name} training failures:\n{failures}")
```

Expected: 每个环境 49 行，无 `failed`；中断后重跑会将完整点显示为 `skipped`。

- [ ] **Step 3: 运行两个环境全部 98 个 DWM-only rollout**

```python
for env_name in ("pendulum", "cartpole"):
    frame = abl.run_rollout_grid(env_name, skip_existing=True, continue_on_error=True)
    print(frame["status"].value_counts())
    failures = frame.loc[frame["status"] == "failed"]
    if not failures.empty:
        raise RuntimeError(f"{env_name} rollout failures:\n{failures}")
```

Expected: 每个环境 49 行，无 `failed`；sampling 日志中没有真实 renderer 或 transition dataset 输出。

- [ ] **Step 4: 生成并核对完整 L2 表**

```python
for env_name in ("pendulum", "cartpole"):
    result = abl.write_summary_tables(env_name)
    combined = result["combined"]
    assert len(combined) == 98  # 49 个网格点 × val/test
    assert combined[["ctrl_mse", "pixel_mse", "max_l2_p95"]].notna().all().all()
    print(env_name)
    print(combined.loc[combined["split"] == "val"].nsmallest(10, "max_l2_p95"))
```

Expected: 每个环境 `combined_metrics.csv` 恰好 98 行，无缺失指标；Pendulum 不出现约 `2π` 的 seam 伪峰。

- [ ] **Step 5: 向用户报告候选，不自动晋升**

报告每个环境 validation `max_l2_p95` 前若干组合，并同时列出：

- `max_l2_mean`；
- validation controller MSE；
- validation pixel MSE；
- 当前 `(alpha=8, lambda=0.1)` 行；
- 对应 test 指标仅作为最终候选报告。

根据设计，先让用户确认最终组合，再运行 notebook 中的 `promote_mainline(candidate, force=True)`。晋升后重新生成 mainline saliency rollout 和最终 baseline 表。

- [ ] **Step 6: 用真实完成数和候选表更新 7/13 报告**

把 `report/2026-07-13.md` 中的计划状态替换为实际证据：

- CartPole/Pendulum 各自成功训练数，目标均为 49；
- CartPole/Pendulum 各自成功 rollout 数，目标均为 49；
- 两个 `combined_metrics.csv` 的行数，目标均为 98；
- validation `max_l2_p95` 最优组合和当前 `(8, 0.1)` 对照；
- test 指标只记录已确认候选，不据此反向调参；
- StarV 仍未运行。

Run:

```bash
git add report/2026-07-13.md
git commit -m "docs: record alpha lambda experiment results"
```

- [ ] **Step 7: 最终证据检查**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib python -m unittest discover -s tests -v
git status --short --branch
```

Expected: 所有测试 PASS；只有用户原有 `notebooks/generate_dataset.ipynb` 修改和被 `.gitignore` 排除的实验产物，不存在意外源码改动。
