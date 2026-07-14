# Rollout 参数选择摘要实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 为 CartPole 和 Pendulum 的完整 rollout 网格生成紧凑的全局、按 alpha、按 lambda 参数选择摘要，并在训练 notebook 中作为默认阅读视图展示。

**架构：** `ablation.py` 从已有的 `combined_metrics.csv`/DataFrame 提取 validation `max_l2_p95` 最优组合，再附带同一组合的 test 行；选择逻辑为纯 DataFrame 函数，文件写入与 notebook 仅作为薄包装。原始 98 行明细和现有 pivot 保留为审计/诊断入口。

**技术栈：** Python 3、pandas 2.3、unittest、nbformat。

## 全局约束

- 参数选择只能使用 validation `max_l2_p95`，值越小越好。
- test split 不参与选择；它只跟随被 validation 选中的 `(alpha, lambda_ctrl)` 展示。
- 平局按 `alpha`、`lambda_ctrl` 升序稳定打破。
- 每个完整 7×7 环境摘要有 30 行：`global` 2 行、`by_alpha` 14 行、`by_lambda` 14 行。
- 不重训 decoder、不重新生成 rollout、不改 canonical 配置或 StarV 文件。
- 用户的 `notebooks/generate_dataset.ipynb` 未提交修改不得触碰或提交。

---

### Task 1: 纯摘要选择与 CSV 写入

**Files:**
- Modify: `ablation.py`
- Modify: `tests/test_ablation.py`

**Interfaces:**
- Consumes: 含有 `env`、`alpha`、`lambda_ctrl`、`seed`、`split`、`max_l2_p95` 的 combined DataFrame。
- Produces: `build_selection_summary(frame: pd.DataFrame) -> pd.DataFrame` 和 `write_selection_summary(env: str) -> pd.DataFrame`。

- [ ] **Step 1: 写失败测试，锁定 validation-only、双 split 回填和每轴一组获胜者**

在 `tests/test_ablation.py` 的 import 中加入：

```python
from ablation import build_selection_summary, write_selection_summary
```

加入测试夹具与断言：

```python
class SelectionSummaryTests(unittest.TestCase):
    def _frame(self):
        rows = []
        values = {
            (1.0, 0.0): (0.30, 0.01),
            (1.0, 0.1): (0.20, 0.99),
            (2.0, 0.0): (0.20, 0.00),
            (2.0, 0.1): (0.40, 0.02),
        }
        for (alpha, lambda_ctrl), (val_p95, test_p95) in values.items():
            for split, p95 in (("val", val_p95), ("test", test_p95)):
                rows.append({
                    "env": "cartpole", "alpha": alpha,
                    "lambda_ctrl": lambda_ctrl, "seed": 2025,
                    "split": split, "best_epoch": 1,
                    "ctrl_mse": 0.1, "pixel_mse": 0.2,
                    "mean_step_l2": p95 / 2, "final_l2": p95 / 2,
                    "max_l2_mean": p95 / 2, "max_l2_p95": p95,
                })
        return pd.DataFrame(rows)

    def test_summary_uses_validation_only_and_preserves_matching_test_rows(self):
        summary = build_selection_summary(self._frame())
        global_rows = summary.loc[summary["selection_scope"] == "global"]
        self.assertEqual(len(global_rows), 2)
        self.assertEqual(set(global_rows["split"]), {"val", "test"})
        self.assertEqual(set(global_rows["alpha"]), {1.0})
        self.assertEqual(set(global_rows["lambda_ctrl"]), {0.1})
        self.assertEqual(
            global_rows.loc[global_rows["split"] == "test", "max_l2_p95"].item(),
            0.99,
        )

    def test_summary_selects_one_winner_for_each_alpha_and_lambda(self):
        summary = build_selection_summary(self._frame())
        by_alpha = summary.loc[summary["selection_scope"] == "by_alpha"]
        by_lambda = summary.loc[summary["selection_scope"] == "by_lambda"]
        self.assertEqual(len(by_alpha), 4)
        self.assertEqual(len(by_lambda), 4)
        self.assertEqual(
            by_alpha.loc[by_alpha["split"] == "val"]
            .sort_values("alpha")["lambda_ctrl"].tolist(),
            [0.1, 0.0],
        )
        self.assertEqual(
            by_lambda.loc[by_lambda["split"] == "val"]
            .sort_values("lambda_ctrl")["alpha"].tolist(),
            [2.0, 1.0],
        )
```

- [ ] **Step 2: 运行测试并确认接口尚不存在**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib python -m unittest discover -s tests -p 'test_ablation.py' -v
```

Expected: FAIL，错误包含无法导入 `build_selection_summary`。

- [ ] **Step 3: 实现 validation 选择和同参 test 回填**

在 `ablation.py` 的 `pivot_metric` 后加入：

```python
SUMMARY_KEYS = ["env", "alpha", "lambda_ctrl", "seed"]


def _summary_rows(
    frame: pd.DataFrame,
    winners: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    selected = frame.merge(winners[SUMMARY_KEYS], on=SUMMARY_KEYS, how="inner")
    if len(selected) != 2 * len(winners):
        raise ValueError(f"{scope}: selected parameter is missing val/test rows")
    return selected.assign(selection_scope=scope)


def build_selection_summary(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(SUMMARY_KEYS + ["split", "max_l2_p95"])
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"summary frame is missing columns: {sorted(missing)}")
    validation = frame.loc[frame["split"] == "val"].copy()
    if validation.empty:
        raise ValueError("summary requires validation rows")
    ordered = validation.sort_values(
        ["max_l2_p95", "alpha", "lambda_ctrl"],
        kind="stable",
    )
    global_winner = ordered.head(1)
    alpha_winners = ordered.groupby("alpha", sort=True, as_index=False).head(1)
    lambda_winners = ordered.groupby("lambda_ctrl", sort=True, as_index=False).head(1)
    result = pd.concat(
        [
            _summary_rows(frame, global_winner, "global"),
            _summary_rows(frame, alpha_winners, "by_alpha"),
            _summary_rows(frame, lambda_winners, "by_lambda"),
        ],
        ignore_index=True,
    )
    columns = ["selection_scope", *frame.columns]
    return result.loc[:, columns].sort_values(
        ["selection_scope", "alpha", "lambda_ctrl", "split"],
        kind="stable",
    ).reset_index(drop=True)


def write_selection_summary(env: str) -> pd.DataFrame:
    output_dir = GRID_ROOT / env / "alpha_lambda_grid"
    combined_path = output_dir / "combined_metrics.csv"
    if combined_path.exists():
        frame = pd.read_csv(combined_path)
    else:
        frame = build_combined_metrics(env)
    summary = build_selection_summary(frame)
    summary.to_csv(output_dir / "selection_summary.csv", index=False)
    return summary
```

保留 `_summary_rows` 的显式双 split 行数检查，避免只存在 validation 或 test 时输出看似完整的摘要。

- [ ] **Step 4: 运行摘要测试和全量测试**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib python -m unittest discover -s tests -v
```

Expected: 所有测试通过；新的测试证明全局胜者由 validation `0.20` 的平局按 alpha 选 `(1.0, 0.1)`，即使该行 test 值为 `0.99`。

- [ ] **Step 5: 提交核心摘要接口**

```bash
git add ablation.py tests/test_ablation.py
git commit -m "feat: add rollout selection summaries"
```

### Task 2: Notebook 默认展示紧凑摘要

**Files:**
- Modify: `notebooks/train_decoder.ipynb`
- Modify: `tests/test_ablation.py`

**Interfaces:**
- Consumes: `abl.write_summary_tables(env)`、`abl.write_selection_summary(env)`。
- Produces: notebook `nb-l2-summary` 中的三张默认摘要表；原始 combined/pivot 仍可按需读取。

- [ ] **Step 1: 扩展 notebook 结构测试，锁定摘要调用**

在 `TrainNotebookTests` 加入：

```python
    def test_notebook_l2_summary_writes_selection_summary(self):
        notebook = nbformat.read("notebooks/train_decoder.ipynb", as_version=4)
        cell = next(cell for cell in notebook.cells if cell.get("id") == "nb-l2-summary")
        self.assertIn("write_selection_summary", cell["source"])
        self.assertIn("selection_scope", cell["source"])
```

- [ ] **Step 2: 运行 notebook 结构测试并确认失败**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib python -m unittest discover -s tests -p 'test_ablation.py' -v
```

Expected: FAIL，`nb-l2-summary` 尚未包含 `write_selection_summary`。

- [ ] **Step 3: 机械更新 `nb-l2-summary` cell**

使用 nbformat 更新该 cell 为：

```python
tables = {}
selection_summaries = {}
for env in ENVS:
    try:
        tables[env] = abl.write_summary_tables(env)
        selection_summaries[env] = abl.write_selection_summary(env)
        print(f"[{env}] 参数选择摘要（仅 validation 选参；test 只跟随展示）")
        for scope, label in (
            ("global", "全局最优"),
            ("by_alpha", "每个 alpha 的最优 lambda"),
            ("by_lambda", "每个 lambda 的最优 alpha"),
        ):
            print(label)
            display(
                selection_summaries[env]
                .loc[lambda frame: frame["selection_scope"] == scope]
            )
    except FileNotFoundError as exc:
        print(f"[{env}] 训练或 rollout 尚未完整：{exc}")
```

保留 `nb-pivots`，但其定位是诊断而非默认选参入口。更新后 cell 的 `execution_count` 必须为 `None`，`outputs` 必须为空。

- [ ] **Step 4: 生成现有两环境的摘要 CSV 并验证行数**

Run:

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python - <<'PY'
import ablation as abl
for env in ("pendulum", "cartpole"):
    summary = abl.write_selection_summary(env)
    assert len(summary) == 30
    assert set(summary["selection_scope"]) == {"global", "by_alpha", "by_lambda"}
    print(env, len(summary))
PY
```

Expected: 输出 `pendulum 30` 与 `cartpole 30`。`selection_summary.csv` 由现有网格产物 ignore 规则忽略，不进入提交。

- [ ] **Step 5: 完整验证并提交 notebook**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib python -m unittest discover -s tests -v
python -m py_compile ablation.py
/home/tealab_shared/starv/env/starv_shared/bin/python - <<'PY'
import nbformat
nb = nbformat.read("notebooks/train_decoder.ipynb", as_version=4)
errors = [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
assert not errors
print("notebook valid")
PY
git diff --check
```

Expected: 全部命令退出码为 0，notebook 未保存错误输出。

Commit:

```bash
git add notebooks/train_decoder.ipynb tests/test_ablation.py
git commit -m "feat: show compact rollout selections"
```

## 自检

- 规格中的全局、按 alpha、按 lambda 三类输出分别由 Task 1 的三组 winners 实现，并由 Task 2 展示。
- validation-only 与 test 跟随规则由 Task 1 的夹具验证；夹具故意让 validation 获胜行的 test 值很差，防止错误使用 test 选择。
- 30 行数、CSV 生成与 notebook 源码引用由 Task 2 验证。
- 没有涉及训练、rollout、canonical JSON、StarV 或用户修改中的 `generate_dataset.ipynb`。
