# Rollout 参数选择摘要设计

## 目标

将当前每个环境 98 行的 `combined_metrics.csv` 与 `rollout_l2.csv` 转换为适合日常阅读的紧凑选择摘要。原始 CSV 继续保留为可追溯证据；notebook 默认展示摘要，不要求用户手工从完整网格中寻找结论。

## 选择规则

所有参数选择只使用 validation split 的 `max_l2_p95`，值越小越好。

- 不在 validation/test 两行之间取较小值。
- test split 不参与参数选择；仅展示被 validation 选中的同一 `(alpha, lambda_ctrl)` 的 test 指标。
- 若 validation `max_l2_p95` 并列，按 `alpha` 升序、再按 `lambda_ctrl` 升序稳定打破平局。

这保证 test 不会泄漏到网格调参过程。

## 摘要内容

每个环境生成三张表：

1. **全局最优**：在 49 个 validation 组合中取一组最优 `(alpha, lambda_ctrl)`，并附带其对应的 test 行。
2. **按 alpha 选择**：每个 alpha 在 7 个 lambda 中选出 validation `max_l2_p95` 最小的 lambda，共 7 组；每组附带对应 test 行。
3. **按 lambda 选择**：每个 lambda 在 7 个 alpha 中选出 validation `max_l2_p95` 最小的 alpha，共 7 组；每组附带对应 test 行。

每组结果以两个 split 的同参行呈现，保留：`alpha`、`lambda_ctrl`、`best_epoch`、`ctrl_mse`、`pixel_mse`、`mean_step_l2`、`final_l2`、`max_l2_mean`、`max_l2_p95`。表额外带有 `selection_scope` 标识：`global`、`by_alpha` 或 `by_lambda`。

因此一个环境的合并摘要共有 30 行：全局 2 行 + 按 alpha 14 行 + 按 lambda 14 行。它足够小，可在 notebook 中直接展示；仍可通过 `selection_scope` 分成三张表。

## 实现边界

在 `ablation.py` 新增纯 DataFrame 接口：

```python
build_selection_summary(frame: pd.DataFrame) -> pd.DataFrame
write_selection_summary(env: str) -> pd.DataFrame
```

`build_selection_summary` 不读写文件，便于单元测试。`write_selection_summary` 读取现有 `combined_metrics.csv`（不存在时从完整网格重建），并写入：

```text
dwm_weight/now_weight/<env>/alpha_lambda_grid/selection_summary.csv
```

`notebooks/train_decoder.ipynb` 的 L2 section 在 `combined_metrics.csv` 写出后调用该接口，按三种 `selection_scope` 展示表格。原有 pivot 保留为可选诊断，不作为默认决策入口。

本功能不重训 decoder、不重新生成 rollout、不修改 canonical 配置，也不修改 StarV 相关文件。

## 测试

新增无 GPU 单元测试，验证：

- 全局选择只基于 validation，即使 test 更好也不能影响胜者；
- 每个 alpha 和 lambda 都恰好产生一个获胜组合；
- 同一获胜组合的 val/test 行同时保留；
- 并列时使用稳定的 alpha/lambda 排序；
- `selection_summary.csv` 可写出并保留预期行数与字段。

完成后运行完整 unittest、notebook nbformat 解析、`py_compile` 和 `git diff --check`。 
