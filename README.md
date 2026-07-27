# Trajectory Predictor

本文档以 `trajectory_predictor(3).zip` 中的实际源码为准，完整说明 predictor 的目标、模型、训练流程、tube 构建流程、输入输出格式、运行方法和当前限制。

## 1. Predictor 的作用

Trajectory Predictor 学习一个从初始状态到整条未来状态轨迹的映射：

```text
初始状态 s0
    ↓
Transformer Predictor
    ↓
[s0, s1, s2, ..., sH]
```

训练完成后，程序会在 verification grid 的每个 cell 中选取若干初始状态，预测这些状态对应的未来轨迹，再对同一 cell 内的预测结果逐时间步、逐状态维度取最小值和最大值，得到该 cell 的 predictor tube。

当前默认设置为：

```text
每个 cell 采样点数：3
预测 transition 数：20
输出状态点数：21，即 s0 到 s20
```

需要注意：当前 tube 是有限采样轨迹的 min/max 包络，不是形式化可达集，也没有理论覆盖保证。

## 2. 项目文件

```text
trajectory_predictor/
├── config.py
├── data_utils.py
├── predictor_model.py
├── train_predictor.py
├── tube_utils.py
├── build_tube.py
├── safety_results/
├── models/
└── predictor_results/
```

各文件的作用如下。

| 文件 | 作用 |
|---|---|
| `config.py` | 设置默认路径、默认 horizon、每个 cell 的采样数、随机种子和计算设备 |
| `data_utils.py` | 加载并检查真实轨迹，截断轨迹，划分训练数据，计算 normalization |
| `predictor_model.py` | 定义 Transformer、loss、checkpoint 加载和批量预测 |
| `train_predictor.py` | 训练 predictor，并保存最佳 checkpoint |
| `tube_utils.py` | 读取 grid、在 cell 中采样、预测轨迹、构建 min/max tube、保存结果 |
| `build_tube.py` | 加载 checkpoint，调用 `tube_utils.py` 生成 NPZ 和 JSON |

## 3. 完整工作流程

```text
real_trajectories.npz
        │
        ▼
train_predictor.py
        │
        ▼
predictor_transformer.pth
        │
        ├──────────────┐
        │              │
safety_result.json     │
        │              │
        └──────┬───────┘
               ▼
         build_tube.py
               │
      ┌────────┴────────┐
      ▼                 ▼
predictor_          predictor_
trajectories.npz    tube.json
```

第一步用真实轨迹训练 predictor。第二步读取 verification grid，在每个 cell 内采样三个初始状态，通过 predictor 生成轨迹并构建 tube。

## 4. 安装依赖

推荐使用 Python 3.9 或更高版本。

```bash
pip install numpy torch
```

<<<<<<< HEAD
如果服务器已经进入项目使用的环境，例如：
=======
MountainCar 命令会生成 `saliency_occlusion_background_median.npz`。Pendulum 使用默认的
white occlusion baseline，只需换成对应配置。

`notebooks/train_decoder.ipynb` 也提供完整的 saliency `alpha × lambda_ctrl` 消融入口：

```text
alpha       = [0.5, 1, 2, 4, 8, 16, 32]
lambda_ctrl = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
seed        = 2025
```

CartPole 和 Pendulum 各有 49 个训练点，不筛选 seed。网格实现位于 `ablation.py`，notebook
只负责调用和显示表格。每个点使用独立目录：

```text
dwm_weight/<env>/alpha_lambda_grid/
  alpha_<alpha>/lambda_<lambda>/seed_2025/
    decoder_best_total.pth
    decoder_last.pth
    metrics.json
    dwm_trajectories_saliency.npz
```

再次执行时，只有 checkpoint、metrics 和配置严格匹配的完整点才会跳过；部分产物、错误参数或
错误 rollout 溯源都会重跑。旧的 CartPole 一维消融目录保留为历史结果。

### 4. 生成 DWM trajectory

在 `notebooks/generate_dataset.ipynb` 中运行
`run_sampling("cartpole", decoder_variant="saliency")`。对应的命令行为：
>>>>>>> origin/main

```bash
conda activate /home/tealab_shared/starv/env/starv_shared
```

可以先检查依赖：

```bash
python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"
```

## 5. 输入一：`real_trajectories.npz`

当前训练代码要求 NPZ 中同时存在以下三个字段：

```text
train_traj
val_traj
test_traj
```

轨迹数组统一使用：

```text
(trajectory_number, horizon + 1, state_dim)
```

例如二维、20-step 环境：

```text
train_traj : (1600, 21, 2)
val_traj   : (400, 21, 2)
test_traj  : (400, 21, 2)
```

CartPole 的状态维度是 4，因此可使用：

```text
train_traj : (1600, 21, 4)
val_traj   : (400, 21, 4)
test_traj  : (400, 21, 4)
```

真实数据中通常还包含：

```text
train_actions
val_actions
test_actions
```

当前 predictor 是 state-only 模型，训练时不会读取 action。

### 数据用途

训练代码会再次把 `train_traj` 随机划分为：

```text
fit set       ：用于更新模型参数，默认占 train_traj 的 90%
selection set ：用于选择最佳 checkpoint，默认占 train_traj 的 10%
```

`val_traj` 和 `test_traj` 不参与模型参数训练：

```text
val_traj  ：预留作 calibration，但当前代码没有实现 conformal calibration
test_traj ：预留作最终 evaluation
```

Normalization 的 mean 和 standard deviation 只根据 fit set 计算，避免使用 validation 或 test 信息。

### 输入检查

`data_utils.py` 会检查：

- 三个 split 是否都存在；
- 每个轨迹数组是否为三维；
- 三个 split 的 `(horizon + 1, state_dim)` 是否一致；
- 数据是否为空；
- 是否包含 NaN 或 Inf；
- 数据是否至少包含命令行要求的 horizon。

## 6. Predictor 模型

模型实现位于 `predictor_model.py`：

```python
TrajectoryTransformer
```

它直接学习：

```text
Fθ(s0) → [s0, s1, ..., sH]
```

它不是一步一步递归预测，而是由一个初始状态直接生成所有时间步。

主要结构：

1. `state_encoder` 把初始状态编码到 `d_model` 维特征空间；
2. 每个时间步有一个可训练的 `time_query`；
3. 初始状态特征与时间 query 相加；
4. Transformer Encoder 同时处理全部时间步；
5. `output_head` 把特征转换回状态空间；
6. 使用 residual connection，把初始状态加到预测结果上；
7. 第 0 步被强制设置为输入的真实初始状态。

默认模型参数：

| 参数 | 默认值 |
|---|---:|
| `d_model` | 128 |
| `nhead` | 4 |
| `num_layers` | 3 |
| `dim_feedforward` | 256 |
| `dropout` | 0.1 |

### Loss

Loss 由未来状态均方误差和末端状态均方误差组成：

```text
loss = future_MSE + terminal_weight × terminal_MSE
```

默认：

```text
terminal_weight = 0.2
```

因为 `s0` 是已知值，future MSE 只计算 `s1` 到 `sH`。

## 7. 配置默认路径

`config.py` 中定义了：

```python
DEFAULT_REAL_PATH
DEFAULT_GRID_RESULT_PATH
DEFAULT_CHECKPOINT_PATH
DEFAULT_TUBE_OUTPUT_PATH
DEFAULT_HORIZON
DEFAULT_SAMPLES_PER_CELL
```

当前附件中实际启用的是 CartPole 默认路径。被注释掉的 Brake System、Mountain Car 和 Pendulum 路径不会自动生效。

如果不希望修改源码，可以直接在命令行覆盖所有路径。优先级是：

```text
命令行参数 > config.py 默认值
```

当前代码不会根据环境名称自动寻找文件，也不会因为修改 `--env-name` 而自动更换路径。

## 8. 训练 Predictor

### 使用 `config.py` 默认路径

```bash
cd trajectory_predictor
python train_predictor.py
```

### 显式指定输入和输出

```bash
python train_predictor.py \
  --real /path/to/real_trajectories.npz \
  --checkpoint /path/to/predictor_transformer.pth \
  --horizon 20
```

常用训练参数：

```bash
python train_predictor.py \
  --real /path/to/real_trajectories.npz \
  --checkpoint /path/to/predictor_transformer.pth \
  --horizon 20 \
  --fit-ratio 0.9 \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --terminal-loss-weight 0.2 \
  --gradient-clip 1.0 \
  --patience 30 \
  --device auto \
  --seed 2025
```

`--device auto` 会优先使用 CUDA；没有 CUDA 时使用 CPU。

### Early stopping

每个 epoch 后，程序会在 selection set 上计算 loss。只有当 selection loss 变好时才覆盖 checkpoint。

如果连续 `--patience` 个 epoch 没有改进，训练提前终止。默认 patience 为 30。

## 9. Checkpoint 格式

训练完成后会生成：

```text
predictor_transformer.pth
```

其中包括：

```text
model_state_dict
state_mean
state_std
state_dim
horizon
model_config
best_epoch
best_selection_loss
source_real_trajectories
training_protocol
```

`build_tube.py` 依靠 checkpoint 中的 `state_dim`、`horizon`、normalization 参数和模型结构重新加载 predictor。

## 10. 输入二：`safety_result.json`

构建 tube 时需要一个 verification grid JSON。至少需要：

```json
{
  "grid": {
    "dims": [
      {
        "name": "state_0",
        "start": 0.0,
        "stop": 1.0,
        "num": 10,
        "step": 0.1
      }
    ]
  },
  "cells": []
}
```

程序主要使用：

```text
grid.dims
cells[i].bounds[0]
```

如果 JSON 中：

- `cells` 数量等于 grid 的 cell 总数；
- 每个 cell 都存在 `bounds[0]`；
- 初始 bounds 的 shape 为 `(state_dim, 2)`；

程序就直接使用 JSON 中的第 0 步 bounds。

否则，程序会根据 `grid.dims` 中的 `start`、`stop`、`num` 和 `step` 重建每个 cell 的初始 bounds。

模型的 `state_dim` 必须与 grid 维度一致。

## 11. 每个 Cell 如何采样

当前默认每个 cell 只选取三个确定性点：

```text
lower corner
cell center
upper corner
```

数学上，采样点位于 cell 从 lower corner 到 upper corner 的主对角线上：

```text
x(w) = lower + w × (upper - lower)
```

默认三个权重：

```text
w = 0.0, 0.5, 1.0
```

这表示每个 cell 总共采样 3 个点，而不是每个维度采样 3 个点。因此无论状态维度是 2 还是 4，每个 cell 都只运行 3 条预测轨迹。

需要注意：对于多维 cell，这三个点只覆盖一条对角线，不会覆盖其他角点和 cell 内部的所有方向。

## 12. 构建 Predictor Tube

### 使用默认路径

```bash
python build_tube.py
```

但是当前 `build_tube.py` 的 `--env-name` 默认值是 `pendulum`，而 `config.py` 实际启用的是 CartPole 路径。使用默认 CartPole 路径时，应显式传入：

```bash
python build_tube.py --env-name cartpole
```

否则模型和 grid 仍可能正常运行，但 JSON 中的 environment 会被错误标记为 `pendulum`。

### 推荐：显式指定全部参数

```bash
python build_tube.py \
  --grid-result /path/to/safety_result.json \
  --checkpoint /path/to/predictor_transformer.pth \
  --trajectory-output /path/to/predictor_trajectories.npz \
  --tube-output /path/to/predictor_tube.json \
  --env-name cartpole \
  --samples-per-cell 3 \
  --horizon 20 \
  --cell-batch-size 1024 \
  --device auto \
  --seed 2025
```

如果没有指定 `--trajectory-output`，程序会把它保存为：

```text
<tube-output 所在目录>/predictor_trajectories.npz
```

### Min/max tube

对于 cell `c`、时间步 `t` 和状态维度 `d`：

```text
lower[c,t,d] = min(该 cell 三条预测轨迹在 t,d 的值)
upper[c,t,d] = max(该 cell 三条预测轨迹在 t,d 的值)
```

最终 bounds 的排列是：

```text
(num_cells, horizon + 1, state_dim, 2)
```

最后一维：

```text
[..., 0] = lower
[..., 1] = upper
```

## 13. `predictor_trajectories.npz` 格式

当前附件源码生成的是 cell 级格式：

| 字段 | Shape | 含义 |
|---|---|---|
| `cell_indices` | `(C,)` | cell 编号 |
| `initial_bounds` | `(C,D,2)` | 每个 cell 的初始上下界 |
| `initial_states` | `(C,S,D)` | 每个 cell 的采样初始状态 |
| `trajectories` | `(C,S,H+1,D)` | 所有 cell 的预测轨迹 |
| `lower` | `(C,H+1,D)` | 每个 cell 的预测下界 |
| `upper` | `(C,H+1,D)` | 每个 cell 的预测上界 |
| `horizon` | scalar | transition 数 |
| `samples_per_cell` | scalar | 每个 cell 的轨迹数 |

其中：

```text
C = num_cells
S = samples_per_cell，默认 3
H = horizon，默认 20
D = state_dim
```

核心数组是：

```text
trajectories[cell, sample, time, state]
```

例如：

```text
CartPole    : (3600, 3, 21, 4)
Mountain Car: (6400, 3, 21, 2)
Pendulum    : (5000, 3, 21, 2)
```

### 与 `real_trajectories.npz` 的区别

`real_trajectories.npz` 按数据集 split 组织：

```text
train_traj[trajectory, time, state]
val_traj[trajectory, time, state]
test_traj[trajectory, time, state]
```

而当前 predictor 输出按 verification cell 组织：

```text
trajectories[cell, sample, time, state]
```

因此两者当前并不是相同格式：

| 对比项 | `real_trajectories.npz` | 当前 `predictor_trajectories.npz` |
|---|---|---|
| 数据组织 | train/val/test split | grid cell |
| 核心字段 | `train_traj`、`val_traj`、`test_traj` | `trajectories` |
| 轨迹维度 | 3D | 4D |
| action | 通常存在 | 不存在 |
| cell 信息 | 不存在 | 存在 |

即使把四维 `trajectories` 展平为三维，也不能保证与真实 `test_traj` 一一对应，因为二者的轨迹数量和初始状态来源不同。

如果要让 predictor NPZ 可直接替代真实轨迹 NPZ，必须另外从真实的 `train_traj`、`val_traj` 和 `test_traj` 的初始状态开始预测，并保存对应的同名三维数组。当前附件源码尚未实现这一功能。

## 14. `predictor_tube.json` 格式

JSON 顶层主要字段：

```text
method
environment
guarantee_type
sampling_strategy
samples_per_cell
horizon
state_dim
source_grid_result
checkpoint
trajectory_file
best_epoch
best_selection_loss
training_protocol
grid
cells
```

每个 cell 的结构：

```json
{
  "bounds": [],
  "raw_bounds": [],
  "initial_bounds": [],
  "trajectory_file": "predictor_trajectories.npz",
  "trajectory_index": 0
}
```

其中：

```text
bounds shape = (horizon + 1, state_dim, 2)
```

`trajectory_index` 对应 NPZ 中：

```text
trajectories[trajectory_index]
```

JSON 会明确记录：

```text
guarantee_type = sampled envelope; no formal coverage guarantee
```

## 15. 与 `compare.py` 配合

`predictor_tube.json` 使用 `grid` 和 `cells[*].bounds` 结构，可以作为 tube JSON 传给相应版本的 `compare.py`：

```bash
python compare.py \
  --env cartpole \
  --safety /path/to/predictor_tube.json \
  --real /path/to/real_trajectories.npz \
  --dwm /path/to/dwm_trajectories.npz \
  --outdir /path/to/predictor_results/cartpole
```

这里：

```text
--safety = predictor_tube.json
--real   = 真实轨迹
--dwm    = compare.py 要求的另一组轨迹数据
```

当前附件源码生成的 `predictor_trajectories.npz` 没有 `test_traj`，不能直接作为要求 `test_traj` 的 `compare.py --dwm` 输入。若直接传入，通常会出现：

```text
KeyError: test_traj
```

如果目标是让 predictor 轨迹直接与真实轨迹逐条对比，需要先实现上一节所述的 split-compatible 输出格式。

## 16. 环境兼容性

模型本身不硬编码 Pendulum、Mountain Car 或 CartPole 的动力学。只要：

- 真实轨迹的 `state_dim` 正确；
- checkpoint 的 `state_dim` 与 grid 维度一致；
- 真实轨迹长度不少于要求的 horizon；
- `real_trajectories.npz` 包含三个必需 split；

同一套代码就可以训练不同环境。

环境名称在当前 `build_tube.py` 中主要是写入 JSON 的元数据，不负责选择数据路径，也不会改变模型结构。

### Brake System

当前附件中的 Brake System 数据只有：

```text
val_traj
test_traj
```

并且是 10 个 transition：

```text
(N, 11, 2)
```

而当前训练代码：

- 强制要求 `train_traj`；
- 默认要求 horizon 20；
- 没有 `--missing-train-policy` 参数。

因此它不能直接训练当前 Brake System 数据。运行：

```bash
python train_predictor.py --missing-train-policy split-val
```

会得到：

```text
error: unrecognized arguments: --missing-train-policy split-val
```

要正式支持 Brake System，推荐重新生成独立的：

```text
train_traj
val_traj
test_traj
```

并使用：

```bash
python train_predictor.py \
  --real /path/to/brake_system/real_trajectories.npz \
  --checkpoint /path/to/brake_system/predictor_transformer.pth \
  --horizon 10
```

如果暂时从 `val_traj` 中划分训练数据，需要先修改 `data_utils.py` 和 `train_predictor.py`；当前源码不会自动执行该策略。

## 17. 常见错误

### 缺少 `train_traj`

```text
KeyError: missing keys ['train_traj']
```

原因：当前 loader 要求 `train_traj`、`val_traj` 和 `test_traj` 同时存在。

### `--missing-train-policy` 无法识别

```text
error: unrecognized arguments: --missing-train-policy split-val
```

原因：该参数不在当前附件版本的 `train_predictor.py` 中。

### 数据只有 10 步，但要求 20 步

```text
only contains 10 transition steps, but horizon=20 was requested
```

解决方法：

```bash
--horizon 10
```

### Checkpoint horizon 不够

```text
checkpoint only predicts 10 steps, but --horizon=20 was requested
```

`build_tube.py` 的 horizon 不能大于训练 checkpoint 的 horizon。

### Grid 维度和模型不一致

```text
grid_dim=... does not match model state_dim=...
```

检查 `--grid-result` 和 `--checkpoint` 是否来自同一环境。

### JSON 环境名称错误

如果没有显式设置 `--env-name`，当前默认值是 `pendulum`。即使模型和 grid 来自其他环境，JSON 仍会写成 Pendulum。

推荐始终显式传入：

```bash
--env-name cartpole
```

### `compare.py` 找不到 `test_traj`

原因：当前 `predictor_trajectories.npz` 是 cell 格式，不是 train/val/test split 格式。

## 18. 检查 NPZ

查看字段、shape 和 dtype：

```bash
python - <<'PY'
import numpy as np

path = "/path/to/predictor_trajectories.npz"
with np.load(path, allow_pickle=False) as data:
    for key in data.files:
        value = data[key]
        print(f"{key:20s} shape={value.shape} dtype={value.dtype}")
PY
```

检查是否存在 NaN 或 Inf：

```bash
python - <<'PY'
import numpy as np

path = "/path/to/predictor_trajectories.npz"
with np.load(path, allow_pickle=False) as data:
    for key in ("initial_bounds", "initial_states", "trajectories", "lower", "upper"):
        value = data[key]
        print(key, np.isfinite(value).all())
PY
```

## 19. 检查 Tube JSON

```bash
python - <<'PY'
import json
import numpy as np

path = "/path/to/predictor_tube.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

bounds = np.asarray([cell["bounds"] for cell in data["cells"]], dtype=float)

print("environment :", data["environment"])
print("cells       :", len(data["cells"]))
print("bounds shape:", bounds.shape)
print("finite      :", np.isfinite(bounds).all())
print("lower<=upper:", np.all(bounds[..., 0] <= bounds[..., 1]))
PY
```

## 20. 推荐运行顺序

```bash
# 1. 检查真实轨迹格式
python -c "import numpy as np; d=np.load('/path/to/real_trajectories.npz'); print(d.files); [print(k, d[k].shape) for k in d.files]"

# 2. 训练 predictor
python train_predictor.py \
  --real /path/to/real_trajectories.npz \
  --checkpoint /path/to/predictor_transformer.pth \
  --horizon 20

# 3. 构建 predictor tube
python build_tube.py \
  --grid-result /path/to/safety_result.json \
  --checkpoint /path/to/predictor_transformer.pth \
  --trajectory-output /path/to/predictor_trajectories.npz \
  --tube-output /path/to/predictor_tube.json \
  --env-name cartpole \
  --samples-per-cell 3 \
  --horizon 20

# 4. 检查 NPZ 和 JSON
# 使用第 18、19 节中的检查命令

# 5. 运行 compare.py
python compare.py \
  --env cartpole \
  --safety /path/to/predictor_tube.json \
  --real /path/to/real_trajectories.npz \
  --dwm /path/to/dwm_trajectories.npz \
  --outdir /path/to/predictor_results/cartpole
```

## 21. 当前版本总结

当前附件版本已经实现：

- Transformer 从初始状态直接预测完整轨迹；
- 每个 cell 总共采样 3 个点；
- 默认预测 20 个 transition；
- 所有 cell 轨迹统一保存到一个 NPZ；
- 根据 3 条轨迹的逐维最小值和最大值构建 tube；
- 输出包含 grid 和 cell bounds 的 JSON；
- checkpoint early stopping 和训练数据 normalization。

当前附件版本尚未实现：

- Brake System 缺少 `train_traj` 时的自动划分；
- `--missing-train-policy split-val`；
- 根据环境名称自动切换路径；
- `--env` 统一环境参数；
- 与 `real_trajectories.npz` 相同的 train/val/test predictor 输出格式；
- conformal calibration；
- predictor tube 的形式化覆盖保证。

实验中应始终区分：

```text
高 containment
```

和：

```text
小而精确、具有实际对比意义的 tube
```

简单扩大 tube 可以提高 containment，但不一定代表 predictor 更准确。当前三点 min/max 方法的 tube 大小主要由模型预测误差、cell 尺寸、采样位置和系统非线性共同决定。
