# Trajectory Predictor

该目录实现一个基于 Transformer 的轨迹预测器：模型根据初始状态
\(s_0\) 直接预测未来轨迹，并在 DWM verification grid 的每个 cell
中采样 3 个初始点，用三条预测轨迹的逐步最小值和最大值构造
Predictor Tube。

当前版本的默认实验设置为：

- 每个 cell **总共采样 3 个点**，不是每个维度采样 3 个点；
- 三个点为 cell 的 lower corner、center 和 upper corner；
- 预测 **20 个 transition steps**；
- 每条轨迹包含 \(s_0,s_1,\ldots,s_{20}\)，因此共有 21 个状态；
- 所有 cell 的预测轨迹集中保存在一个
  `predictor_trajectories.npz` 文件中；
- `predictor_tube.json` 中的上下界直接由三条轨迹的 min/max 决定；
- 不使用 conformal inflation。

> 当前 Predictor Tube 是有限采样得到的经验包络，不是形式化
> reachable set，也不保证覆盖 cell 内所有未采样的初始状态。

## 1. 工作流程

```text
trajectory data (.npz)
        │
        ├── 读取 train_traj / val_traj / test_traj
        ├── 截取 s0...s20
        └── 训练 Transformer
                    │
                    ▼
        predictor_transformer.pth

safety_result.json + predictor_transformer.pth
        │
        ├── 读取 verification grid 和 cell
        ├── 每个 cell 采样 lower / center / upper
        ├── 分别预测 20 步
        ├── 保存所有 cell 的预测轨迹
        └── 对三条轨迹逐步、逐维取 min/max
                    │
                    ├── predictor_trajectories.npz
                    └── predictor_tube.json
```

## 2. 文件说明

```text
trajectory_predictor/
├── README.md
├── __init__.py
├── config.py
├── data_utils.py
├── predictor_model.py
├── train_predictor.py
├── build_tube.py
└── tube_utils.py
```

| 文件 | 作用 |
| --- | --- |
| `config.py` | 默认路径、默认 horizon、随机种子和 device 设置 |
| `data_utils.py` | 读取、检查、截断和归一化轨迹数据 |
| `predictor_model.py` | Transformer 模型、loss、checkpoint 加载和批量推理 |
| `train_predictor.py` | 训练 Predictor 并保存 `.pth` checkpoint |
| `build_tube.py` | 读取 grid 和 checkpoint，生成 NPZ 与最终 tube JSON |
| `tube_utils.py` | cell 采样、轨迹汇总、min/max tube 和 JSON 保存 |

## 3. 环境依赖

代码需要：

- Python 3
- NumPy
- PyTorch

请在已经能够运行本项目其他 PyTorch 脚本的环境中执行。例如：

```bash
conda activate starv_shared
```

## 4. 训练 Predictor

### 4.1 输入轨迹格式

训练输入是一个 `.npz` 文件，必须包含以下三个数组：

```text
train_traj
val_traj
test_traj
```

三个数组的形状都必须是：

```text
(num_trajectories, T + 1, state_dim)
```

输入数据至少需要包含 20 个 transition steps，即第二个维度至少为
21。即使输入轨迹包含 30 步，默认训练也只使用
\(s_0,\ldots,s_{20}\)。

训练时：

- `train_traj` 被进一步划分为 parameter-fit set 和
  checkpoint-selection set；
- `val_traj` 保留为 calibration split，但当前无 conformal
  calibration，因此不会参与训练；
- `test_traj` 不参与训练。

### 4.2 训练命令

建议显式填写输入和输出路径：

```bash
python trajectory_predictor/train_predictor.py \
  --real /path/to/trajectories.npz \
  --checkpoint trajectory_predictor/models/pendulum/predictor_transformer.pth \
  --horizon 20 \
  --epochs 300 \
  --device auto
```

常用参数：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--horizon` | `20` | transition steps 数量 |
| `--fit-ratio` | `0.9` | `train_traj` 中用于拟合模型的比例 |
| `--epochs` | `300` | 最大训练轮数 |
| `--batch-size` | `64` | 训练 batch size |
| `--learning-rate` | `1e-4` | AdamW learning rate |
| `--patience` | `30` | early stopping patience |
| `--device` | `auto` | 自动选择 CUDA 或 CPU |

查看全部参数：

```bash
python trajectory_predictor/train_predictor.py --help
```

### 4.3 Checkpoint

训练结果保存在 `predictor_transformer.pth`。该文件主要包含：

- Transformer 权重 `model_state_dict`；
- 模型结构参数 `model_config`；
- `state_mean` 和 `state_std`；
- `state_dim` 和 `horizon`；
- 最佳 epoch 和 selection loss；
- 训练数据来源与 split 信息。

`.pth` 是模型 checkpoint，不是预测轨迹。预测轨迹保存在后续生成的
`.npz` 文件中。

## 5. 构建 Predictor Tube

### 5.1 输入

构建 tube 需要：

1. `safety_result.json`：提供 DWM verification 使用的 grid 和 cells；
2. `predictor_transformer.pth`：训练好的 Predictor checkpoint。

如果 JSON 中每个 cell 已包含有效的初始 `bounds[0]`，代码会直接使用；
否则会根据 `grid.dims` 重建所有 cell 的初始边界。

### 5.2 构建命令

```bash
python trajectory_predictor/build_tube.py \
  --grid-result /path/to/safety_result.json \
  --checkpoint trajectory_predictor/models/pendulum/predictor_transformer.pth \
  --tube-output trajectory_predictor/models/pendulum/predictor_tube.json \
  --trajectory-output trajectory_predictor/models/pendulum/predictor_trajectories.npz \
  --samples-per-cell 3 \
  --horizon 20 \
  --env-name pendulum \
  --device auto
```

其中：

- `--samples-per-cell 3` 表示每个 cell **总共**采样 3 个初始状态；
- `--horizon 20` 表示预测 20 次状态转移，输出 21 个状态；
- `--trajectory-output` 必须是一个以 `.npz` 结尾的文件路径；
- 如果省略 `--trajectory-output`，默认在
  `predictor_tube.json` 同级目录生成 `predictor_trajectories.npz`；
- `--cell-batch-size` 是模型推理时的 batch size，不改变每个 cell
  的采样点数。

查看全部参数：

```bash
python trajectory_predictor/build_tube.py --help
```

如果 checkpoint 能够预测 30 步，可以通过 `--horizon 20` 只保留前
20 步；如果 checkpoint 的 horizon 小于 20，程序会报错。为了让训练
和 tube 构建设置完全一致，推荐重新训练 20-step checkpoint。

## 6. 三点采样与 Tube 计算

对状态维度为 \(D\) 的 cell，初始边界为：

```text
initial_bounds.shape == (D, 2)
```

代码在 lower corner 到 upper corner 的对角线上等距采样。默认
`samples_per_cell=3` 时得到：

```text
p0 = lower corner
p1 = (lower corner + upper corner) / 2
p2 = upper corner
```

对第 \(c\) 个 cell、时间步 \(t\) 和状态维度 \(d\)，tube 定义为：

\[
L_{c,t,d}=\min_{k\in\{0,1,2\}}\hat{s}_{c,k,t,d},
\qquad
U_{c,t,d}=\max_{k\in\{0,1,2\}}\hat{s}_{c,k,t,d}.
\]

因此，最终 bounds 的形状为：

```text
(num_cells, 21, state_dim, 2)
```

最后一个维度中的 `0` 是 lower，`1` 是 upper。

## 7. 输出格式

### 7.1 `predictor_trajectories.npz`

所有 cell 的轨迹保存在同一个压缩 NPZ 文件中：

| key | 形状 | 含义 |
| --- | --- | --- |
| `cell_indices` | `(num_cells,)` | cell 的线性索引 |
| `initial_bounds` | `(num_cells, state_dim, 2)` | 每个 cell 的初始范围 |
| `initial_states` | `(num_cells, 3, state_dim)` | 每个 cell 的 3 个采样点 |
| `trajectories` | `(num_cells, 3, 21, state_dim)` | 所有预测轨迹 |
| `lower` | `(num_cells, 21, state_dim)` | 三条轨迹逐元素最小值 |
| `upper` | `(num_cells, 21, state_dim)` | 三条轨迹逐元素最大值 |
| `horizon` | scalar | transition steps，默认为 `20` |
| `samples_per_cell` | scalar | 每个 cell 的采样数，默认为 `3` |

读取第 10 个 cell：

```python
import numpy as np

with np.load("predictor_trajectories.npz", allow_pickle=False) as data:
    cell_index = 10
    initial_states = data["initial_states"][cell_index]  # (3, state_dim)
    trajectories = data["trajectories"][cell_index]     # (3, 21, state_dim)
    lower = data["lower"][cell_index]                    # (21, state_dim)
    upper = data["upper"][cell_index]                    # (21, state_dim)
```

如果需要通过 `cell_indices` 查找 cell，而不是假设数组位置与 cell
索引相同：

```python
with np.load("predictor_trajectories.npz", allow_pickle=False) as data:
    wanted_cell = 10
    trajectory_index = np.flatnonzero(
        data["cell_indices"] == wanted_cell
    )[0]
    trajectories = data["trajectories"][trajectory_index]
```

### 7.2 `predictor_tube.json`

JSON 的顶层包含：

- `method`、`environment` 和 `guarantee_type`；
- `sampling_strategy`、`samples_per_cell` 和 `horizon`；
- grid、checkpoint 和集中 NPZ 的路径；
- `cells`：所有 cell 的 tube。

每个 cell 的核心字段为：

```text
cells[i]["bounds"][step][state_dim] = [lower, upper]
```

每个 cell 还包含：

| 字段 | 含义 |
| --- | --- |
| `raw_bounds` | 与 `bounds` 相同，均为三条轨迹的直接 min/max |
| `initial_bounds` | 该 cell 的原始初始边界 |
| `trajectory_file` | 集中 NPZ 文件的路径 |
| `trajectory_index` | 该 cell 在 NPZ 第一个维度中的位置 |

所有 cell 都指向同一个 `trajectory_file`，通过不同的
`trajectory_index` 读取对应数据。

## 8. 结果检查

快速查看 JSON 元数据：

```bash
jq \
  '{method, samples_per_cell, horizon, trajectory_file, cell_count: (.cells | length)}' \
  trajectory_predictor/models/pendulum/predictor_tube.json
```

检查 NPZ 形状以及 min/max 是否一致：

```python
import numpy as np

with np.load("predictor_trajectories.npz", allow_pickle=False) as data:
    trajectories = data["trajectories"]

    print("trajectories:", trajectories.shape)
    print("initial_states:", data["initial_states"].shape)
    print("lower:", data["lower"].shape)
    print("upper:", data["upper"].shape)

    np.testing.assert_allclose(
        data["lower"],
        trajectories.min(axis=1),
    )
    np.testing.assert_allclose(
        data["upper"],
        trajectories.max(axis=1),
    )

print("NPZ min/max check passed")
```

## 9. 与 `compare.py` 配合

比较时应保证 Predictor、DWM 和 real trajectories 使用相同的：

- environment；
- verification grid；
- 最大比较步数；
- 状态维度和状态顺序；
- 测试轨迹。

20-step Predictor 的比较命令示例：

```bash
python compare.py \
  --env pendulum \
  --safety trajectory_predictor/models/pendulum/predictor_tube.json \
  --real /path/to/real_trajectories.npz \
  --dwm /path/to/dwm_trajectories.npz \
  --max-steps 20 \
  --outdir trajectory_predictor/predictor_results/pendulum
```

## 10. 当前版本的主要修改

相较于原始 Predictor，本版本进行了以下修改：

1. 将采样参数从“每个维度的采样数”改为“每个 cell 的总采样数”；
2. 默认每个 cell 仅采样 lower corner、center、upper corner 三个点；
3. 默认 horizon 从 30 个 transition steps 改为 20；
4. tube 直接使用三条预测轨迹的逐步、逐维 min/max；
5. 移除 conformal calibration 和 conformal inflation；
6. 将所有 cell 的预测轨迹集中保存到一个 NPZ 文件；
7. 在 JSON 中增加 `trajectory_file` 和 `trajectory_index`，用于定位
   NPZ 中对应 cell 的轨迹；
8. 保留 `bounds` / `raw_bounds` 结构，便于现有比较程序读取。

## 11. 注意事项

- 三点只覆盖 cell 对角线上的 lower、center 和 upper，不代表覆盖
  整个高维 cell；
- 减少采样点会降低计算量，但也可能遗漏 cell 内其他位置产生的极值；
- 当前 tube 没有 conformal coverage guarantee；
- `predictor_trajectories.npz` 的大小约与
  `num_cells × 3 × 21 × state_dim` 成正比；
- 建议在实验记录中同时保存 checkpoint、grid JSON、集中 NPZ 和最终
  tube JSON，以保证结果可复现。
