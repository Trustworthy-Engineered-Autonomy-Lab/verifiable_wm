# Trajectory Predictor

本目录实现一个基于 Transformer 的闭环轨迹预测器。它从真实闭环轨迹中学习：

```text
初始状态 s0  ->  完整未来轨迹 [s0, s1, ..., sH]
```

训练完成后，程序可以：

1. 根据真实轨迹的初始状态生成 Predictor 预测轨迹；
2. 在 verification grid 的每个 cell 中采样初始状态；
3. 将采样点的预测轨迹合并成 Predictor tube；
4. 使用独立真实轨迹进行 Conformal 校准；
5. 比较 Raw、CP-D、CP-R 三种 tube 的 coverage、signed margin 和面积。

Predictor 只学习初始状态到闭环轨迹的映射，不读取或预测 action，也不会修改原始
`real_trajectories.npz`、`safety_result.json`、DWM 或 StarV 程序。

---

## 1. 整体逻辑

```text
real_trajectories.npz
        │
        ├── 训练数据 ──────────────> train_predictor.py
        │                                  │
        │                                  └──> predictor_transformer.pth
        │
        ├── 独立校准真实轨迹 ──────> 校准输入
        │                                  │
        └── 各 split 初始状态 ─────> predictor_trajectories.npz
                                           │
safety_result.json                         │
        │                                  │
        └──> build_predictor.py <── checkpoint
                    │
                    └──> predictor_tube.json

真实测试轨迹 + Predictor 测试轨迹 + Predictor tube
        │
        └──> predictor_signed_tube_margin.py
                    │
                    ├── Raw tube 指标
                    ├── CP-D tube 指标
                    ├── CP-R tube 指标
                    └── JSON、CSV 和比较图片
```

完整流程分为三个阶段。

### 阶段一：训练轨迹 Predictor

`train_predictor.py` 使用真实闭环轨迹训练 Transformer。模型输入一条轨迹的初始
状态，直接输出从第 0 步到第 $H$ 步的完整状态序列。

### 阶段二：生成预测轨迹和 Predictor tube

`build_predictor.py` 使用训练好的 checkpoint 完成两类预测：

- 对真实数据中的初始状态进行预测，得到 `predictor_trajectories.npz`；
- 对 verification grid 中每个 cell 的 3 个采样点进行预测，并取逐时刻 min/max，
  得到 `predictor_tube.json`。

如果源 NPZ 没有 `train_traj`，程序需要从 `val_traj` 中拆出一部分用于训练。此时
它还会把剩余的独立校准样本另存为
`conformal_real_trajectories.npz`。如果源 NPZ 已有独立 `train_traj`，原始
`val_traj` 本身就是校准集，不需要额外复制。

### 阶段三：校准和评估

`predictor_signed_tube_margin.py` 使用未参与模型训练的独立真实轨迹进行
Conformal 校准，然后在测试轨迹上比较原始 tube 和两种校准 tube。

---

## 2. 目录中的程序

| 文件 | 是否直接运行 | 作用 |
|---|---:|---|
| `config.py` | 否 | 设置环境、输入输出路径、训练参数和 tube 构建参数 |
| `predictor_model.py` | 否 | 定义 Transformer、loss、checkpoint 加载和批量推理 |
| `train_predictor.py` | 是 | 训练 Predictor 并保存最佳 checkpoint |
| `build_predictor.py` | 是 | 生成预测轨迹、真实校准集和 Predictor tube |
| `predictor_conformal.py` | 可选 | 单独计算 Predictor 的 CP-D 校准半径 |
| `predictor_signed_tube_margin.py` | 是 | 综合评估 Raw、CP-D 和 CP-R 三种 tube |

### 2.1 `config.py`

`config.py` 是训练和构建阶段的配置中心，主要控制：

```text
ENVIRONMENT              环境名称
REAL_TRAJECTORIES         真实轨迹 NPZ
GRID_RESULT               verification grid / safety result JSON
CHECKPOINT                模型 checkpoint 输出位置
TRAJECTORY_OUTPUT         Predictor 轨迹输出位置
TUBE_OUTPUT               Predictor tube 输出位置
CONFORMAL_REAL_OUTPUT     独立真实校准集输出位置
HORIZON                   预测步数
SAMPLES_PER_CELL          每个 cell 的采样数量
MISSING_TRAIN_POLICY      缺少 train split 时的处理方式
DEVICE                    CPU 或 CUDA
```

`train_predictor.py` 和 `build_predictor.py` 直接读取该文件，不接收命令行参数。
运行前应先确认环境名称、数据路径、horizon 和输出目录。

### 2.2 `predictor_model.py`

该文件定义核心模型：

```python
TrajectoryTransformer
```

模型执行：

$$
s_0 \longrightarrow [\hat{s}_0,\hat{s}_1,\ldots,\hat{s}_H]
$$

输入 shape：

```text
(batch_size, state_dim)
```

输出 shape：

```text
(batch_size, horizon + 1, state_dim)
```

模型结构可以概括为：

```text
初始状态
-> state encoder
-> 加入各时间步的可学习 time query
-> Transformer Encoder
-> output head
-> 完整预测轨迹
```

程序会强制：

```python
prediction[:, 0, :] = initial_state
```

因此预测轨迹的第 0 步与输入初始状态完全相同。

训练 loss 为完整轨迹误差与终点误差之和：

$$
L =
\operatorname{MSE}(\hat{s}_{1:H},s_{1:H})
+ \lambda_{\mathrm{terminal}}
\operatorname{MSE}(\hat{s}_H,s_H)
$$

该文件还负责：

- 使用训练集统计量对状态归一化；
- 从 `.pth` 恢复模型结构和权重；
- 分 batch 运行预测；
- 将预测结果反归一化。

### 2.3 `train_predictor.py`

这是第一个主要运行入口。

处理流程：

```text
读取 real_trajectories.npz
-> 检查 split、shape、horizon、NaN 和 Inf
-> 准备 Predictor 训练数据
-> 划分 fit set 与 selection set
-> 使用 fit set 计算 normalization
-> 训练 Transformer
-> 根据 selection loss 保存最佳模型
-> early stopping
-> 写入 predictor_transformer.pth
```

其中：

- fit set 用于更新模型参数；
- selection set 用于选择最佳 epoch；
- calibration set 不参与训练和 checkpoint 选择；
- test set 只用于最终评估。

checkpoint 不仅保存模型权重，还保存：

```text
model_state_dict
state_mean / state_std
state_dim / horizon
model_config
best_epoch / best_selection_loss
environment
训练和校准数据索引
源数据数量和指纹
training_protocol
```

数据索引和指纹用于确保构建阶段使用的仍是训练时的同一份数据及同一套划分。
如果旧 checkpoint 不包含这些信息，需要重新训练。

### 2.4 `build_predictor.py`

这是第二个主要运行入口，必须在训练完成后运行。它完成三项工作。

#### A. 生成 Predictor 轨迹

程序从真实轨迹中提取初始状态：

```python
initial_states = real_trajectories[:, 0, :]
```

然后调用 Transformer 预测完整未来轨迹，保存为：

```text
predictor_trajectories.npz
```

校准 split 中保存独立校准初始状态对应的 Predictor 预测；测试 split 中保存测试
初始状态对应的 Predictor 预测。

NPZ 中的 actions 来自真实数据，只用于保持格式和样本对应关系。Predictor 本身
既不读取 action，也不预测 action。

#### B. 生成独立真实校准集

当训练阶段使用 `split_val` 策略时，程序根据 checkpoint 中保存的 calibration
indices，从原始真实 NPZ 中提取对应的真实轨迹和 actions，保存为：

```text
conformal_real_trajectories.npz
```

该文件中的校准轨迹与 `predictor_trajectories.npz` 中的预测校准轨迹一一对应：

```text
conformal_real_trajectories.npz / val_traj
              真实轨迹
                  ↕
predictor_trajectories.npz / val_traj
              预测轨迹
```

它不是 `predictor_trajectories.npz` 的重复副本，而是计算 Predictor 预测误差所需
的真实答案。

如果源 NPZ 已有独立 `train_traj`，该文件不会生成。此时原始
`real_trajectories.npz` 的 `val_traj` 从未参与训练，可以直接作为真实校准数据。

#### C. 构建 Predictor tube

程序从 `safety_result.json` 读取 verification grid。在每个 cell 中使用：

```text
lower corner
center
upper corner
```

作为 3 个初始状态。对三点分别预测完整轨迹后，在每个时间步、每个状态维度取：

```python
lower = predictions.min(axis=0)
upper = predictions.max(axis=0)
```

最终把每个 cell 的时序 min/max 包络保存到：

```text
predictor_tube.json
```

cell 的三条采样轨迹只在构建过程中存在，不写入
`predictor_trajectories.npz`。

### 2.5 `predictor_conformal.py`

该文件计算 CP-D（Conformal Prediction based on trajectory Distance）。

对第 $i$ 对真实/预测校准轨迹，先计算 nonconformity score：

$$
\delta_i =
\max_t
\left\|s_{i,t}^{\mathrm{real}}-s_{i,t}^{\mathrm{pred}}\right\|_2
$$

再使用 finite-sample conformal rank：

$$
k=\left\lceil(n+1)(1-\alpha)\right\rceil
$$

从排序后的误差中取得膨胀半径 $\Gamma_D$。

该程序可以单独运行，也会被最终评估入口调用。命令行参数 `--dwm` 沿用了原项目
接口名称；在本目录中应传入 `predictor_trajectories.npz`。

### 2.6 `predictor_signed_tube_margin.py`

这是最终综合评估入口，用于比较：

| 方法 | 含义 |
|---|---|
| Raw | 每个 cell 三点预测得到的原始 min/max 包络 |
| CP-D | 根据真实轨迹与 Predictor 轨迹的 L2 误差膨胀 Raw tube |
| CP-R | 根据真实轨迹相对 Raw tube 的越界程度膨胀 Raw tube |

Signed margin 表示状态相对 tube 边界的位置：

```text
margin > 0  状态位于 tube 内部
margin = 0  状态位于 tube 边界
margin < 0  状态位于 tube 外部
```

一条轨迹的 signed margin 是所有时间步和检查维度中的最小边界距离。只要其中
一个状态在任一检查维度越界，该轨迹的最终 margin 就会小于 0。

程序最终统计：

- trajectory coverage rate；
- signed margin 的 mean、min 和 max；
- 平均 tube area；
- CP-D 半径；
- CP-R 半径；
- 有效轨迹和有效 cell 数量。

它会只读调用项目根目录 `compare.py` 中的匹配和绘图函数，不会修改
`compare.py`。目录结构应为：

```text
verifiable_wm/
├── compare.py
└── trajectory_predictor/
    ├── predictor_signed_tube_margin.py
    └── ...
```

---

## 3. 数据划分与防止数据泄漏

推荐的数据角色为：

```text
训练来源
├── fit set          更新模型参数
└── selection set    选择最佳 checkpoint

calibration set      计算 CP-D 和 CP-R
test set             最终评价
```

这四个角色应严格分离。尤其不能使用参与 Predictor 训练的轨迹进行 Conformal
校准，否则得到的误差半径和 coverage 会过于乐观。

如果输入 NPZ 已包含独立训练、验证和测试 split：

```text
train_traj  -> Predictor 训练来源，再拆成 fit/selection
val_traj    -> 独立 calibration set
test_traj   -> 最终评价
```

这种情况下不会另外生成 `conformal_real_trajectories.npz`，因为原始
`val_traj` 已经是独立校准集。

如果输入数据缺少 `train_traj` 且：

```python
MISSING_TRAIN_POLICY = "split_val"
```

程序会确定性地将原始 `val_traj` 拆为 Predictor 训练来源和独立 calibration set，
再把训练来源拆成 fit/selection。实际数量由 `DERIVED_TRAIN_RATIO` 和
`TRAIN_FIT_RATIO` 决定。

例如原始 `val_traj` 有 400 条、两级比例分别为 0.8 和 0.9 时：

```text
原始 val：400
├── Predictor 训练来源：320
│   ├── fit：288
│   └── selection：32
└── 独立 calibration：80

原始 test：400，只用于最终测试
```

原始 NPZ 始终只读。拆分索引与源数据指纹保存在 checkpoint 中，构建阶段会重新
验证并复用同一组 calibration indices。

---

## 4. 输入文件

### 4.1 `real_trajectories.npz`

典型字段：

```text
train_traj / val_traj / test_traj
train_actions / val_actions / test_actions
```

并非每个数据集都必须同时存在三个 trajectory split；具体处理方式由
`MISSING_TRAIN_POLICY` 决定。

基本要求：

- trajectory shape 为 `(N, H+1, state_dim)`；
- action shape 为 `(N, H, action_dim)`；
- 同一 split 的 trajectory 与 action 样本数量一致；
- 数据 horizon 与 `config.py` 一致；
- 不包含 NaN 或 Inf。

### 4.2 `safety_result.json`

必须提供 verification grid 和 cells，主要字段为：

```text
grid.dims
cells
```

`grid.dims` 的维数必须与 checkpoint 中的 `state_dim` 一致。该文件仅提供 cell
初始范围；Predictor 不使用其中的 StarV reachable tube 作为训练标签。

---

## 5. 输出文件

### 5.1 `predictor_transformer.pth`

训练好的 PyTorch checkpoint，包含模型权重、normalization、网络结构、训练信息、
数据划分索引和源数据指纹。

它不是轨迹文件，不能直接传给 `compare.py` 或 signed-margin 程序。

### 5.2 `predictor_trajectories.npz`

保存 Predictor 对真实数据初始状态的预测结果。主要包括：

```text
val_traj / val_actions
test_traj / test_actions
环境、checkpoint、源索引等元数据
```

具体包含哪些 split 取决于输入数据和构建策略。

其中：

- `*_traj` 是 Predictor 生成的状态轨迹；
- `*_actions` 是从真实 NPZ 中复制的对应 actions；
- calibration split 用于与真实校准轨迹配对；
- test split 用于最终评价。

### 5.3 `conformal_real_trajectories.npz`（按需生成）

保存独立 calibration set 的真实轨迹及对应 actions。它只用于 Conformal 校准，
不用于模型训练，也不替代最终测试数据。

该文件只在 `MISSING_TRAIN_POLICY="split_val"` 实际生效时生成。源数据已有
`train_traj` 时，应直接使用原始 NPZ 中未参与训练的 `val_traj` 进行校准。

两个 NPZ 的核心区别：

| 文件 | `val_traj` 的含义 |
|---|---|
| `conformal_real_trajectories.npz` | 独立校准样本的真实轨迹 |
| `predictor_trajectories.npz` | 同一批初始状态对应的 Predictor 预测轨迹 |

### 5.4 `predictor_tube.json`

保存每个 cell 的时序 min/max 包络，主要字段包括：

```text
method
environment
guarantee_type
sampling_strategy
samples_per_cell
horizon
state_dim
grid
cells[].bounds
```

Raw Predictor tube 是有限采样包络，不是形式化 reachable set。JSON 会明确记录：

```text
sampled envelope; no formal coverage guarantee
```

### 5.5 最终评估结果

`predictor_signed_tube_margin.py` 会生成：

```text
tube_table_metrics.csv
tube_table_metrics_raw.csv
tube_calibrations.json
signed_tube_margin_summary.json

real_vs_raw_tube.png
model_vs_raw_tube.png
real_vs_cp_d_tube.png
model_vs_cp_d_tube.png
real_vs_cp_r_tube.png
model_vs_cp_r_tube.png
```

这些文件分别保存汇总指标、校准半径和可视化结果。当前 `main()` 不生成逐轨迹的
`*_signed_tube_margins.csv`。

---

## 6. 配置与运行

### 6.1 放入项目

推荐目录结构：

```text
verifiable_wm/
├── compare.py
└── trajectory_predictor/
    ├── README.md
    ├── config.py
    ├── predictor_model.py
    ├── train_predictor.py
    ├── build_predictor.py
    ├── predictor_conformal.py
    └── predictor_signed_tube_margin.py
```

进入项目根目录：

```bash
cd /home/UFAD/xinyangwang/projects/verifiable_wm
```

核心依赖：

```text
Python 3.9+
NumPy
PyTorch
Matplotlib
```

### 6.2 修改配置

打开：

```text
trajectory_predictor/config.py
```

至少检查：

```python
ENVIRONMENT
REAL_TRAJECTORIES
GRID_RESULT
HORIZON
OUTPUT_DIRECTORY
MISSING_TRAIN_POLICY
```

确保 `HORIZON` 与真实 NPZ 的轨迹长度以及 checkpoint 一致。

### 6.3 训练

```bash
python trajectory_predictor/train_predictor.py
```

输出：

```text
trajectory_predictor/models/<environment>/
└── predictor_transformer.pth
```

### 6.4 构建 Predictor 输出

```bash
python trajectory_predictor/build_predictor.py
```

输出：

```text
trajectory_predictor/models/<environment>/
├── predictor_transformer.pth
├── predictor_trajectories.npz
├── predictor_tube.json
└── conformal_real_trajectories.npz  # 仅 split_val 时生成
```

构建程序会检查：

- checkpoint 与源数据是否一致；
- state dimension 与 horizon 是否一致；
- calibration indices 是否有效；
- 真实/预测校准轨迹是否一一对应；
- 初始状态和 actions 是否对齐；
- grid 维数是否正确；
- 输出是否包含 NaN 或 Inf。

验证通过后才写入最终文件。

### 6.5 可选：单独计算 CP-D

```bash
python trajectory_predictor/predictor_conformal.py \
  --env <environment> \
  --real trajectory_predictor/models/<environment>/conformal_real_trajectories.npz \
  --dwm trajectory_predictor/models/<environment>/predictor_trajectories.npz \
  --split val \
  --alpha 0.05 \
  --output trajectory_predictor/models/<environment>/conformal_result.json
```

这一步不是必须的；最终评估程序会自动计算 CP-D。

### 6.6 最终评估

```bash
python trajectory_predictor/predictor_signed_tube_margin.py \
  --env <environment> \
  --safety trajectory_predictor/models/<environment>/predictor_tube.json \
  --real /path/to/real_trajectories.npz \
  --model trajectory_predictor/models/<environment>/predictor_trajectories.npz \
  --calibration-real /path/to/independent_calibration_real.npz \
  --outdir trajectory_predictor/predictor_results/<environment>/signed_margin
```

`--calibration-real` 应指向独立真实校准集。不要把参与 Predictor 训练的数据用于
Conformal 校准：

- 源 NPZ 有 `train_traj`：可将原始 `real_trajectories.npz` 传给该参数，程序使用
  其中未参与训练的 `val_traj`；
- 使用 `split_val`：传入构建阶段生成的
  `conformal_real_trajectories.npz`。

默认情况下，程序会保护已有结果并拒绝覆盖非空输出目录。如需明确覆盖：

```bash
... --overwrite
```

---

## 7. CP-D 与 CP-R 的区别

### CP-D：校准轨迹预测误差

CP-D 直接比较真实轨迹与 Predictor 轨迹：

```text
真实校准轨迹
      ↕ L2 trajectory distance
Predictor 校准轨迹
      ↓
Gamma_D
      ↓
膨胀 Raw tube
```

它回答的是：

> Predictor 的完整轨迹预测误差通常需要多大的半径覆盖？

### CP-R：校准 Raw tube 的越界程度

CP-R 直接计算真实校准轨迹相对于 Raw Predictor tube 的 signed margin：

```text
真实校准轨迹
      ↕ Raw tube boundary
越界分数
      ↓
Gamma_R
      ↓
膨胀 Raw tube
```

它回答的是：

> Raw tube 还需要向外扩大多少，才能达到目标校准覆盖水平？

因此：

| 方法 | 校准对象 | 反映的问题 |
|---|---|---|
| CP-D | 真实轨迹与 Predictor 轨迹 | 模型轨迹预测误差 |
| CP-R | 真实轨迹与 Raw Predictor tube | 原始 tube 的覆盖缺口 |

---

## 8. 检查 `split_val` 生成的校准 NPZ

以下检查适用于构建阶段生成了
`conformal_real_trajectories.npz` 的情况。查看字段、shape 和 dtype：

```bash
python - <<'PY'
import numpy as np

root = "trajectory_predictor/models/<environment>"
paths = [
    f"{root}/conformal_real_trajectories.npz",
    f"{root}/predictor_trajectories.npz",
]

for path in paths:
    print(f"\n=== {path} ===")
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            value = data[key]
            print(f"{key:32s} shape={str(value.shape):18s} dtype={value.dtype}")
PY
```

检查校准数据是否正确配对：

```bash
python - <<'PY'
import numpy as np

root = "trajectory_predictor/models/<environment>"
with np.load(
    f"{root}/conformal_real_trajectories.npz",
    allow_pickle=False,
) as real, np.load(
    f"{root}/predictor_trajectories.npz",
    allow_pickle=False,
) as pred:
    print("real val:", real["val_traj"].shape)
    print("pred val:", pred["val_traj"].shape)
    print(
        "initial states equal:",
        np.array_equal(
            real["val_traj"][:, 0, :],
            pred["val_traj"][:, 0, :],
        ),
    )
    print(
        "actions equal:",
        np.array_equal(real["val_actions"], pred["val_actions"]),
    )
PY
```

正确结果应满足：

```text
真实和预测 val_traj 的 shape 相同
initial states equal: True
actions equal: True
```

---

## 9. 常见问题

### checkpoint 缺少数据划分信息

旧 checkpoint 可能没有保存 calibration indices 或源数据指纹。重新训练并构建：

```bash
python trajectory_predictor/train_predictor.py
python trajectory_predictor/build_predictor.py
```

### 数据指纹不一致

当前 `real_trajectories.npz` 与训练 checkpoint 使用的数据不同，或相关轨迹的内容、
顺序发生了变化。确认 `config.py` 中的数据路径；如果数据已经更新，需要重新训练。

### horizon 不一致

真实轨迹的时间长度应为：

```text
HORIZON + 1
```

actions 的时间长度应为：

```text
HORIZON
```

修改 `config.py` 后必须重新训练，不能直接用不同 horizon 的旧 checkpoint。

### 输出目录非空

最终评估默认不覆盖旧结果。可改用新的 `--outdir`，或确认后添加：

```bash
--overwrite
```

### 找不到 `compare.py`

从项目根目录运行，并确保：

```text
verifiable_wm/compare.py
verifiable_wm/trajectory_predictor/
```

### CUDA 或内存不足

在 `config.py` 中减小训练或推理 batch size，也可以设置：

```python
DEVICE = "cpu"
```

---

## 10. 当前方法的边界

1. Predictor 只根据初始状态预测完整轨迹，不把 actions 作为输入。
2. 该模型预测的是训练数据中闭环控制策略对应的轨迹，不是任意 action 下的动力学。
3. 每个 cell 仅采样 lower corner、center 和 upper corner。
4. 三点位于 cell 的一条对角线上，不能代表 cell 内所有可能初始状态。
5. Raw Predictor tube 是采样轨迹的 min/max 包络，不是形式化 reachable set。
6. CP-D 和 CP-R 的有效性依赖独立、可交换的校准数据和正确的数据划分。
7. 如果环境、控制器、轨迹 horizon 或数据分布改变，应重新训练和校准。

---

## 11. 最短运行流程

```bash
cd /home/UFAD/xinyangwang/projects/verifiable_wm

# 1. 在 config.py 中确认环境、输入路径和输出路径

# 2. 训练 Predictor
python trajectory_predictor/train_predictor.py

# 3. 生成预测轨迹、真实校准集和 Predictor tube
python trajectory_predictor/build_predictor.py

# 4. 校准并评估 Raw、CP-D 和 CP-R tube
python trajectory_predictor/predictor_signed_tube_margin.py \
  --env <environment> \
  --safety trajectory_predictor/models/<environment>/predictor_tube.json \
  --real /path/to/real_trajectories.npz \
  --model trajectory_predictor/models/<environment>/predictor_trajectories.npz \
  --calibration-real /path/to/independent_calibration_real.npz \
  --outdir trajectory_predictor/predictor_results/<environment>/signed_margin
```

如果训练时使用 `split_val`，最后一个命令中的校准路径应为：

```text
trajectory_predictor/models/<environment>/conformal_real_trajectories.npz
```
