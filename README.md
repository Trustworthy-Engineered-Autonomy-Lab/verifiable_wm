# Trajectory Predictor Baseline

## 1. 分支目的

本分支在 `verifiable_wm` 项目中新增一个 **Trajectory Predictor baseline**，用于与 DWM（Deterministic World Model）生成的 reachable tube 进行对比。

DWM 通过“状态 → 图像 → 控制器 → 动力学”的闭环可达性分析得到 reachable tube；本分支不使用世界模型或符号可达性，而是直接从真实闭环轨迹中训练一个 Transformer，根据初始状态预测未来轨迹，再将多个预测轨迹转换为 Predictor Tube，并通过 Conformal Prediction 对该 Tube 进行校准。

最终比较对象为：

```text
DWM reachable tube
        vs
Conformal Predictor Tube
```

两种方法都输出按初始 cell 和时间步组织的状态范围，因此可以使用同一个 `compare.py`，在相同测试轨迹上比较覆盖率和 Tube 大小。

> 注意：Trajectory Predictor 是数据驱动的概率基线，不是 StarV 意义下的确定性形式化可达性方法。其保证来自独立校准数据上的 Conformal Prediction。

---

## 2. 整体方法

```text
real_trajectories.npz
        │
        ├── train_traj
        │      ├── fit：训练 Transformer
        │      └── selection：选择最佳 checkpoint / early stopping
        │
        ├── val_traj：Conformal calibration
        │
        └── test_traj：最终 containment evaluation

训练阶段：
initial state s0
        ↓
Transformer trajectory predictor
        ↓
predicted trajectory [s0, s1, ..., sT]
        ↓
predictor_transformer.pth

Tube 构建阶段：
verification grid / initial cells
        ↓
在每个 cell 内规则采样多个初始状态
        ↓
Transformer 预测对应轨迹
        ↓
每个时间步、每个状态维度取 min/max
        ↓
Raw Predictor Tube
        ↓
用 val_traj 计算 conformal margin Δ
        ↓
Raw Tube 上下界扩张
        ↓
Conformal Predictor Tube
        ↓
predictor_tube.json
```

### 2.1 Transformer 点轨迹预测

模型学习映射：

```text
Fθ(s0) → [ŝ0, ŝ1, ..., ŝT]
```

输入是一个初始状态，输出是完整的未来状态序列。模型内部使用：

- 初始状态编码器；
- 可学习的时间查询向量；
- Transformer Encoder；
- 状态输出层。

预测结果的第 0 步会被强制设置为输入初始状态，避免模型重复近似已知的 `s0`。

### 2.2 Raw Predictor Tube

从 `safety_result.json` 读取与 DWM verification 相同的 grid 和 initial cells。对每个 cell，在每个状态维度上均匀取 `samples_per_dim` 个点。

对于二维 MountainCar，默认：

```text
samples_per_dim = 11
samples_per_cell = 11 × 11 = 121
```

将一个 cell 内全部采样点送入 Transformer，得到多条预测轨迹；随后在每个时间步和状态维度上取预测值的最小值与最大值，形成该 cell 的 Raw Predictor Tube。

该 Raw Tube 是有限采样得到的预测包络，本身不等同于形式化 reachable set。

### 2.3 Conformal calibration

使用独立的 `val_traj`，计算每条真实轨迹相对于对应 Raw Tube 的最大越界距离：

```text
score = max over all time steps and state dimensions
        of the distance outside the Raw Tube
```

根据指定的 `alpha` 选择 conformal quantile，得到全局校准量 `delta`。然后对未来时间步的上下界进行扩张：

```text
lower = raw_lower - delta
upper = raw_upper + delta
```

默认 `alpha=0.05`，对应 95% 的边际轨迹包含目标。校准样本必须与测试样本来自相同的初始状态分布，并且必须有足够多的有效 grid 内轨迹。

### 2.4 测试评估

使用从未参与训练或校准的 `test_traj`，分别评估：

- Raw Predictor Tube；
- Conformal Predictor Tube。

主要指标包括：

- fully contained trajectories；
- trajectory containment rate；
- mean step containment；
- mean / worst maximum violation。

---

## 3. 目录结构

```text
trajectory_predictor/
├── __init__.py
├── config.py
├── data_utils.py
├── predictor_model.py
├── tube_utils.py
├── train_predictor.py
├── build_tube.py
├── README.md
└── models/
    └── mountain_car/
        ├── predictor_transformer.pth   # 训练后生成
        └── predictor_tube.json         # Tube 构建后生成
```

外部输入数据不属于本模块：

```text
real_trajectories.npz
safety_result.json
```

---

## 4. 各程序作用

### `config.py`

集中管理默认绝对路径、随机种子、计算设备和输出目录创建。

当前默认路径：

```text
真实轨迹：
/home/tealab_shared/trajectories/mountain_car/starv_state/real_trajectories.npz

Verification grid：
/home/tealab_shared/dwm_reachable_tube/mountain_car/safety_result.json

模型 checkpoint：
/home/UFAD/xinyangwang/projects/verifiable_wm/trajectory_predictor/models/mountain_car/predictor_transformer.pth

Predictor Tube：
/home/UFAD/xinyangwang/projects/verifiable_wm/trajectory_predictor/models/mountain_car/predictor_tube.json
```

所有路径都可以通过终端参数覆盖。

### `data_utils.py`

负责：

- 读取并检查 `train_traj`、`val_traj`、`test_traj`；
- 将 `train_traj` 拆分为 fit 和 selection；
- 只根据 fit 数据计算状态均值和标准差；
- 构造 PyTorch `TrajectoryDataset`。

### `predictor_model.py`

负责：

- 定义 `TrajectoryTransformer`；
- 计算 trajectory loss；
- 加载 checkpoint；
- 对大量初始状态进行批量轨迹预测；
- 完成归一化和反归一化。

### `train_predictor.py`

训练入口程序。它只训练点轨迹预测器，不生成 Tube。

主要流程：

```text
读取 real_trajectories.npz
→ 拆分 fit / selection
→ 计算 normalization
→ 训练 Transformer
→ 根据 selection loss 保存最佳模型
→ early stopping
```

输出：

```text
predictor_transformer.pth
```

### `tube_utils.py`

负责 Predictor Tube 的核心功能：

- 读取 grid；
- 将初始状态映射到 cell；
- 在 cell 内规则采样；
- 构造 Raw Predictor Tube；
- 计算 trajectory-tube violation；
- 计算 conformal score 和 `delta`；
- 扩张 Tube；
- 在 test 数据上评估 containment；
- 保存与 `compare.py` 兼容的 JSON。

### `build_tube.py`

Tube 构建入口程序。它不会重新训练模型。

主要流程：

```text
加载 predictor_transformer.pth
→ 读取 safety_result.json 中的 grid/cells
→ 每个 cell 内采样并预测
→ 构造 Raw Predictor Tube
→ val_traj 做 conformal calibration
→ test_traj 做最终评估
→ 保存 predictor_tube.json
```

---

## 5. 输入数据要求

### `real_trajectories.npz`

必须包含：

```text
train_traj
val_traj
test_traj
```

每个数组的形状必须为：

```text
(N, T+1, state_dim)
```

三个 split 必须具有相同的时间长度和状态维度。

### `safety_result.json`

至少需要包含：

```text
grid.dims
cells
```

本模块使用其中的 grid 定义和每个 cell 的初始边界。DWM 后续时间步的 reachable bounds 不参与 Predictor Tube 的生成。

为了保证实验有效，`real_trajectories.npz` 的初始状态范围应与 `safety_result.json` 的 verification grid 一致。

---

## 6. 使用方法

从项目根目录运行：

```bash
cd /home/UFAD/xinyangwang/projects/verifiable_wm
```

### 6.1 训练 Predictor

使用默认路径：

```bash
python trajectory_predictor/train_predictor.py
```

小规模运行测试：

```bash
python trajectory_predictor/train_predictor.py \
  --epochs 10 \
  --batch-size 64 \
  --patience 5 \
  --device auto
```

正式训练示例：

```bash
python trajectory_predictor/train_predictor.py \
  --real /home/tealab_shared/trajectories/mountain_car/starv_state/real_trajectories.npz \
  --checkpoint /home/UFAD/xinyangwang/projects/verifiable_wm/trajectory_predictor/models/mountain_car/predictor_transformer.pth \
  --epochs 300 \
  --batch-size 64 \
  --patience 30 \
  --device auto
```

### 6.2 构建 Predictor Tube

必须先完成训练并生成 checkpoint。

使用默认路径：

```bash
python trajectory_predictor/build_tube.py
```

小规模测试：

```bash
python trajectory_predictor/build_tube.py \
  --samples-per-dim 5 \
  --alpha 0.05 \
  --device auto
```

正式运行示例：

```bash
python trajectory_predictor/build_tube.py \
  --real /home/tealab_shared/trajectories/mountain_car/starv_state/real_trajectories.npz \
  --grid-result /home/tealab_shared/dwm_reachable_tube/mountain_car/safety_result.json \
  --checkpoint /home/UFAD/xinyangwang/projects/verifiable_wm/trajectory_predictor/models/mountain_car/predictor_transformer.pth \
  --tube-output /home/UFAD/xinyangwang/projects/verifiable_wm/trajectory_predictor/models/mountain_car/predictor_tube.json \
  --samples-per-dim 11 \
  --alpha 0.05 \
  --device auto
```

---

## 7. 输出文件

### `predictor_transformer.pth`

保存：

- Transformer 参数；
- 状态归一化参数；
- horizon 和 state dimension；
- 模型结构配置；
- 最佳 epoch 和 selection loss；
- 数据划分协议。

该文件只由 `build_tube.py` 加载，不直接传给 `compare.py`。

### `predictor_tube.json`

核心字段：

```text
cells[i]["raw_bounds"]
```

有限采样得到的 Raw Predictor Tube。

```text
cells[i]["bounds"]
```

经过 conformal inflation 的最终 Predictor Tube。`compare.py` 默认读取该字段。

JSON 还包含：

- alpha / coverage；
- conformal delta；
- calibration statistics；
- raw 和 conformal test evaluation；
- grid 信息；
- checkpoint 与数据来源。

---

## 8. 与 DWM 的比较

`predictor_transformer.pth` 不直接参与比较。先通过 `build_tube.py` 生成 `predictor_tube.json`，再用相同的 `real test trajectories` 分别评估两种 Tube。

### 8.1 DWM reachable tube

```bash
python compare.py \
  --env mountain_car \
  --safety /home/tealab_shared/dwm_reachable_tube/mountain_car/safety_result.json \
  --real /home/tealab_shared/trajectories/mountain_car/starv_state/real_trajectories.npz \
  --dwm /home/tealab_shared/trajectories/mountain_car/starv_state/dwm_trajectories.npz \
  --real-key test_traj \
  --dwm-key test_traj \
  --plot-dims 0 1 \
  --check-dims 0 1 \
  --max-steps 30 \
  --outdir trajectory_predictor/mountain_car/dwm_result
```

### 8.2 Predictor Tube

```bash
python compare.py \
  --env mountain_car \
  --safety /home/UFAD/xinyangwang/projects/verifiable_wm/trajectory_predictor/models/mountain_car/predictor_tube.json \
  --real /home/tealab_shared/trajectories/mountain_car/starv_state/real_trajectories.npz \
  --dwm /home/tealab_shared/trajectories/mountain_car/starv_state/dwm_trajectories.npz \
  --real-key test_traj \
  --dwm-key test_traj \
  --plot-dims 0 1 \
  --check-dims 0 1 \
  --max-steps 30 \
  --outdir trajectory_predictor/mountain_car/predictor_result
```

公平比较时，重点记录两次运行中的 `[Real trajectory]`：

- fully contained；
- containment rate；
- mean step containment。

还应同时比较 Tube width / area / volume，否则仅靠覆盖率无法反映保守性。两次实验必须使用完全相同的：

- `test_traj`；
- grid；
- horizon；
- checked dimensions；
- cell 定位方式。

---

## 9. 常见问题

### Conformal delta is infinite

错误示例：

```text
ValueError: conformal delta is infinite; calibration set is too small for alpha=0.05
```

原因是落在 verification grid 内的有效 `val_traj` 数量不足。对于 `alpha=0.05`，至少需要 19 条有效 calibration trajectories 才可能得到有限 quantile；实际实验建议使用更多样本。

应首先检查：

- `val_traj` 总数；
- 初始状态落在 grid 内的数量；
- trajectory 数据范围是否与 verification grid 一致。

不要直接截断 rank 或删除附加的 `inf`，否则会破坏标准 conformal 的有限样本规则。

### Transformer nested tensor warning

```text
enable_nested_tensor is True ... norm_first was True
```

这是 PyTorch 的性能提示，不是运行错误，不影响 checkpoint 或 Tube 生成。

### Real trajectory containment 明显高于 DWM trajectory containment

如果 `--safety` 使用的是 `predictor_tube.json`，这是可能的，因为 Predictor Tube 是使用真实轨迹训练和校准的，其目标是覆盖新的真实轨迹，而不是保证覆盖 DWM rollout。

DWM 与 Predictor 的公平比较应始终比较：

```text
相同 real test trajectories vs DWM reachable tube
相同 real test trajectories vs Predictor Tube
```

而不是直接比较同一次运行中的 real containment 与 DWM containment。

---

## 10. 当前方法的定位与限制

本分支提供的是一个直接的数据驱动 baseline：

- 优点：实现简单，不依赖世界模型、图像生成器和符号网络传播；
- 优点：可输出与 DWM reachable tube 格式一致的 Predictor Tube；
- 优点：可以通过 conformal calibration 获得分布无关的边际覆盖保证；
- 限制：Raw Tube 由有限初始状态采样构造，不覆盖 cell 内未采样状态的最坏情况；
- 限制：保证依赖训练、校准和测试数据的分布一致性；
- 限制：使用全局 `delta` 可能导致部分 cell 过度保守；
- 限制：高覆盖率并不一定意味着 Tube 更好，还需要结合 Tube 大小判断。

因此，该 baseline 的主要作用是回答：

> 在相同真实轨迹数据、initial grid 和测试条件下，一个直接学习初始状态到未来轨迹映射的 Transformer，经过采样包络与 conformal calibration 后，能否生成比 DWM reachable tube 更紧或更有效的概率 Tube？
