# Trajectory Predictor：Pendulum 周期角修正版

本版本修复了 Pendulum 在 `+π/-π` 附近的训练和 tube 表示问题，并把
cell 采样从三点主对角线改为“全部有效角点 + 中心”。

## 为什么不能只给输出层加 `π * tanh`

`tanh` 只能限制数值范围，不能表达角度的周期等价关系。比如：

```text
3.13 rad 和 -3.15 rad 的真实圆周距离很小，
普通 MSE 却会把它们当作相差接近 2π。
```

而且 `tanh` 在接近上下限时会饱和，边界附近的梯度反而更小。因此，本版
仍使用线性输出层，但对 Pendulum 采用以下流程：

```text
real theta in [-π, π)
        ↓ np.unwrap（沿时间轴）
continuous theta
        ↓ normalization + Transformer + MSE
unwrapped prediction and min/max tube
        ├─ predictor_trajectories.npz：wrap 回 [-π, π)
        └─ predictor_tube.json：跨边界时写成两个区间
```

跨边界角度 tube 示例：

```json
[3.05, 3.141592653589793, -3.141592653589793, -3.10]
```

`compare.py` 会把它解释为：

```text
[3.05, π] ∪ [-π, -3.10]
```

## 主要修改

| 文件 | 修改 |
|---|---|
| `angle_utils.py` | 新增角度 unwrap、wrap 和跨边界双区间转换 |
| `train_predictor.py` | Pendulum 训练前沿时间轴 unwrap θ，并把角度表示写入 checkpoint |
| `build_tube.py` | 拒绝旧 Pendulum checkpoint，避免误用 wrapped-theta 权重 |
| `tube_utils.py` | 内部使用 unwrapped θ 构建 tube，保存时正确 wrap；采样改为全部角点 + 中心 |
| `sampling_utils.py` | 实现全部有效角点、中心和额外 Halton 点采样 |
| `config.py` | 默认采样数从 3 改为 5 |
| `test_pendulum_support.py` | 添加周期角和采样单元测试 |

其他环境继续使用原始线性状态，不会进行角度 unwrap/wrap。

## Cell 采样

程序只把“上下界宽度非零”的状态维度视为有效维度。若有 `k` 个有效维度，
至少需要：

```text
2^k 个角点 + 1 个中心
```

当前 Pendulum grid 有两个有效维度，所以默认：

```text
4 corners + center = 5 samples per cell
```

如果 `--samples-per-cell` 大于最低数量，额外点使用确定性的 Halton 序列。
如果参数小于最低数量，程序会直接报错，而不会退回主对角线采样。

## 1. 重新训练 Pendulum

旧的 `predictor_transformer.pth` 是在 wrapped θ 和普通 MSE 上训练的，
不能复用。必须运行：

```bash
cd trajectory_predictor_pendulum_fixed

python train_predictor.py \
  --env pendulum \
  --real /home/tealab_shared/safety_results/pendulum/real_trajectories.npz \
  --checkpoint models/pendulum/predictor_transformer.pth \
  --horizon 20 \
  --epochs 300 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --device auto
```

训练日志应显示：

```text
angle repr : unwrapped_theta
```

新 checkpoint 中应包含：

```text
angle_representation = unwrapped_theta
angle_dim = 0
```

## 2. 构建 Predictor Tube

```bash
python build_tube.py \
  --env pendulum \
  --grid-result /home/tealab_shared/safety_results/pendulum/safety_result_big_cell_a16_lambda05.json \
  --real /home/tealab_shared/safety_results/pendulum/real_trajectories.npz \
  --checkpoint models/pendulum/predictor_transformer.pth \
  --trajectory-output models/pendulum/predictor_trajectories.npz \
  --tube-output models/pendulum/predictor_tube.json \
  --samples-per-cell 5 \
  --horizon 20 \
  --cell-batch-size 1024 \
  --device auto
```

输出包括：

- `predictor_trajectories.npz`
  - `train_traj`、`val_traj`、`test_traj` 与真实轨迹文件保持相同的公开格式；
  - 所有 θ 已 wrap 到 `[-π, π)`；
  - cell 级 `trajectories` 也使用 wrapped θ；
  - `lower`、`upper` 保存内部 unwrapped tube，并通过
    `tube_internal_representation=unwrapped_theta` 明确标记。
- `predictor_tube.json`
  - 普通角度区间仍是 `[lower, upper]`；
  - 跨越 `±π` 时使用四端点双区间；
  - `raw_bounds` 保留内部 unwrapped 数值，`bounds` 用于 `compare.py`。

## 3. 运行测试

```bash
python -m unittest -v test_pendulum_support.py
```

测试会检查：

- `np.unwrap` 能消除 `+π → -π` 跳变；
- 保存轨迹时能 wrap 回 `[-π, π)`；
- 跨边界 tube 会转成两个区间；
- 二维 cell 的 5 个采样点确实包含四个角点和中心。

## 4. 使用 compare.py

重新生成 `predictor_tube.json` 后，不要再用人为放大的 `--delta`：

```bash
python compare.py \
  --env pendulum \
  --safety /path/to/predictor_tube.json \
  --real /path/to/real_trajectories.npz \
  --dwm /path/to/predictor_trajectories.npz \
  --outdir /path/to/output \
  --delta 0
```

先比较修复后、不加 inflation 的结果。周期角修复解决的是边界错误，不能保证
有限采样 tube 自动达到 95% 覆盖率。如果 tube 仍偏窄，下一步应使用独立
`val_traj` 做 conformal calibration，而不是重新加入统一的大倍数。

## 5. 当前仍然存在的限制

- Predictor 仍然只输入初始状态，不显式输入控制 action。
- Tube 仍是有限采样预测的 min/max 包络，不是形式化可达集。
- 本版尚未实现 conformal calibration；`val_traj` 仍只保留作 calibration。
- 增加采样和修复周期角会改善几何表示，但最终真实轨迹覆盖率仍取决于模型误差。
