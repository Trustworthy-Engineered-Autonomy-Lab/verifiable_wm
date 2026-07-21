# Trajectory Predictor

该文件夹包含 Predictor baseline 的全部代码和生成结果。除了两个输入数据文件外，不会在项目其他位置生成 Predictor 文件。

## 目录结构

```text
verifiable_wm/
├── trajectory_predictor/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── predictor_model.py
│   ├── tube_utils.py
│   ├── train_predictor.py
│   ├── build_tube.py
│   ├── README.md
│   ├── predictor_transformer.pth   # 训练后自动生成
│   └── predictor_tube.json         # 构建 tube 后自动生成
│
├── results/
│   └── mountain_car/
│       └── safety_result.json      # 外部输入，不属于 Predictor 输出
│
└── ...
```

真实轨迹默认从服务器共享文件读取：

```text
/home/tealab_shared/trajectories/mountain_car/real_trajectories.npz
```

## 文件职责

- `train_predictor.py`：读取真实轨迹并训练 Transformer。
- `build_tube.py`：读取 checkpoint 和 verification grid，构建 Raw Predictor Tube，再进行 conformal calibration。
- `predictor_model.py`：Transformer 结构、checkpoint 加载和批量预测。
- `data_utils.py`：真实轨迹读取、fit/selection 划分和归一化。
- `tube_utils.py`：grid、cell 采样、tube、conformal score 和 JSON 保存。
- `config.py`：集中定义默认路径。

## 默认路径

```text
输入真实轨迹：
/home/tealab_shared/trajectories/mountain_car/real_trajectories.npz

输入 Grid：
<项目根目录>/results/mountain_car/safety_result.json

模型输出：
<项目根目录>/trajectory_predictor/predictor_transformer.pth

Tube 输出：
<项目根目录>/trajectory_predictor/predictor_tube.json
```

所有路径仍可通过终端参数修改。

## 1. 训练 Predictor

从项目根目录运行：

```bash
python trajectory_predictor/train_predictor.py
```

小规模测试：

```bash
python trajectory_predictor/train_predictor.py \
  --epochs 10 \
  --patience 5 \
  --device auto
```

正式运行：

```bash
python trajectory_predictor/train_predictor.py \
  --real /home/tealab_shared/trajectories/mountain_car/real_trajectories.npz \
  --checkpoint trajectory_predictor/predictor_transformer.pth \
  --epochs 300 \
  --batch-size 64 \
  --patience 30 \
  --device auto
```

## 2. 构建 Predictor Tube

小规模测试：

```bash
python trajectory_predictor/build_tube.py \
  --samples-per-dim 5 \
  --device auto
```

正式运行：

```bash
python trajectory_predictor/build_tube.py \
  --real /home/tealab_shared/trajectories/mountain_car/real_trajectories.npz \
  --grid-result results/mountain_car/safety_result.json \
  --checkpoint trajectory_predictor/predictor_transformer.pth \
  --tube-output trajectory_predictor/predictor_tube.json \
  --samples-per-dim 11 \
  --alpha 0.05 \
  --device auto
```

## 数据用途

```text
train_traj
├── 90% fit：更新 Transformer 参数
└── 10% selection：选择最佳 checkpoint

val_traj：Conformal calibration
test_traj：最终 containment evaluation
```

`predictor_tube.json` 中：

- `cells[i]["raw_bounds"]`：采样得到的 Raw Predictor Tube。
- `cells[i]["bounds"]`：经过 conformal inflation 的 Predictor Tube。
