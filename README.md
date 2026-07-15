# Verifiable World Model

本项目研究如何用可验证的 image decoder 替代真实 renderer，并对得到的闭环系统进行
reachability analysis。系统的数据流如下：

```text
state -> renderer / decoder -> controller -> dynamics -> next state
```

我们首先比较真实 renderer 与 decoder world model（DWM）的闭环轨迹，再使用 StarV 验证
decoder-controller-dynamics 闭环的 reachable set。

仓库提供 CartPole、Pendulum 和 MountainCar 的数据生成、decoder 训练、闭环 rollout 与 StarV 验证入口。
当前实验以 CartPole 和 Pendulum 为主。MountainCar 的 controller 仍有问题，occlusion heatmap
的关注区域不合理，因此暂时冻结实验内容，只维护代码和配置结构的一致性。

当前 saliency 主线参数为：CartPole `(alpha=8, lambda_ctrl=0.1)`，Pendulum
`(alpha=16, lambda_ctrl=0.5)`。

## 运行环境

实验默认使用：

```text
/home/tealab_shared/starv/env/starv_shared
```

`verify.py` 依赖 MPI、StarV、pybdr 和 gurobi。渲染在 headless 环境运行，脚本会设置：

```text
PYGLET_HEADLESS=True
```

## 数据流

decoder 训练 state 和 StarV 验证 state 的范围与用途不同：

```text
make_decoder_dataset.state_space        大范围
        |
        +--> decoder_states.npz + images --> train_decoder.py

starv_verification.grid                小范围
        |
        +--> starv_states.npz
                +--> real_trajectories.npz
                +--> sampling.py --> dwm_trajectories_<variant>.npz
                +--> compare.py --> StarV cell lookup
```

二者满足以下关系：

```text
decoder training range  ⊇  StarV grid
sampling initial states =  StarV grid samples
compare 使用 trajectory t=0 state 定位 StarV cell
```

decoder 使用较大的 state range 训练，以覆盖从 StarV 初始 grid rollout 后可能访问的区域。
StarV grid 只定义需要验证的初始状态集合，不要求后续轨迹始终停留在初始 box 内。

## 配置职责

| 配置 | 职责 |
|---|---|
| `config/make_decoder_dataset/<env>.json` | `state_space` 定义 decoder 训练图片的 state 范围；`starv_config` 指向对应 StarV 配置。 |
| `config/train_decoder/<env>/<variant>.json` | decoder loss、训练超参数、controller 和输出 checkpoint。 |
| `config/starv_verification/<env>.json` | StarV grid、验证步数、模型权重，以及从 grid 采样 `starv_states.npz` 的数量/seed。 |
| `config/sampling/<env>.json` | controller、decoder checkpoint、dynamics；读取 StarV 生成的 trajectory states，不独立采样初始 state。 |

三个环境在 `dynamic.py` 中定义的完整 state 为：

```text
CartPole  [position, velocity, angle, angular_velocity]   4 维
Pendulum  [theta, omega]                                  2 维
MountainCar [position, velocity]                          2 维
```

CartPole 是唯一需要裁剪 decoder 输入的环境。它只使用第 `[0, 2]` 维，即
`[position, angle]`（`decoder_state_indices=[0, 2]`）；完整四维 state 仍用于 dynamics、
trajectory 和 StarV cell 定位。Pendulum 与 MountainCar 的 decoder 直接接收完整 state，
因此 `decoder_states.npz` 中的 `{split}_states` 保留完整维度。

## 主要产物

默认目录：

```text
datasets/<env>/data/dataset_v1/
```

| 文件 | 含义 |
|---|---|
| `decoder_states.npz` | decoder 训练数据，包含 `{split}_states` 与 `{split}_images`。CartPole state 是二维 decoder 输入。 |
| `saliency_occlusion.npz` | controller occlusion heatmap。 |
| `starv_states.npz` | 从 StarV grid 生成的完整初始 state；供 real/DWM trajectory 共用。 |
| `real_trajectories.npz` | 从 `starv_states.npz` 出发，使用真实 renderer 的完整闭环轨迹。 |
| `dwm_trajectories_<variant>.npz` | decoder 替代 renderer 后得到的完整闭环轨迹；`variant` 和 `decoder_weights` 字段记录所用 checkpoint。 |

旧的无 variant 文件 `dwm_trajectories.npz` 已停用并删除。当前 variant 为 `old`、`intensity` 和
`saliency`。

常用 shape：

```text
CartPole decoder_states.npz:
  train_states           (N, 2)
  train_images           (N, 1, 96, 96)

CartPole starv_states.npz:
  test_states            (N, 4)

CartPole real/DWM trajectories:
  test_traj              (N, 31, 4)   # t=0 + 30 steps
  test_actions           (N, 30, 1)
```

### 1. 生成 decoder 训练数据

在 `notebooks/generate_dataset.ipynb` 中运行 `run_make_decoder_dataset("cartpole")`。
只有需要重新生成训练图片时才运行完整流程。对应的命令行为：

```bash
python make_decoder_dataset.py config/make_decoder_dataset/cartpole.json
python make_decoder_dataset.py config/make_decoder_dataset/pendulum.json
```

完整生成会重写 `decoder_states.npz` 中的训练 state 和图片，同时更新与 StarV 对齐的 trajectory
文件。

### 2. 只生成 StarV 对齐的产物

如果要保留现有 decoder 训练数据，可在同一个 notebook 中运行
`run_make_decoder_dataset("cartpole", starv_only=True)`。对应的命令行为：

```bash
python make_decoder_dataset.py config/make_decoder_dataset/cartpole.json --starv-only
python make_decoder_dataset.py config/make_decoder_dataset/pendulum.json --starv-only
```

该模式只写入：

```text
starv_states.npz
real_trajectories.npz
```

### 3. 计算 saliency 并训练 decoder

在 `notebooks/saliency_diagnostics.ipynb` 第 0 节运行 `precompute_saliency_maps`，然后在
`notebooks/train_decoder.ipynb` 中运行 `train_decoder`。对应的命令行为：

```bash
python saliency_map/scripts/precompute_saliency_maps.py \
  --config config/make_decoder_dataset/cartpole.json

python train_decoder.py config/train_decoder/cartpole/saliency.json
python train_decoder.py config/train_decoder/cartpole/intensity.json
```

Pendulum 的运行方式相同，只需换成对应配置。

`notebooks/train_decoder.ipynb` 也提供完整的 saliency `alpha × lambda_ctrl` 消融入口：

```text
alpha       = [0.5, 1, 2, 4, 8, 16, 32]
lambda_ctrl = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
seed        = 2025
```

CartPole 和 Pendulum 各有 49 个训练点，不筛选 seed。网格实现位于 `ablation.py`，notebook
只负责调用和显示表格。每个点使用独立目录：

```text
dwm_weight/now_weight/<env>/alpha_lambda_grid/
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

```bash
python sampling.py config/sampling/cartpole.json --decoder-variant saliency
python sampling.py config/sampling/pendulum.json --decoder-variant saliency
```

sampling 直接读取本地 `datasets/<env>/data/dataset_v1/starv_states.npz`，所以此前必须至少完成一次
完整生成或 `starv-only` 生成。

当前 sampling 是 DWM-only：只运行 `decoder -> controller -> analytic dynamics` 的闭环，不再生成
没有下游消费者的 `transition_dataset.npz`。旧 transition helper 暂时保留在 `sampling.py` 并标记
为停用，后续如果恢复 learned dynamics 实验可以重新启用。

`dwm_trajectories_<variant>.npz` 内部保存 `variant` 和 `decoder_weights`，因此不再依赖容易被后一次
运行覆盖的 `metadata.json` 来判断来源。

### 5. 运行 StarV

`verify.py` 依赖 `mpi4py` 多进程启动，不适合放进 notebook 的单进程 kernel，因此继续从命令行
运行：

```bash
python verify.py config/starv_verification/cartpole.json
python verify.py config/starv_verification/pendulum.json
```

大规模验证通常使用 MPI 启动，输出默认写入：

```text
results/<env>/safety_result.json
```

### 6. 比较 trajectory 与 reachable tube

`compare.py` 按 `--env` 自动使用当前仓库中的本地 trajectory NPZ：

```bash
python compare.py --env cartpole
python compare.py --env pendulum
```

默认输入和输出为：

```text
results/<env>/safety_result.json
datasets/<env>/data/dataset_v1/real_trajectories.npz
datasets/<env>/data/dataset_v1/dwm_trajectories_saliency.npz
results/<env>/compare_plot/
```

`compare.py` 没有 `--variant` 参数。比较其他 variant 时使用 `--dwm` 显式指定文件，例如：

```bash
python compare.py \
  --env cartpole \
  --dwm datasets/cartpole/data/dataset_v1/dwm_trajectories_intensity.npz \
  --outdir results/cartpole/compare_plot_intensity
```

脚本根据每条 trajectory 的 `t=0` state 定位 StarV cell，再逐时间步检查真实轨迹和 DWM 轨迹是否
位于对应 bounds 内。它不直接读取 `starv_states.npz`。

CartPole 默认绘制和检查 `(0, 2)` 两个维度，即 position-angle。velocity 和 angular velocity
仍参与 dynamics，只是在当前初始 grid 中固定为 0。

## Decoder 训练

`model.py` 包含：

- `Controller`：图片输入，输出 action；训练 decoder 时固定。
- `Decoder`：state 输入，输出 96×96 灰度图。

目前支持三种像素加权方式：

```text
intensity  基于亮度阈值的论文 baseline
saliency   使用 controller occlusion heatmap 作为空间权重
hybrid     intensity + saliency，仅用于诊断
```

saliency 权重为：

$$
w_{i,p}=1+\alpha H_{i,p}
$$

每张图按像素均值归一化。总损失包括加权重建误差与 controller action consistency：

$$
L_{total}=L_{rec}^{w}+\lambda_{ctrl}L_{ctrl}
$$

训练根据 `training.selection_metric` 保存最佳 checkpoint，例如：

```text
decoder_best_total.pth
```

## Rollout 指标与 StarV 比较

真实与 DWM 从同一个 `s_0` 出发：

```text
real: renderer -> controller -> dynamics
DWM : decoder  -> controller -> dynamics
```

同一时间步的 full-state L2 偏差为：

$$
d_t=\lVert s_t^{dwm}-s_t^{real}\rVert_2
$$

CartPole 直接对四维完整 state 计算 L2。Pendulum 的 theta 差先映射为最短圆周差，再与 omega
一起计算 full-state L2，避免跨越 `-π/π` 时产生接近 `2π` 的伪误差。完整消融表对 validation 和
test 分别报告：

```text
mean_step_l2   所有 trajectory、所有 t=0..30 的均值
final_l2       t=30 的均值
max_l2_mean    每条 trajectory 最大偏差的均值
max_l2_p95     每条 trajectory 最大偏差的 95% 分位数
```

`ablation.py` 将单帧 controller/pixel MSE 与闭环 L2 合并，并在网格根目录写出
`training_metrics.csv`、`rollout_l2.csv` 和 `combined_metrics.csv`。validation 用于选择
`alpha/lambda_ctrl`；test 只用于最终候选报告。主线晋升必须在 notebook 中显式调用
`promote_mainline(..., force=True)`，不会由排序结果自动覆盖。

StarV 为每个初始 grid cell 计算 decoder world model 的 reachable tube。`compare.py` 根据轨迹的
初始 state 找到对应 cell，然后逐时间步检查真实轨迹和 DWM 轨迹是否位于 bounds 内。

未经 inflation 的 tube 不能直接作为真实系统的安全保证。要复现论文中的统计保证，还需要在与
StarV 初始分布一致的 trajectory 上计算 L1 non-conformity score，并按 Theorem 1 对 tube 做
inflation。

## 测试

从仓库根目录运行当前全部测试：

```bash
MPLCONFIGDIR=/tmp/matplotlib-codex \
  /home/tealab_shared/starv/env/starv_shared/bin/python \
  -m unittest discover -s tests -v
```

当前共 26 项测试，正常结果为 `OK`。这些测试使用 mock 和 `/tmp` 中的临时文件，不会训练模型、
运行完整 sampling/StarV、覆盖正式 NPZ 或修改 notebook；完整运行通常不到 1 秒。

测试分别覆盖：

- `test_sampling.py`：variant 文件名以及 `variant`、`decoder_weights` 溯源字段；
- `test_starv_states.py`：StarV grid 范围、完整状态维度和本地 NPZ 加载；
- `test_ablation.py`：消融网格、断点续跑、rollout 溯源、L2 指标和 mainline 晋升保护。

运行时可能出现 Gym 维护状态或系统用户 ID 的环境警告；只要最终显示 `Ran 26 tests` 和 `OK`，就表示
测试通过。真正的失败会显示 `FAIL` 或 `ERROR`，并返回非零退出码。

## 文件树

```text
verifiable_wm/
├── README.md
├── config/
│   ├── make_decoder_dataset/   # 各环境的数据生成配置
│   ├── sampling/               # DWM rollout 配置
│   ├── starv_verification/     # StarV grid、权重和验证参数
│   └── train_decoder/          # 各环境、各 variant 的训练配置
├── datasets/
│   ├── cartpole/data/dataset_v1/
│   ├── pendulum/data/dataset_v1/
│   └── mountain_car/data/dataset_v1/
├── dwm_weight/                 # decoder checkpoints 与训练指标
├── notebooks/
│   ├── generate_dataset.ipynb
│   ├── train_decoder.ipynb
│   └── saliency_diagnostics.ipynb
├── report/                     # 每日工作记录
├── results/                    # StarV 验证和对比结果
├── saliency_map/
│   ├── README.md
│   ├── methods.py              # saliency 方法与公共加载逻辑
│   └── scripts/
│       ├── precompute_saliency_maps.py
│       └── diagnostics/        # heatmap、重建和 renderer 诊断
├── starv_verification/
│   ├── dynamic.py
│   ├── model.py
│   └── verifiers.py
├── tests/
│   ├── test_ablation.py
│   ├── test_sampling.py
│   └── test_starv_states.py
├── tools/
│   └── visualize.py
├── make_decoder_dataset.py     # 训练数据、StarV states 与真实轨迹生成
├── ablation.py                 # alpha-lambda 网格、DWM rollout、L2 表与显式晋升
├── sampling.py                 # variant-aware DWM 闭环 rollout
├── train_decoder.py            # decoder 训练
├── verify.py                   # StarV/MPI 验证入口
├── compare.py                  # trajectory 与 reachable tube 对比
├── dynamic.py                  # 真实环境 dynamics
├── env.py                      # renderer 与环境封装
├── model.py                    # controller 和 decoder
└── utils.py                    # 数据采样与通用工具
```
