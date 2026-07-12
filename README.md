# Verifiable World Model

用可验证 image decoder 替代真实 renderer 的闭环系统：

```text
state -> renderer / decoder -> controller -> dynamics -> next state
```

目标是比较真实 renderer 与 decoder world model（DWM）的闭环轨迹，并用 StarV 对 decoder-controller-dynamics 闭环做 reachability 验证。

仓库提供 CartPole、Pendulum 和 MountainCar 的数据生成、decoder 训练、闭环 rollout 与 StarV 验证入口。
当前 CartPole 和 Pendulum 是活跃实验；MountainCar 因 controller 存在问题（occlusion heatmap 关注区域异常）
实验内容冻结，仅保持代码与配置结构和另外两个环境一致。
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

训练 decoder 的 state 和用于 StarV 对比的 trajectory state 有不同职责：

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

因此范围关系为：

```text
decoder training range  ⊇  StarV grid
sampling initial states =  StarV grid samples
compare initial states  =  StarV grid samples
```

大范围训练让 decoder 尽量覆盖从 StarV 初始 grid rollout 后会访问到的 state；StarV grid 只定义要声明安全的初始状态集合，并不限制后续轨迹必须停留在该初始框内。

## 配置职责

| 配置 | 职责 |
|---|---|
| `config/make_decoder_dataset/<env>.json` | `state_space` 定义 decoder 训练图片的 state 范围；`starv_config` 指向对应 StarV 配置。 |
| `config/train_decoder/<env>/<variant>.json` | decoder loss、训练超参数、controller 和输出 checkpoint。 |
| `config/starv_verification/<env>.json` | StarV grid、验证步数、模型权重，以及从 grid 采样 `starv_states.npz` 的数量/seed。 |
| `config/sampling/<env>.json` | controller、decoder checkpoint、dynamics；读取 StarV 生成的 trajectory states，不独立采样初始 state。 |

三个环境的完整 state（取自 `dynamic.py`）：

```text
CartPole  [position, velocity, angle, angular_velocity]   4 维
Pendulum  [theta, omega]                                  2 维
MountainCar [position, velocity]                          2 维
```

只有 CartPole 的 decoder 输入是完整 state 的子集：只吃第 `[0, 2]` 维，即 `[position, angle]`
（`decoder_state_indices=[0, 2]`）；完整四维 state 仍用于 dynamics、trajectory 和 StarV cell 定位。
Pendulum 和 MountainCar 没有这层裁剪，decoder 直接吃完整 state，`decoder_states.npz` 里的
`{split}_states` 就是完整维度。

## 主要产物

默认目录：

```text
datasets/<env>/data/dataset_v1/
```

| 文件 | 含义 |
|---|---|
| `decoder_states.npz` | decoder 训练数据，包含 `{split}_states` 与 `{split}_images`。CartPole state 是二维 decoder 输入。 |
| `saliency_occlusion.npz` | controller occlusion heatmap。 |
| `starv_states.npz` | 从 StarV grid 生成的完整初始 state；供 real/DWM trajectory 和 compare 共用。 |
| `real_trajectories.npz` | 从 `starv_states.npz` 出发，使用真实 renderer 的完整闭环轨迹。 |
| `transition_dataset.npz` | 真实 renderer 下的单步 `(s,a,s')` transition。 |
| `dwm_trajectories_<variant>.npz` | 用指定 decoder 替代 renderer 得到的完整闭环轨迹。 |

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

## 常用工作流

主要通过 `notebooks/` 下的 notebook 跑（方便边跑边看中间结果）；每个 notebook 都是对应 `.py` 脚本的薄
封装（`parse_args(argv)` / `run(args)` 或等价的可调用函数），脚本本身也保留 CLI 可用性，下面每步都附
等价命令行。notebook 第一个 cell 会自动定位仓库根目录，用 `starv_shared` kernel
（`/home/tealab_shared/starv/env/starv_shared`）打开即可。

### 1. 生成 decoder 训练数据

`notebooks/generate_dataset.ipynb`：`run_make_decoder_dataset("cartpole")` 等，仅在需要重新生成训练
图片时运行。等价命令行：

```bash
python make_decoder_dataset.py config/make_decoder_dataset/cartpole.json
python make_decoder_dataset.py config/make_decoder_dataset/pendulum.json
```

这会重写训练 `decoder_states.npz`、图片，以及 StarV 对齐的 trajectory 文件。

### 2. 只生成 StarV 对齐的产物

同一个 notebook：`run_make_decoder_dataset("cartpole", starv_only=True)`，保留现有 decoder 训练
数据时使用。等价命令行：

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

`notebooks/saliency_diagnostics.ipynb` 第 0 节跑 `precompute_saliency_maps`；
`notebooks/train_decoder.ipynb` 跑 `train_decoder`。等价命令行：

```bash
python saliency_map/scripts/precompute_saliency_maps.py \
  --config config/make_decoder_dataset/cartpole.json

python train_decoder.py config/train_decoder/cartpole/saliency.json
python train_decoder.py config/train_decoder/cartpole/intensity.json
```

Pendulum 使用相同模式，替换相应环境配置。

### 4. 生成 DWM trajectory

同 `notebooks/generate_dataset.ipynb`：`run_sampling("cartpole", decoder_variant="saliency")`。等价
命令行：

```bash
python sampling.py config/sampling/cartpole.json --decoder-variant saliency
python sampling.py config/sampling/pendulum.json --decoder-variant saliency
```

sampling 会读取 `starv_states.npz`。因此必须先运行上一步的 starv-only 或完整数据生成。

### 5. 运行 StarV

`verify.py` 依赖 `mpi4py` 多进程启动，天然没法塞进 notebook 的单进程 kernel，这一步继续用命令行 /
`mpirun`：

```bash
python verify.py config/starv_verification/cartpole.json
python verify.py config/starv_verification/pendulum.json
```

实际大规模验证通常通过 MPI 启动；输出默认写到：

```text
results/<env>/safety_result.json
```

### 6. 比较 trajectory 与 reachable tube

`notebooks/compare_tube.ipynb`：`run_compare_tube("cartpole", variant="saliency")`。等价命令行：

```bash
python compare.py --env cartpole --variant saliency
python compare.py --env pendulum --variant saliency
```

`compare.py` 会强制检查：

```text
starv_states == real_traj[:,0,:] == dwm_traj[:,0,:]
starv_states 全部位于 StarV grid 内
```

CartPole 默认 plot/check 维度为 `(0, 2)`，即 position-angle。velocity 与 angular velocity 虽参与 dynamics，但当前初始 grid 中固定为 0。

## Decoder 训练

`model.py` 包含：

- `Controller`：图片输入，输出 action；训练 decoder 时固定。
- `Decoder`：state 输入，输出 96×96 灰度图。

支持三种像素加权方式：

```text
intensity  论文的亮度阈值 baseline
saliency   用 controller occlusion heatmap 作为空间权重
hybrid     intensity + saliency，仅作诊断
```

saliency 权重为：

$$
w_{i,p}=1+\alpha H_{i,p}
$$

每张图按像素均值归一化。总损失包括加权重建误差与 controller action consistency：

$$
L_{total}=L_{rec}^{w}+\lambda_{ctrl}L_{ctrl}
$$

最佳 checkpoint 按 `training.selection_metric` 保存；例如：

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

StarV 对每个初始 grid cell 计算的是 decoder world model 的 reachable tube。`compare.py` 再检查真实轨迹和 DWM 轨迹是否处在对应 cell 的各时间步 bounds 内。

注意：未膨胀 tube 不直接给真实系统安全保证。若要复现论文中的统计保证，仍需在与 StarV 初始分布匹配的 trajectory 上计算论文定义的 L1 non-conformity score，并实施 Theorem 1 的 tube inflation。

## 目录速览

| 路径 | 内容 |
|---|---|
| `make_decoder_dataset.py` | 训练数据、StarV-aligned trajectory states 与真实轨迹生成。 |
| `sampling.py` | 同一 trajectory states 上的 transition 与 DWM rollout。 |
| `train_decoder.py` | decoder 训练。 |
| `verify.py` | StarV/MPI 验证入口。 |
| `compare.py` | trajectory 与 reachable tube 比较。 |
| `starv_verification/` | StarV 版本模型、dynamics 与 verifier。 |
| `saliency_map/` | saliency 计算方法库（`methods.py`）、预计算主线脚本与诊断脚本，见 `saliency_map/README.md`。 |
| `notebooks/` | 上面「常用工作流」里各步骤的 notebook 入口；逻辑都在对应 `.py`，notebook 只是薄封装+可视化。 |
| `config/` | 各阶段配置。 |
| `report/` | 每日实验记录与阶段性结论。 |
