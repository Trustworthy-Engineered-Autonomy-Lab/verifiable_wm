# Verifiable World Model

这份仓库现在主要在做一件事：用一个可验证的 image decoder 替代真实 renderer，再把这个 decoder 接到 controller 和 dynamics 后面，比较闭环轨迹，最后接 StarV 做形式化验证。

当前主线先放在 CartPole 上。MountainCar 和 Pendulum 的数据生成、动力学和验证入口都在，但主要实验还没有像 CartPole 一样完整收口。

## 运行背景

实验默认用 `starv_shared` 环境，也就是 `/home/tealab_shared/starv/env/starv_shared`。`verify.py` 还依赖 `mpi4py`、StarV、pybdr、gurobi。渲染侧在 notebook 和脚本里都按 headless 处理，关键环境变量是 `PYGLET_HEADLESS=True`。

## 主线流程

CartPole 当前的实验链条是：

- `make_decoder_dataset.py` 生成 decoder 训练用的 `(state, image)`，同时保存真实 renderer 闭环下的 `real_trajectories.npz`。
- 再用 `saliency_map/scripts/precompute_saliency_maps.py` 对真实图片算 controller saliency，产物是 `saliency_occlusion.npz`。
- `train_decoder.py` 训练 decoder，当前主线比较 intensity baseline 和 saliency 方法。
- `sampling.py` 生成真实 renderer 下的 `transition_dataset.npz`，也用训练好的 decoder 生成 `dwm_trajectories_<variant>.npz`。
- 轨迹偏差比较的是 `real_trajectories.npz` 和 `dwm_trajectories_<variant>.npz`，同一个初始 state、同一个时间步上算 full-state L2。
- 最后才是 `verify.py` / `compare.py`：StarV 先在 grid cell 上做 reachability，`compare.py` 再检查真实/DWM 轨迹和 reachable tube 的关系。

## 当前 CartPole 主线状态

CartPole decoder 现在主要看两组：

- `intensity`：论文 baseline，按像素亮度阈值加权；
- `saliency`：当前方法，用 occlusion heatmap 做空间权重。

当前 saliency 主线配置在：

```text
config/train_decoder/cartpole/saliency.json
```

当前参数是 $\alpha=8.0$、$\lambda_{ctrl}=0.1$，best checkpoint 按 `total_loss` 选。

对应权重目录：

```text
dwm_weight/now_weight/cartpole/saliency/
```

里面主要看：

```text
decoder_last.pth
decoder_best_total.pth
metrics.json
```

`intensity` 对照组在：

```text
dwm_weight/now_weight/cartpole/intensity/
```

`hybrid` 现在只当诊断组，不是后续主线比较对象。

## 配置文件

`config/` 按实验环节分目录：`make_decoder_dataset` 管 decoder 数据和真实闭环轨迹，`sampling` 管 transition 数据和 DWM rollout，`train_decoder` 管 decoder 训练，`starv_verification` 管 StarV grid 和步数。

几个容易混的字段：

- `state_space`：只控制采样初始 state 的范围。它不会限制 rollout 后的 state。
- `decoder_state_indices`：CartPole 当前是 `[0, 2]`，也就是 Decoder 只吃 position 和 angle。
- `rollout_steps`：数据生成和 sampling 里当前是 30。
- `num_steps`：StarV verifier 的验证步数，CartPole 当前也是 30。
- `seed_pool` / `num_pool`：只在 `make_decoder_dataset.py` 里有意义，用来先采一个 pool 再切 train/val；`sampling.py` 不用这两个字段。

现在有一个还没最终定下来的问题：`make_decoder_dataset.py` / `sampling.py` 的 CartPole `state_space` 还是比较大的全局范围，而 StarV grid 关注的是更小的区域，比如：

```text
pos   : 0.00 ~ 0.60
vel   : 0.00
angle : 0.060 ~ 0.120
avel  : 0.00
```

这件事后面要单独处理。严格说，`sampling` 的初始范围应该贴近 StarV grid；decoder 训练范围则最好覆盖从这个初始 grid 出发 30 步内实际会访问到的 state 区域，而不只是初始小框。

## 数据文件

数据默认在：

```text
datasets/<env>/data/<dataset_name>/
```

CartPole 当前主线是：

```text
datasets/cartpole/data/dataset_v1/
```

主要文件：

| 文件 | 由谁生成 | 用途 |
|---|---|---|
| `states.npz` | `make_decoder_dataset.py` | decoder 训练数据，包含 `{split}_states` 和 `{split}_images` |
| `real_trajectories.npz` | `make_decoder_dataset.py` | 真实 renderer 闭环轨迹，后面当 ground truth |
| `saliency_occlusion.npz` | `precompute_saliency_maps.py` | saliency / hybrid 训练用 heatmap |
| `transition_dataset.npz` | `sampling.py` | 真实 renderer 下的单步 `(s, a, s')` 数据 |
| `dwm_trajectories_<variant>.npz` | `sampling.py` | decoder 替代 renderer 后跑出的闭环轨迹 |
| `metadata.json` | 两个数据脚本都会写 | 保存生成数据时用的 config；当前会被后跑的脚本覆盖 |

常见 shape：

```text
states.npz:
  train_states      (N, 2)          # CartPole: [position, angle]
  train_images      (N, 1, 96, 96)

real_trajectories.npz:
  train_traj        (N, 31, 4)      # 初始点 + 30 步
  train_actions     (N, 30, 1)

dwm_trajectories_<variant>.npz:
  train_traj        (N, 31, 4)
  train_actions     (N, 30, 1)
```

## Decoder 训练

`model.py` 里只有两个网络：

- `Controller`：输入图片，输出 action；
- `Decoder`：输入 2 维 state，输出 96x96 灰度图。

训练时 controller 固定，不重训。变的是 decoder。

`train_decoder.py` 支持三种加权方式：

```text
intensity : 按目标图像亮度阈值加权
saliency  : 按 controller saliency heatmap 加权
hybrid    : intensity + saliency
```

也就是 saliency 分支先把 heatmap 变成像素权重：

$$
w_{i,p}=1+\alpha H_{i,p}
$$

每张图的权重再做一次均值归一化：

$$
\bar w_{i,p} =
\frac{w_{i,p}}
{\frac{1}{P}\sum_{q=1}^{P} w_{i,q}}
$$

当前 total loss 实际按下面这个式子算：

$$
L_{total}(\theta)=
\frac{1}{BP}
\sum_{i=1}^{B}\sum_{p=1}^{P}
\bar w_{i,p}
\left(D_{\theta}(s_i)_p-y_{i,p}\right)^2
+
\lambda_{ctrl}
\frac{1}{BA}
\sum_{i=1}^{B}\sum_{a=1}^{A}
\left(C(D_{\theta}(s_i))_a-C(y_i)_a\right)^2
$$

当前 best checkpoint 用 `training.selection_metric` 选，CartPole 主线是 `total_loss`，所以文件名是：

```text
decoder_best_total.pth
```

这里的 $s_i$ 是 decoder 输入 state，CartPole 当前只取 `[position, angle]`；$y_i$ 是真实 renderer 图片；$D_\theta$ 是正在训练的 decoder；$C$ 是固定 controller；$H$ 是 saliency heatmap；$B$ 是 batch size；$P$ 是像素数；$A$ 是 action 维度；$\alpha$ 控制 saliency 权重强度；$\lambda_{ctrl}$ 控制 action 一致性项强度。

`train_decoder.ipynb` 里的 `run_train(...)` 会直接改 config 后调用 `train_decoder.train(...)`。脚本保留 alpha、lambda、seed、output directory 这些 override，主要给 sweep 和补实验用。

消融实验目录：

```text
dwm_weight/now_weight/cartpole/saliency_alpha_sweep/alpha_<value>/seed_<seed>/
dwm_weight/now_weight/cartpole/lambda_ablation/<intensity|saliency_alpha8>/lambda_<value>/seed_<seed>/
```

这些目录只是实验记录，不作为下游 pipeline 默认输入。

## Rollout 偏差

闭环偏差比较的是同一个初始 state 出发后的两条轨迹：

```text
real : 真实 renderer -> controller -> dynamics
DWM  : decoder      -> controller -> dynamics
```

每个时间步算 full-state L2：

$$
d_t = \lVert s_t^{dwm} - s_t^{real} \rVert_2
$$

CartPole state 是：

```text
[position, velocity, angle, angular_velocity]
```

所以这个偏差不是“差了几步”，而是同一时间步上两个 4 维 state 的距离。常看的统计量是 mean-step、最后一步、每条轨迹最大偏差的均值，以及 max-p95。

## Saliency

主线脚本：

```text
saliency_map/scripts/precompute_saliency_maps.py
```

当前实现的方法是 occlusion：遮挡图片上的 patch，看 controller action 变化有多大。默认输出：

```text
datasets/cartpole/data/dataset_v1/saliency_occlusion.npz
```

诊断脚本先保留：

```text
saliency_map/scripts/diagnostics/preview_cartpole_render.py
saliency_map/scripts/diagnostics/compare_heatmap_methods.py
saliency_map/scripts/diagnostics/compare_alpha2_recon.py
```

诊断图片默认放在：

```text
saliency_map/output/diagnostics/previews/
```

这些图帮助确认 renderer、heatmap 和重建图有没有明显问题，不是主线 pipeline 必跑项。

## StarV 验证

`verify.py` 读取 `config/starv_verification/<env>.json`，把 grid 切成 cell，用 MPI 分发，每个 cell 做 reachability。

CartPole 当前配置：

```text
config/starv_verification/cartpole.json
```

当前 `num_steps = 30`。grid 里的 `start/stop/num` 不是采样点，而是连续区间 cell。比如 `num=60` 表示把 `[start, stop]` 切成 60 个小区间。

`starv_verification/` 下的文件和根目录同名，但作用不同：

- `starv_verification/model.py`：把根目录的 `Decoder` / `Controller` 换成 StarV reachability 版本；
- `starv_verification/dynamic.py`：把 tensor dynamics 换成 bound / reachable-set 版本；
- `starv_verification/verifiers.py`：定义每个环境的安全条件和逐步验证逻辑。

CartPole verifier 有一个特殊点：模型 reachability 只把 `[position, angle]` 喂给 decoder，因为根目录 `Decoder` 的输入就是这两维；完整 dynamics 仍然维护 4 维 CartPole state。

验证结果默认写到：

```text
results/cartpole/safety_result.json
```

`tools/visualize.py` 可以把 `safety_result.json` 画成红绿 safety map；这一步只是看结果，不改变验证产物。

## 轨迹 vs reachable tube

`compare.py` 是 `verify.py` 之后用的脚本。它读取：

```text
safety_result.json
real_trajectories.npz
dwm_trajectories_<variant>.npz
```

然后检查真实轨迹和 DWM 轨迹是否落在对应 cell 的 reachable tube 里。它会输出两张图：

```text
real_vs_reachable_tube_examples.png
dwm_vs_reachable_tube_examples.png
```

CartPole 默认画/check 的维度是 position 和 velocity，也就是 `0 1`。如果要看 position-angle 这类平面，再改 `plot_dims` / `check_dims`。注意 `compare.py` 里 cartpole 有一套历史默认路径，和现在 `datasets/cartpole/data/dataset_v1/` 的主线布局不完全一致；正式对结果时最好显式指定 `safety`、`real`、`dwm` 和 `states` 路径。

## 文件说明

### 根目录脚本

| 文件 | 说明 |
|---|---|
| `make_decoder_dataset.py` | 生成 `states.npz` 和 `real_trajectories.npz` |
| `saliency_map/scripts/precompute_saliency_maps.py` | 生成 `saliency_occlusion.npz` |
| `train_decoder.py` | 训练 decoder，写入 `decoder_last.pth` / `decoder_best_*.pth` / `metrics.json` |
| `sampling.py` | 生成 `transition_dataset.npz` 和 `dwm_trajectories_<variant>.npz` |
| `verify.py` | StarV/MPI 验证入口 |
| `compare.py` | 轨迹和 reachable tube 的包含性对比 |
| `tools/visualize.py` | 把验证结果画成红绿 safety map |

### 核心模块

| 文件 | 说明 |
|---|---|
| `model.py` | PyTorch 版 `Controller` / `Decoder` |
| `dynamic.py` | PyTorch 版 CartPole / MountainCar / Pendulum dynamics 和 renderer wrapper |
| `env.py` | CartPole 连续动作环境，主要供 renderer 使用 |
| `utils.py` | config、seed、device、uniform state 采样、批量 render 等小工具 |

### notebook 和记录

- `notebooks/generate_dataset.ipynb`：封了 `run_make_decoder_dataset(env_name)` 和 `run_sampling(env_name, decoder_variant)`，现在用于整理数据生成和 rollout 的几个 case。
- `notebooks/train_decoder.ipynb`：封了 `run_train(env_name, weight_mode, alpha=None, seed=None, output_dir=None)`，也放了 CartPole rollout 和轨迹偏差统计。
- `report/`：每日实验记录。当前 CartPole 的 $\alpha$、$\lambda_{ctrl}$、rollout 决策都在 2026-07-08 的记录里。
- `explore.ipynb`：早期探索记录，主要逻辑已经迁到 `compare.py`。

## 目前还要注意的事

- CartPole 的 `state_space` 还没和 StarV grid 完全理顺。后面如果正式进入 StarV 对比，最好先重新确认 sampling 初始范围和 decoder 训练覆盖范围。
- `metadata.json` 会被 `make_decoder_dataset.py` 和 `sampling.py` 后跑的一方覆盖，追实验时最好同时看 config 和文件时间。
- `sampling.py` 里的 `transition_dataset.npz` 和 DWM rollout 是同一个入口生成的，但如果只是更新 decoder 权重，很多时候只需要重新生成 `dwm_trajectories_<variant>.npz`，不一定要重算 transition 数据。
- StarV 配置里的 decoder 权重路径要和你想验证的 decoder 对上。现在配置里如果还指向共享目录的老权重，就不是在验证 `dwm_weight/now_weight/cartpole/saliency/` 里的新 decoder。
