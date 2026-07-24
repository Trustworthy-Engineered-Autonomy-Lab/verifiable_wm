# 逐 Cell 轨迹采样 Reachable Tube 设计

日期：2026-07-24

状态：设计已确认，等待用户审阅文档

## 1. 背景

当前项目包含两条彼此独立的数据链：

1. `make_decoder_dataset.py` 根据 StarV 配置中的整个 grid 外包 box，全局均匀采样固定数量的初始状态，生成 `starv_states.npz` 和真实闭环 trajectory。
2. `verify.py` 根据 `grid.dims[*].num` 将初始集合划分为小 cell，并对每个完整 cell 做 StarV 集合传播，生成形式化 reachable tube。

当前的 `sampling.py` 本身不生成初始状态；它读取 `starv_states.npz`，再运行

```text
decoder / G_MLP -> controller -> analytic dynamics
```

得到 DWM 或 CGAN trajectory。

因此，改变随机 trajectory 的初始点不会改变 StarV reachable tube。StarV tube 的输入是完整 cell 边界，而不是有限个采样点。CGAN 需要单独的 StarV 配置，是因为网络结构、权重及 latent 区间改变了形式化集合传播，而不是因为初始状态的采样方式不同。

本功能新增第三条数据链：在每个 StarV 小 cell 内采样 3 个初始状态，运行 DWM 或 CGAN 闭环 trajectory，并对同一 cell 的 3 条 trajectory 逐时间步取包络，构造独立的经验 sampled reachable tube。

该 tube：

- 通常比对应的 StarV formal tube 窄；
- 依赖实际采到的初始状态及 CGAN latent；
- 使用固定 seed，因此随机但可复现；
- 只描述有限 trajectory 的经验包络，不提供对 cell 内所有未采样状态的形式化覆盖保证。

## 2. 目标

本功能必须：

1. 支持 CartPole、MountainCar、Pendulum 和 Brake 四个环境。
2. 同时支持当前确定性 DWM `Decoder` 和随机 CGAN `G_MLP`。
3. 使用与 StarV 完全一致的 grid、cell 数量、cell 大小和线性顺序。
4. 在每个 cell 内均匀随机采样 3 个初始状态。
5. 使用固定主 seed `2025`。
6. 让同一环境的 DWM 与 CGAN 使用完全相同的 3 个初始状态。
7. 对 CGAN 的每个初始状态只运行一次 trajectory；每个时间步使用一组显式、可复现的 latent。
8. 分别保存 DWM 和 CGAN 的 construction trajectories 及 sampled tube。
9. 让 sampled tube JSON 可以被现有 `compare.py` 和 `signed_tube_margin.py` 消费。
10. 保持现有 sampling、StarV、conformal、ablation 和 real/DWM trajectory 配对流程不变。

## 3. 非目标

本功能不负责：

- 用 3 条 trajectory 替代 StarV 的形式化验证；
- 声称 sampled tube 覆盖 cell 内所有初始状态；
- 覆盖现有 `starv_states.npz`、`real_trajectories.npz` 或 `dwm_trajectories_<variant>.npz`；
- 将每-cell 3 点改造成新的 train/val/test 数据划分；
- 对 sampled tube 做 conformal inflation；
- 在自动测试中生成八套完整规模的生产 tube；
- 为 CGAN 的同一个初始状态重复采样多组 latent trajectory。

未来若需要 sampled tube 的 conformal inflation，应作为独立功能设计，继续保持 construction、calibration 和 test 数据分离。

## 4. 已选方案

在 `sampling.py` 中增加并列的 `cellwise_tube` 模式，同时保留当前默认模式：

```text
sampling_mode = "state_splits"    # 当前默认流程
sampling_mode = "cellwise_tube"   # 新增流程
```

未设置 `sampling_mode` 的现有配置继续进入当前 `generate_dataset()` 路径，其行为和输出命名不得改变。

新模式由 `sampling.py` 负责配置解析、模型加载、controller/dynamics 构造和分批 rollout。新增 `sampled_tube.py`，集中实现以下纯数据逻辑：

- StarV grid 到有序 cell bounds 的转换；
- 每-cell 初始状态采样及共享 states 文件校验；
- CGAN 显式 latent 生成；
- trajectory 到 sampled tube 的聚合；
- Pendulum 周期角度区间处理；
- sampled safety result 判定；
- trajectory NPZ 和 tube JSON 的保存与校验。

这一边界让 `sampling.py` 继续作为统一运行入口，同时避免将 grid、tube 和 JSON 细节全部堆入现有文件。

## 5. 总体数据流

```text
config/sampled_tube/<env>[_g_mlp].json
        |
        +--> base_sampling_config
        |       |
        |       +--> controller / decoder / dynamic
        |       +--> model-specific StarV config（模型及 verifier 溯源）
        |
        +--> grid_config（DWM/CGAN 共享的规范 grid）
                |
                +--> ordered cell bounds
                |
                +--> 每 cell 3 个固定-seed初始状态
                        |
                        +--> shared cellwise_states.npz
                        |
                        +--> DWM rollout
                        |       +--> sampled_trajectories_<dwm-variant>.npz
                        |       +--> sampled_reachable_tube_<dwm-variant>.json
                        |
                        +--> CGAN rollout + explicit latent
                                +--> sampled_trajectories_g_mlp.npz
                                +--> sampled_reachable_tube_g_mlp.json
```

construction trajectory 只用于构造 tube。最终 containment 评价继续使用独立的现有 `val_traj`/`test_traj`，不能使用每-cell 3 条 construction trajectory 评价其自身构造的 tube。

## 6. 配置设计

新增以下八个轻量 wrapper 配置：

```text
config/sampled_tube/
├── cartpole.json
├── cartpole_g_mlp.json
├── mountain_car.json
├── mountain_car_g_mlp.json
├── pendulum.json
├── pendulum_g_mlp.json
├── brake_system.json
└── brake_system_g_mlp.json
```

wrapper 配置引用现有 sampling config，避免重复 controller、decoder、dynamic 和权重路径。DWM 示例：

```json
{
  "sampling_mode": "cellwise_tube",
  "base_sampling_config": "config/sampling/cartpole.json",
  "grid_config": "config/starv_verification/cartpole.json",
  "model_id": "saliency",
  "samples_per_cell": 3,
  "seed": 2025,
  "cell_batch_size": 128,
  "states_file": "datasets/cartpole/sampled_tube/seed_2025/cellwise_states.npz",
  "trajectory_file": "datasets/cartpole/sampled_tube/seed_2025/sampled_trajectories_saliency.npz",
  "tube_file": "results/cartpole/sampled_tube/sampled_reachable_tube_saliency_seed_2025.json"
}
```

对应的 CGAN wrapper：

- `base_sampling_config` 指向 `config/sampling/cartpole_g_mlp.json`；
- `grid_config` 仍指向同一个 `config/starv_verification/cartpole.json`；
- `model_id` 为 `g_mlp`；
- 与 DWM wrapper 使用完全相同的 `states_file`；
- 使用独立的 `trajectory_file` 和 `tube_file`。

四个环境遵循相同规则。wrapper 声明的 `model_id` 必须与基础 sampling config 中解析到的 decoder variant 一致，否则运行失败。当前三个 Gym 环境的确定性配置解析为 `saliency`，Brake 的确定性配置未显式声明 variant，因此沿用现有 `decoder_variant()` 语义解析为 `old`；四个 CGAN 配置均解析为 `g_mlp`。

同一环境的两个 wrapper 必须引用同一个非 `_g_mlp` StarV 配置作为规范 `grid_config`。基础 sampling config 自带的 `starv_config` 仍用于模型层、verifier 参数和结果溯源；运行前必须验证其 `grid` 和 `verifier.kwargs.num_steps` 与规范 `grid_config` 完全一致。这样共享 states 文件的身份不依赖先运行 DWM 还是 CGAN，也不会因两个模型使用不同的 StarV 配置路径而产生伪冲突。

rollout 前还必须完成模型语义校验：

1. 基础 sampling config 的 decoder 类型必须对应 model-specific StarV config 的 `Decoder` 或 `G_MLP` 层。
2. 实际解析后的模型 checkpoint 路径、`z_range`（CGAN）、controller checkpoint 和 controller 激活必须一致；比较时先相对项目根目录规范化路径，并解析各自默认参数。
3. `sampling.rollout_steps`、model-specific `verifier.kwargs.num_steps` 和规范 `grid_config` 的 `verifier.kwargs.num_steps` 必须三者相等。
4. model-specific 与规范 `grid_config` 的完整 verifier `name`、`args` 和 `kwargs` 必须一致，保证 DWM/CGAN 的 `cells[].result` 使用同一安全任务。

DWM 的输出激活始终记录实际 sampling 值，但不统一要求它与 StarV 的有效传播算子一致：Brake DWM 的 clamp 必须与 StarV SatLin 对齐，`G_MLP` 的 clamp 也必须与 StarV SatLin 对齐；三个 Gym DWM 已知是 pointwise sigmoid、StarV SatLin，不将这一已知差异作为 empirical tube construction 的失败条件，并记录 `formal_semantics_match=false`。

sampled 产物中的显式 provenance 字段以实际加载的 sampling config 为准，其中顶层 `decoder_output_activation` 是 pointwise rollout 激活的权威来源。只有上述校验全部通过后，才允许复制 model-specific StarV config 的 `layers` 和 `verifier` 到 sampled JSON；`layers` 只为现有 `grid/cells` 消费者保留结构兼容和 checkpoint 来源，不表示 Gym DWM pointwise 与 formal StarV 激活语义完全等价。其他配置漂移会使 JSON 错误描述实际生成 tube 的模型，必须直接失败。

命令入口保持统一：

```bash
python sampling.py config/sampled_tube/cartpole.json
python sampling.py config/sampled_tube/cartpole_g_mlp.json
```

## 7. Cell 生成与顺序

cell 生成必须与 `verify.py` 保持一致：

1. 对每个维度计算 `np.linspace(start, stop, num + 1)`。
2. 取相邻 edge 作为该维度的 cell 起止边界。
3. 使用 `np.meshgrid(..., indexing="ij")`。
4. 按当前 NumPy C-order 展平为线性 cell 顺序。

cell 数量为各维 `num` 的乘积：

| 环境 | Grid | Cell 数 | 每 cell 3 点后的 trajectory 数/模型 |
|---|---:|---:|---:|
| CartPole | `60 × 1 × 60 × 1` | 3,600 | 10,800 |
| MountainCar | `80 × 80` | 6,400 | 19,200 |
| Pendulum | `100 × 50` | 5,000 | 15,000 |
| Brake | `40 × 40` | 1,600 | 4,800 |

两个模型合计生成 99,600 条 construction trajectory。

grid fingerprint 由规范 `grid_config` 中维度名称、`start`、`stop` 和 `num` 的规范化表示计算。共享 states 文件同时保存这份规范 grid 的 JSON 快照。DWM 与 CGAN 的 grid fingerprint、state dimension 和 horizon 必须一致，才能共享 states 文件并进行公平比较。

## 8. 初始状态采样

每个 cell 独立创建确定性随机子流。实现使用整数 stream id 区分用途：

```text
SeedSequence([主 seed, 0, cell_index])    # 0 表示 state stream
```

其语义等价于为每个 `cell_index` 派生独立 seed，但避免 state 与 latent 随机流互相影响。修改 batch size、模型运行顺序或先运行 DWM/CGAN 中的任意一个，都不会改变某个 cell 的初始状态。

对每个状态维度：

```text
sample = lower + (upper - lower) * U[0, 1)
```

零宽度维度自然保持固定值。生成完成后统一转换为 `float32`，输出 shape 为：

```text
(num_cells, 3, state_dim)
```

首次运行在 `states_file` 不存在时生成并保存。后续运行只复用，不重新采样。若已存在文件的 seed、grid fingerprint、shape 或 `samples_per_cell` 不匹配，则明确失败，不静默覆盖。

## 9. DWM 与 CGAN Rollout

### 9.1 DWM

确定性 DWM 对每个初始状态运行一次：

```text
Decoder(state_t) -> Controller(image_t) -> Dynamic.step(state_t, action_t)
```

输出：

```text
trajectories: (num_cells, 3, T+1, state_dim)
actions:      (num_cells, 3, T,   action_dim)
```

CartPole 继续只将配置的 `decoder_state_indices=[0, 2]` 传给 decoder；完整 4 维状态用于 dynamics、trajectory、cell 定位和 tube。

### 9.2 CGAN

CGAN 对每个初始状态也只运行一次 trajectory。与当前 `G_MLP.forward(state, z=None)` 内部临时采样不同，cellwise 模式预先生成并保存显式 latent：

```text
latents: (num_cells, 3, T, latent_dim)
```

latent 使用独立的整数 stream id：

```text
SeedSequence([主 seed, 1, cell_index])    # 1 表示 latent stream
```

每个 latent 分量均匀采自对应配置的：

```text
[-z_range, z_range]
```

latent 在生成时即转换为 `float32`。rollout 使用的必须是这份可保存数组中的精确数值，并将同一数组原样写入 construction NPZ；不得在保存前或 rollout 时再次随机生成。rollout 每一步显式调用 `G_MLP(state_t, z_t)`。这样 states 和 latent 都不依赖 Torch 全局 RNG，也不依赖 cell batch size。不同硬件或矩阵计算内核仍可能产生浮点末位差异，因此验收使用数值容差，而不要求跨设备逐 bit 相同。

## 10. Tube 聚合

对每个 cell 独立聚合：

```text
t = 0:
    bounds[0] = 完整 cell bounds

t = 1..T:
    lower[t, dim] = min(三条 trajectory 在该时刻、该维的状态)
    upper[t, dim] = max(三条 trajectory 在该时刻、该维的状态)
```

`t=0` 必须使用完整 cell，而不是三个初始点的 min/max。现有 `compare.py` 会用 `trajectory[0]` 和 `cells[i].bounds[0]` 复核 cell；若使用 sampled min/max，独立 test trajectory 很可能无法定位到正确 cell。

### 10.1 Pendulum 周期角度

Pendulum 的 `theta` 被规范化到 `[-π, π)`。若三个样本跨越 `-π/π`，直接取普通 min/max 会错误地产生接近整个圆的宽区间。

对 `theta`：

1. 将三个角度排序；
2. 计算圆周上的三个间隙；
3. 找到最大间隙；
4. 取其补集作为包含全部样本的最短圆弧；
5. 若最短圆弧跨越 `-π/π`，在 JSON 中编码为两个区间。

JSON 沿用现有下游的扁平 low/high pair 协议：

```text
普通区间：  [low, high]
跨边界圆弧：[arc_start, π, -π, arc_end]
```

第二种表示等价于 `[arc_start, π] ∪ [-π, arc_end]`。现有 `compare.py` 和 `signed_tube_margin.py` 都按相邻元素解析区间对，因此该表示与下游兼容。其他状态维度继续使用普通 min/max。

## 11. Sampled Safety Result

每个 cell 的 `result` 沿用对应 StarV verifier 的当前任务语义，但结论只针对这三条 sampled trajectory 的包络：

- CartPole：最终时刻的 angle 区间全部位于 `[-goal_angle_threshold, goal_angle_threshold]`。
- MountainCar：最终时刻的 position 下界不小于 `goal_position_threshold`。
- Pendulum：最终时刻的所有 theta 区间全部位于 `[-goal_angle_threshold, goal_angle_threshold]`。
- Brake：整个 horizon 中所有 sampled trajectory 的 distance 都严格大于 0。

JSON 必须明确记录：

```text
guarantee_type = "empirical_only"
```

不得将 sampled `result=true` 描述为对完整 cell 的形式化认证。

## 12. 输出文件

### 12.1 共享初始状态

```text
datasets/<env>/sampled_tube/seed_2025/cellwise_states.npz
```

字段：

```text
states               (num_cells, 3, state_dim)
cell_bounds          (num_cells, state_dim, 2)
cell_indices         (num_cells,)
seed
samples_per_cell
grid_fingerprint
grid_json
state_dim
horizon
source_grid_config
```

### 12.2 模型 construction trajectories

```text
datasets/<env>/sampled_tube/seed_2025/sampled_trajectories_<variant>.npz
```

字段：

```text
trajectories         (num_cells, 3, T+1, state_dim)
actions              (num_cells, 3, T, action_dim)
latents              仅 CGAN；(num_cells, 3, T, latent_dim)
cell_indices
seed
samples_per_cell
grid_fingerprint
rollout_steps
model_id
model_type
decoder_weights
decoder_output_activation  # 仅 DWM
formal_semantics_match
z_range                    # 仅 CGAN
controller_name
controller_weights
controller_activation
dynamic_name
dynamic_args_json
source_sampling_config
source_grid_config
source_starv_config
source_states_file
```

construction 文件不使用 `train_traj`、`val_traj` 或 `test_traj` 键，避免被误当成独立评价数据。

### 12.3 Sampled tube JSON

```text
results/<env>/sampled_tube/sampled_reachable_tube_<variant>_seed_2025.json
```

顶层保留现有工具需要的：

```text
layers
verifier
grid
cells
```

每个 cell 包含：

```text
cell_index
bounds          # T+1 个时间步
result
```

同时增加：

```text
method = "cellwise_sampled_trajectory_envelope"
guarantee_type = "empirical_only"
model_id
model_type
samples_per_cell
seed
horizon
state_dim
source_sampling_config
source_grid_config
source_starv_config
source_states
source_trajectories
grid_fingerprint
decoder_weights
decoder_output_activation  # 仅 DWM
formal_semantics_match
controller_name
controller_weights
controller_activation
dynamic_name
dynamic_args
z_range                    # 仅 CGAN
```

`layers` 和 `verifier` 从基础 sampling config 指向的 model-specific StarV config 复制；`grid` 从 wrapper 的规范 `grid_config` 复制。两者已在运行前通过 grid/horizon 一致性校验，因此 sampled JSON 可以作为现有 `--safety` 输入，同时保留模型来源和共享 cell 身份。

## 13. 下游连接

construction 数据与评价数据严格分离：

```text
每 cell 3 条 trajectory
        |
        +--> 只构造 sampled tube

现有独立 real test_traj
+ 对应模型的全局 test_traj
+ sampled tube JSON
        |
        +--> compare.py / signed_tube_margin.py
```

典型评价：

```bash
python compare.py \
  --env cartpole \
  --safety results/cartpole/sampled_tube/sampled_reachable_tube_saliency_seed_2025.json \
  --real datasets/cartpole/data/dataset_v1/real_trajectories.npz \
  --dwm datasets/cartpole/data/dataset_v1/dwm_trajectories_saliency.npz \
  --outdir results/cartpole/compare_sampled_tube_saliency
```

CGAN 使用相同 real test trajectories，以及从相同全局 `starv_states.npz` 出发的 `dwm_trajectories_g_mlp.npz`。

`compare.py` 需要做范围内的兼容增强：

1. sampled tube JSON 继续通过现有 `grid/cells` loader 读取。
2. provenance 校验同时识别 `layers.Decoder` 和 `layers.G_MLP`。
3. 对 sampled tube 检查 tube 与评价 trajectory 的模型类型/权重、controller、dynamic、grid 和 horizon 一致；真实 trajectory 不含 decoder，只校验 controller、dynamic、grid 和 horizon。
4. 默认最终报告使用 `test_traj`；`val_traj` 只可用于开发诊断。

严格 provenance 是 sampled tube 新流程的要求：若 `method` 为
`cellwise_sampled_trajectory_envelope`，评价 trajectory 缺少必要 metadata 时应提示用户用当前
sampling 配置重新生成。对不含该 `method` 的历史 StarV JSON，`compare.py` 继续保持当前的
向后兼容行为，只校验实际存在的 metadata，避免破坏旧实验。

`signed_tube_margin.py` 还需在 `ENV_DIMS` 中增加
`"brake_system": (0, 1)`，使 Brake 与其他三个环境一样可以直接评价；其现有扁平
low/high pair 解析逻辑继续复用。该脚本的 `--env` choices 直接来自 `ENV_DIMS`，因此新增
映射后 CLI 也同步支持 Brake。

对带 `method="cellwise_sampled_trajectory_envelope"` 的 JSON，
`signed_tube_margin.py` 必须执行与 `compare.py` 相同的严格 provenance 规则：DWM/CGAN
trajectory 校验模型、controller、dynamic、grid 和 horizon，real trajectory 校验
controller、dynamic、grid 和 horizon；缺字段或不一致均明确失败。对历史 StarV JSON 继续保持现有宽松行为。两个工具应复用
同一套 metadata 解析/校验 helper，避免规则随维护产生分叉。

为使上述校验可落地，现有全局评价 trajectory 的保存逻辑需要增加 metadata，但不改变数组、
split、文件名或 rollout 行为：

- `sampling.py` 保存的 DWM/CGAN NPZ 增加模型类型、解析后的 checkpoint、输出激活或
  `z_range`、controller 名称/权重/激活、dynamic 名称/参数、grid fingerprint 和配置来源；
- `make_decoder_dataset.py` 保存的 real NPZ 增加 controller、dynamic、grid fingerprint、
  horizon 和配置来源；
- sampled tube 的严格评价要求这些字段；现有旧 NPZ 若缺少字段，应提示按当前配置重新生成；
- 历史 formal StarV JSON 的评价仍允许读取旧 NPZ，保持向后兼容。

`conformal.py` 不读取 reachable tube，本功能不改变其流程。

## 14. 错误处理与产物安全

以下情况必须失败并给出包含实际值和期望值的错误：

- `samples_per_cell <= 0`；
- wrapper `model_id` 与基础 sampling config 的 decoder variant 不一致；
- 基础 sampling config 的 model-specific StarV grid/horizon 与规范 `grid_config` 不一致；
- 实际 sampling 模型与 model-specific StarV 层的类型、checkpoint、`z_range` 或 controller 配置不一致；
- Brake DWM 或 G_MLP 的输出激活与 StarV 有效传播语义不一致；三个 Gym DWM 的已知 sigmoid/SatLin 差异除外；
- model-specific verifier 与规范 verifier 的名称或任务判定参数不一致；
- sampled tube 与评价 trajectory 的模型、controller、dynamic、grid 或 horizon metadata 缺失或不一致；
- DWM 与 CGAN 的 grid、state dimension 或 horizon 不一致；
- 共享 states 文件的 seed、shape、grid fingerprint 或每-cell 数量不一致；
- 任一 cell 不是恰好 3 个点；
- 初始状态不在对应 cell 内；
- trajectory/action/latent shape 不符合协议；
- trajectory、action 或 latent 含 NaN/Inf；
- CGAN latent 超出 `[-z_range, z_range]`；
- tube 的时间步数量不是 `T+1`；
- tube、评价 trajectory 和模型 provenance 不一致。

所有 NPZ/JSON 先写入目标目录中的临时文件，写入、重新加载和结构校验成功后再原子替换目标文件。中断不能留下名称正确但内容不完整的最终产物。

Brake sampled tube construction 只使用 learned renderer、controller 和解析 dynamics，不依赖 CARLA。若本地缺少真实 Brake trajectory，construction 仍可完成；只有独立真实 trajectory 评价需要预先采集的 CARLA 数据。

## 15. 性能设计

不能将某环境的全部初始状态一次送入 decoder。以 `cell_batch_size=128` 为例，每批处理：

```text
128 cells × 3 samples = 384 trajectories
```

每批完成全部 horizon rollout，再写入预分配的 trajectory/action 数组。显式保存的 states 和 latent 使随机输入不随 batch 划分变化。

命令行输出至少包括：

- 环境和模型；
- cell 总数；
- samples per cell；
- trajectory 总数；
- 当前已完成 cell 数和百分比；
- states、trajectory 和 tube 的最终路径。

本期不实现中断恢复。运行中断后重新执行；固定 per-cell 随机子流确保重新生成相同 construction inputs。

## 16. 测试设计

仓库当前忽略 `/tests`。实现阶段应恢复一个可跟踪的测试目录，并增加聚焦本功能的测试，不恢复已删除的无关历史测试。

### 16.1 单元测试

1. 四环境 cell 数量、边界、大小和顺序与 StarV 算法一致。
2. 每个 cell 恰好 3 个点，且所有点位于该 cell 内。
3. 相同 seed 生成相同 states；不同 seed 至少有一个非固定维度不同。
4. DWM 与 CGAN wrapper 复用同一个 states 文件。
5. 显式 CGAN latent 可复现且全部位于配置范围。
6. 保存的 CGAN latent 为 `float32`，rollout 使用值与重载后的保存值完全相同。
7. 改变 cell batch size 不改变 states/latent，并使 rollout 结果在浮点容差内一致。
8. `t=0` tube 等于完整 cell。
9. `t>=1` tube 等于三条测试 trajectory 的逐维包络。
10. Pendulum 跨 `-π/π` 时输出精确的 `[arc_start, π, -π, arc_end]`。
11. 四个环境的 sampled `result` 判定符合各自规则。
12. metadata、shape、grid 或 seed 不一致时明确失败。
13. sampling 与 model-specific StarV config 的模型、checkpoint、latent、controller 或 verifier 语义漂移时明确失败；Brake/G_MLP 的激活漂移也失败。
14. 三个 Gym DWM 的已知 sigmoid/SatLin 差异不会阻止 construction，且写出 `formal_semantics_match=false`。
15. JSON 能被 `compare.py` 和 `signed_tube_margin.py` 的 loader 读取。
16. `signed_tube_margin.py` 能以 `(0, 1)` 两维运行 Brake sampled tube smoke case。
17. 两个评价工具都拒绝 sampled tube 与 trajectory provenance mismatch。
18. 现有无 `sampling_mode` 配置仍走原 sampling 路径。
19. 历史 StarV JSON 继续使用宽松 metadata 兼容规则；sampled tube 使用严格 provenance 规则。

### 16.2 集成测试

使用缩小的 smoke grid 和轻量模型/动态替身分别运行：

- deterministic Decoder cellwise pipeline；
- G_MLP cellwise pipeline；
- shared states 复用；
- trajectory NPZ 保存及重载；
- sampled tube JSON 保存及重载；
- 独立 test trajectory containment。

自动测试不依赖 MPI、StarV、Gurobi、CARLA 或完整生产权重。

### 16.3 可选一致性检查

只有在 pointwise rollout 与 StarV 传播使用完全相同的 decoder 输出激活、controller、dynamic、权重、latent 范围、grid 和 horizon 时，才启用“sampled trajectory 位于 formal tube 内”的可选检查。该检查允许小的浮点容差。

当前 Gym 确定性 DWM 的 pointwise `Decoder` 默认使用 sigmoid，而 StarV `Decoder` 末层使用 SatLin/clamp，两条语义并不相同，因此该检查默认关闭，超出 formal tube 也不能据此判定 sampled pipeline 错误。Brake DWM 的 clamp 与 StarV SatLin 对齐；`G_MLP` 两侧也都使用 clamp，但仍需先通过其余 provenance 和动态语义检查后才能启用。

在满足上述前提后，若 sampled trajectory 超出 formal tube，应报告 cell、时间步、维度、sampled state 和 formal bounds，不能静默忽略。该检查是诊断项，不是生成 empirical sampled tube 的前置条件。

## 17. 完成标准

实现完成必须同时满足：

1. 八个 wrapper 配置均可解析。
2. 四个环境的 DWM/CGAN 使用完全相同的每-cell 3 个初始状态。
3. 固定 seed 可重现相同 states 和 CGAN latent。
4. 两类 construction trajectory NPZ 均符合约定 shape 和 metadata。
5. 两类 sampled tube JSON 均包含每个 cell 的完整 `T+1` bounds。
6. Pendulum 的周期角度不会因为跨边界产生虚大的普通区间。
7. `compare.py` 可以使用独立 `test_traj` 评价 DWM 和 CGAN sampled tube。
8. `signed_tube_margin.py` 可以加载并评价四个环境（含 Brake）的 sampled tube JSON。
9. metadata 不一致会失败，而不是产生不可比较的结果。
10. sampled JSON 明确记录 pointwise/formal 输出激活语义是否一致。
11. 原有 sampling、StarV、conformal 和 ablation 行为不回归。
12. 单元测试和 smoke 集成测试通过。
13. 完整生产配置存在，但自动验证阶段不要求生成八套大规模产物。

## 18. 实现影响范围

预计实现涉及：

- 修改 `sampling.py`：增加 wrapper config 解析和 `cellwise_tube` 调度，并为现有全局 DWM/CGAN NPZ 补充 additive provenance metadata；
- 修改 `make_decoder_dataset.py`：为 real trajectory NPZ 补充 additive provenance metadata；
- 新增 `sampled_tube.py`：实现 grid、sampling、latent、aggregation、判定和保存；
- 修改 `compare.py`：补充 Decoder/G_MLP 的严格 provenance 校验；
- 修改 `signed_tube_margin.py`：增加 Brake 的 `(0, 1)` 评价维度和 sampled tube 严格 provenance 校验；
- 提取两个评价工具共享的 metadata 解析/校验 helper，保持 sampled/legacy 两套规则一致；
- 新增 `config/sampled_tube/*.json` 八个配置；
- 增加本功能测试并调整 `.gitignore` 以跟踪这些测试；
- 更新 `README.md`，说明 formal StarV tube 与 empirical sampled tube 的区别及运行命令。

不修改 `verify.py` 的 StarV 算法；不改变现有 trajectory 数组、split 和文件命名协议，只增加可向后兼容读取的 provenance metadata。
