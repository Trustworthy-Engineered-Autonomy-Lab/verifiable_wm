
## 环境启动
conda activate /home/tealab_shared/starv/env/starv_shared

## 目录结构


```
verifiable_wm/
├── config/                  # 各脚本用的 json 配置
├── datasets/                # 生成的数据集（按环境/版本存放）
├── dwm_weight/               # world model 权重（老权重 / 当前权重）
├── dynamic.py                # 可微分环境动力学（pytorch tensor 版本）
├── env.py                    # 老版本连续动作 CartPole 环境（供渲染用）
├── make_decoder_dataset.py   # real renderer pipeline：生成 decoder 训练数据和真实闭环轨迹
├── model.py                  # Controller / Decoder 网络结构
├── notebooks/                # 实验用 ipynb（generate_dataset：造 6 组数据集；train_decoder：训三组 decoder + rollout 对比）
├── readme.md
├── saliency_map/             # saliency/heatmap 脚本与临时可视化输出
├── sampling.py                # sampling / DWM pipeline：生成 transition 数据和 DWM 闭环轨迹
├── starv_verification/        # 基于 StarV（区间/星集）的安全验证部分
├── tools/                     # 可视化工具（红绿安全图）
├── utils.py                   # 公共小工具
└── verify.py                  # 验证入口脚本（mpi4py 并行跑 grid）
```

## 各文件 / 目录详细说明

### `config/`

存放各脚本需要的 json 配置，按脚本名分子文件夹，每个子文件夹下按环境（cartpole / mountain_car / pendulum）分文件：

- `config/make_decoder_dataset/` — `make_decoder_dataset.py` 用的配置，控制 real renderer pipeline 怎么采样、怎么生成 decoder 训练数据和 real rollout ground truth
- `config/sampling/` — `sampling.py` 用的配置，控制 transition 数据怎么采样，以及 DWM rollout 用哪个 decoder（主线只比较 intensity / saliency）
- `config/train_decoder/<环境>/` — 训练 decoder 的配置，按环境分子文件夹，每个环境下三份文件对应三组 loss（`intensity.json` = 论文 baseline、`saliency.json` = 我们的方法、`hybrid.json` = 诊断组），三份只差 `weight_mode` 和 `output_dir` 两行；mountain_car / pendulum 是预留位
- `config/starv_verification/` — 跑安全验证的配置（grid 划分、verifier 选择等）

### `datasets/`

数据集生成结果，按 `环境/data/版本号/` 存放，比如 `datasets/cartpole/data/dataset_v1/`。每个版本目录下包含 `states.npz`、`real_trajectories.npz`、`metadata.json` 等文件（具体字段见下方「生成的数据集文件说明」）。

### `dwm_weight/`

存放 world model 相关权重：

- `raw_weight/` — 各环境（cartpole / mountain_car / pendulum）的老 controller + decoder 权重
- `now_weight/` — 当前正在训练的新权重

### `dynamic.py`

`DynamicModel` 是抽象基类，规定了 `step`（状态转移）和 `render`（渲染图片）两个接口。下面 `CartPole` / `MountainCar` / `Pendulum` 三个类是具体实现：

- CartPole 的物理公式是手写的
- MountainCar 和 Pendulum 是照抄 gym 环境里的公式改成 tensor 版本
- `render` 用的是 gym 画图逻辑，不是 tensor 化的，只是用来生成训练图片

### `env.py`

老版本连续动作 CartPole 环境（`ContinuousCartPoleEnv`），`dynamic.py` 里 `CartPole` 类的 `render` 靠这个来画图。MountainCar 和 Pendulum 不需要这个文件，直接用 gym 自带的 env 来 render。

### `notebooks/`

实验用 notebook，从 `verifiable_wm/` 或 `notebooks/` 目录启动 Jupyter 均可（首个 cell 自动定位仓库根目录）：

- `generate_dataset.ipynb` — 一键跑三个环境的 `make_decoder_dataset` 和 `sampling`（共 6 个 case）；sampling 的 cell 里用 `decoder_variant` 选闭环用哪个 decoder。
- `train_decoder.ipynb` — 训练三组 decoder（intensity / saliency / hybrid）、汇总 test 指标、α 扫描模板、生成 WM 闭环轨迹并计算 rollout 偏差对比。

### `make_decoder_dataset.py`

这个脚本负责 real renderer pipeline。运行一次会生成两类产物：

- `states.npz` — decoder 训练数据，即真实 renderer 给出的 `(state, image)` 对。
- `real_trajectories.npz` — 真实 renderer + controller + dynamics 跑出的闭环轨迹，后面算 DWM rollout 偏差时拿它当 ground truth。

训练 decoder 时主要用 `states.npz`；后面做闭环对比时再用 `real_trajectories.npz`。这两份数据都来自同一条 real renderer pipeline，所以放在一个脚本里一起生成。

### `model.py`

两个网络：

- `Controller` — 卷积网络，输入图片输出一个动作
- `Decoder` — 输入 state 输出图片，也就是 wm

训练脚本和验证脚本用的都是这两个类的结构，`starv_verification` 里会继承过去改写 `forward`，换成用区间去算。

### `saliency_map/`

用来放置不同 study case 的 controller saliency / heatmap 相关脚本和临时输出。当前约定：

- `saliency_map/scripts/` — 放主线 saliency map 计算脚本。
- `saliency_map/scripts/diagnostics/` — 放临时 preview、方法可视化检查脚本。
- `saliency_map/output/diagnostics/previews/` — 放临时图片，靠文件名区分 study case 和用途。
- 训练用的 `.npz` 放回对应的 `datasets/<env>/data/<version>/`，不要放在 `saliency_map/output/`。

主方法先按 study case 检查后再定，不强行要求所有环境都用同一个 saliency 算法。

### `sampling.py`

这个脚本负责 sampling / DWM rollout。运行一次会加载 controller、dynamic 和指定 decoder，然后生成两类产物：

- `transition_dataset.npz` — 用真实 renderer 生成图像，让 controller 选 action，再由 dynamics 得到下一步 state，最后把 rollout 展平成 `(s, a, s')`。
- `dwm_trajectories_<decoder>.npz` — 用 decoder 生成图像，让 controller 选 action，再由 dynamics 往前推，保留完整 DWM 闭环轨迹。

可以理解成：`transition_dataset.npz` 是按单步转移摊平的数据，`dwm_trajectories_<decoder>.npz` 是保留时间顺序的闭环轨迹。后者就是拿来和 `real_trajectories.npz` 逐步比较 rollout 偏差的。

### `starv_verification/`

用于后续验证 ground truth 的安全验证模块。

- **`dynamic.py` / `model.py`** — 文件名和根目录一样，但内容是基于 StarV（区间/星集）重写的版本，直接继承根目录 `dynamic.py` / `model.py` 的类，再重载 `step` / `forward`，把原来 batch tensor 的计算换成对一个区间（bound）做 reachability 分析，这样就能证明"某个范围内的所有 state 都安全"而不是只测单个点。
- **`verifiers.py`** — `PendulumVerifier` / `MountainCarVerifier` / `CartpoleVerifier` 三个具体验证器，每一步把当前 state bound 喂给 model 算出 action bound，再喂给 dynamic 算下一步的 state bound，循环 `num_steps` 步，每步检查是否满足 safe 条件（比如 pendulum 要求角度在阈值内），满足就提前退出。

### `tools/`

- **`visualize.py`** — 读 `verify.py` 跑出来的结果 json，把 grid 里每个格子的 safe/unsafe 画成红绿两色的热力图（safety map），支持存图或者存成 npy 矩阵。

### `utils.py`

公共工具：

- `load_config` — 读 json
- `set_seed` — 随机种子
- `resolve_device` — 选 cpu 还是 cuda
- `sample_uniform_states` — 按 state_space 的上下界采样
- `render_images` — 批量渲染 state 对应的图片
- `to_numpy` — 转 numpy

`make_decoder_dataset.py` 和 `sampling.py` 都是从这里导入用的。

### `verify.py`

真正跑验证用的入口，用 `mpi4py` 把整个 state 空间切成网格（grid），每个进程分到一部分格子，每个格子用 `starv_verification` 里的 `Verifier` 去验证安全还是不安全，最后把结果存成 json。跑这个之前得先有决策/动力学模型，还有一个 json 配置文件描述 grid 和用哪个 verifier。

---

## 生成的数据集文件说明

数据集生成完之后会得到几个 npz 文件（存放在 `datasets/<env>/data/<version>/` 下）：

| 文件 | 生成脚本 | 内容 |
|---|---|---|
| `states.npz` | `make_decoder_dataset.py` | decoder 的训练数据。`states` 是精简过的 2 维 state（cartpole 用 `decoder_state_indices=[0,2]` 从 4 维里挑的），`images` 是真实渲染器渲出来的图，一一对应，训练时直接拿 `decoder(states)` 去拟合 `images`。 |
| `real_trajectories.npz` | `make_decoder_dataset.py` | 真实渲染器 + 训练好的 controller 跑 30 步闭环得到的完整轨迹，31 = 起点 + 30 步。这个算标准答案，留着后面跟 decoder 驱动出来的轨迹做对比用。 |
| `transition_dataset.npz` | `sampling.py` | 单步转移 `(s, a, s')`，48000 = 1600 条初始状态 x 30 步展平的，不保留轨迹顺序，是训练转移/动力学模型用的数据，跟训练 decoder 没关系。 |
| `dwm_trajectories_<decoder>.npz` | `sampling.py` / rollout 评估 | 形状跟 `real_trajectories` 对齐，但每一步的图是 decoder 生成的而不是真实渲染器，再喂给 controller 走出来的轨迹。存在的意义就是拿去跟 `real_trajectories` 逐条对比，差异越小说明 decoder 这个 wm 学得越像真实环境，这也是 verifiable_wm 的核心验证信号，后面应该会喂给 `starv_verification` 用。**主线命名按生成轨迹用的 decoder 区分**：`_intensity` = 论文 baseline，`_saliency` = 当前方法。`_old` 和 `_hybrid` 只作为历史/诊断记录，不再进入后续主线对比。 |
| `metadata.json` | 两者都会写 | 两个脚本各自把收到的 config 原样 dump 进去，方便回头查这批数据是什么参数生成的。**注意**：现在两个脚本 `output_dir` 是同一个，`metadata.json` 会被后跑的那个覆盖掉，先不管，以后再改。 |

### npz 文件 key

`{split}` 指 `train` / `val` / `test`，每个文件里三个 split 各存一份。

**`states.npz`**（6 个 key）

| key | shape | 说明 |
|---|---|---|
| `{split}_states` | `(N, 2)` | 精简过的 2 维 state（cartpole 用 `decoder_state_indices=[0,2]` 从 4 维里挑的） |
| `{split}_images` | `(N, 1, 96, 96)` | 对应的渲染图 |

**`real_trajectories.npz`**（6 个 key）

| key | shape | 说明 |
|---|---|---|
| `{split}_traj` | `(N, 31, state_dim)` | 31 = 起点 + 30 步闭环轨迹（cartpole state_dim=4） |
| `{split}_actions` | `(N, 30, action_dim)` | 对应每一步的动作 |

**`transition_dataset.npz`**（6 个 key，`sampling.py` 生成）

| key | shape | 说明 |
|---|---|---|
| `{split}_states` | `(N, state_dim)` | 单步转移起点 state |
| `{split}_actions` | `(N, action_dim)` | 该步动作 |
| `{split}_next_states` | `(N, state_dim)` | 单步转移后的 state |

**`dwm_trajectories_<decoder>.npz`**（6 个 key；主线只继续生成 / 比较 `_intensity` 和 `_saliency`，已有 `_old` / `_hybrid` 仅保留作历史记录）

| key | shape | 说明 |
|---|---|---|
| `{split}_traj` | `(N, 31, state_dim)` | 跟 `real_trajectories` 里的 `traj` 对齐，但每步图片是 decoder 生成的而不是真实渲染器 |
| `{split}_actions` | `(N, 30, action_dim)` | 对应每一步的动作 |
