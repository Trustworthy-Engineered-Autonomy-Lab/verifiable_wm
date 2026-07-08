# compare.py 使用说明

本 README 对应当前版本的：

```bash
compare.py
```

激活环境：
```bash
conda activate /home/tealab_shared/starv/env/starv_shared
```

---

## 1. 这个脚本的作用

这个脚本用于比较：

```text
real trajectory / DWM trajectory
```

是否被 verifier 生成的：

```text
reachable tube
```


它会读取：

```text
1. safety_result.json
2. real_trajectories.npz
3. dwm_trajectories.npz
4. states.npz，可选
```

然后输出两张图：

```text
real_vs_reachable_tube_examples.png
dwm_vs_reachable_tube_examples.png
```

每张图会直接在标题中显示：

```text
fully contained: x / N
containment rate
mean step containment
check dims
max_steps
```

每个子图左下角也会显示该条 trajectory 的详细信息，例如：

```text
FULLY CONTAINED / NOT fully contained
traj idx
inside states
checked states
first out step
max violation
cell index
cell status
```

---

## 2. 支持的三种环境

当前脚本支持：

```text
mountain_car
pendulum
cartpole
```

三种环境的默认设置如下：

| 环境 | 状态含义 | 默认画图维度 `plot_dims` | 默认检查维度 `check_dims` |
|---|---|---:|---:|
| `mountain_car` | `[pos, vel]` | `0 1` | `0 1` |
| `pendulum` | `[theta, omega]` | `0 1` | `0 1` |
| `cartpole` | `[x, x_dot, theta, theta_dot]` | `0 1` | `0 1` |

对于 CartPole，当前最终设定是：

```text
只比较小车的位置和速度
```

也就是：

```text
dim 0 = x
dim 1 = x_dot
```

因此默认：

```bash
--plot-dims 0 1
--check-dims 0 1
```

---

## 3. 目录结构要求

如果你使用 `--env`，脚本会自动按下面的结构找文件：

```text
project_root/
├── compare.py
├── results/
│   ├── mountain_car/
│   │   └── safety_result.json
│   ├── pendulum/
│   │   └── safety_result.json
│   └── cartpole/
│       └── safety_result.json
└── trajectories/
    ├── mountain_car/
    │   ├── real_trajectories.npz
    │   ├── dwm_trajectories.npz
    │   └── states.npz
    ├── pendulum/
    │   ├── real_trajectories.npz
    │   ├── dwm_trajectories.npz
    │   └── states.npz
    └── cartpole/
        ├── real_trajectories.npz
        ├── dwm_trajectories.npz
        └── states.npz
```

默认输出路径是：

```text
results/<env>/compare_tube/
```

例如：

```text
results/pendulum/compare_tube/
```

---

## 4. 最常用命令

### 4.1 MountainCar

```bash
python compare.py \
  --env mountain_car \
  --split test \
  --only-in-grid
```

输出：

```text
results/mountain_car/compare_tube/real_vs_reachable_tube_examples.png
results/mountain_car/compare_tube/dwm_vs_reachable_tube_examples.png
```

---

### 4.2 Pendulum

```bash
python compare.py \
  --env pendulum \
  --split test \
  --only-in-grid
```

输出：

```text
results/pendulum/compare_tube/  
real_vs_reachable_tube_examples.png  
results/pendulum/compare_tube/  
dwm_vs_reachable_tube_examples.png  
```

---

### 4.4 CartPole

当前 CartPole 最终比较的是：

```text
x vs x_dot
```

因此可以直接运行：

```bash
python compare.py \
  --env cartpole \
  --split test \
  --only-in-grid \
  --init-source traj
```

它默认等价于：

```bash
python compare.py \
  --env cartpole \
  --split test \
  --only-in-grid \
  --init-source traj \
  --plot-dims 0 1 \
  --check-dims 0 1
```

---

## 5. `plot_dims` 和 `check_dims` 的区别

这是当前版本最重要的功能。

### 5.1 `--plot-dims`

控制图上画哪两个状态维度。

例如：

```bash
--plot-dims 0 1
```

表示图上画：

```text
state dim 0 vs state dim 1
```

对于 CartPole，就是：

```text
x vs x_dot
```

### 5.2 `--check-dims`

控制 containment 判断检查哪些维度。

例如：

```bash
--check-dims 0 1
```

表示判断 `fully contained` 时，只检查：

```text
state dim 0
state dim 1
```

对于 CartPole，就是只检查：

```text
x
x_dot
```

不会检查：

```text
theta
theta_dot
```

---

## 6. CartPole 的不同检查方式

### 6.1 只检查小车位置和速度

这是当前默认设置：

```bash
python compare.py \
  --env cartpole \
  --split test \
  --only-in-grid \
  --init-source traj \
  --plot-dims 0 1 \
  --check-dims 0 1
```

含义：

```text
图上画 x vs x_dot
containment 也只检查 x 和 x_dot
```

### 6.2 图上画小车位置和速度，但 containment 检查完整 4 维

```bash
python compare.py \
  --env cartpole \
  --split test \
  --only-in-grid \
  --init-source traj \
  --plot-dims 0 1 \
  --check-dims 0 1 2 3
```

含义：

```text
图上仍然画 x vs x_dot
但 fully contained 判断会检查 x, x_dot, theta, theta_dot 全部 4 维
```

### 6.3 图上画角度和角速度

```bash
python compare.py \
  --env cartpole \
  --split test \
  --only-in-grid \
  --init-source traj \
  --plot-dims 2 3 \
  --check-dims 2 3
```

含义：

```text
图上画 theta vs theta_dot
containment 只检查 theta 和 theta_dot
```

---

## 7. 手动路径模式

如果不想使用 `--env`，可以手动指定所有路径。

例如 MountainCar：

```bash
python compare.py \
  --safety results/mountain_car/safety_result.json \
  --real trajectories/mountain_car/real_trajectories.npz \
  --dwm trajectories/mountain_car/dwm_trajectories.npz \
  --states trajectories/mountain_car/states.npz \
  --split test \
  --outdir results/mountain_car/compare_tube \
  --only-in-grid
```

这种方式适合文件不在默认目录结构下的情况。

---

## 8. 查看 npz 文件里的 key

如果不确定 `.npz` 文件里有哪些 key，可以运行：

```bash
python compare.py \
  --env pendulum \
  --print-keys
```

它会输出类似：

```text
real keys : ['train_traj', 'val_traj', 'test_traj']
dwm keys  : ['train_traj', 'val_traj', 'test_traj']
state keys: ['train_states', 'val_states', 'test_states']
```

如果自动识别 key 失败，可以手动指定：

```bash
--real-key test_traj
--dwm-key test_traj
--state-key test_states
```

---

## 9. 使用 delta / tube inflation

默认情况下，脚本使用原始 reachable tube：

```bash
--delta 0.0
```

如果你想把 tube 上下界稍微扩大，可以使用：

```bash
--delta 0.022
```

例如：

```bash
python compare.py \
  --env mountain_car \
  --split test \
  --only-in-grid \
  --delta 0.022
```

也可以给 real 和 DWM 使用不同的 delta：

```bash
python compare.py \
  --env mountain_car \
  --split test \
  --only-in-grid \
  --real-delta 0.022 \
  --dwm-delta 0.0
```

含义：

```text
real trajectory 检查时 tube 扩大 0.022
DWM trajectory 检查时 tube 不扩大
```

---

## 10. 参数总览

| 参数 | 作用 | 示例 |
|---|---|---|
| `--env` | 指定环境并自动补全默认路径 | `--env pendulum` |
| `--safety` | 手动指定 safety result | `--safety results/pendulum/safety_result.json` |
| `--real` | 手动指定真实轨迹文件 | `--real trajectories/pendulum/real_trajectories.npz` |
| `--dwm` | 手动指定 DWM 轨迹文件 | `--dwm trajectories/pendulum/dwm_trajectories.npz` |
| `--states` | 手动指定 states 文件 | `--states trajectories/pendulum/states.npz` |
| `--split` | 选择 train / val / test | `--split test` |
| `--outdir` | 指定输出目录 | `--outdir results/pendulum/compare_tube` |
| `--plot-dims` | 指定图上画哪两个维度 | `--plot-dims 0 1` |
| `--check-dims` | 指定 containment 检查哪些维度 | `--check-dims 0 1` |
| `--max-steps` | 指定最多比较多少个 transition step | `--max-steps 20` |
| `--only-in-grid` | 只检查初始状态在 verification grid 内的轨迹 | `--only-in-grid` |
| `--init-source` | 指定初始状态来源 | `--init-source traj` |
| `--delta` | 同时设置 real 和 DWM 的 tube inflation | `--delta 0.022` |
| `--real-delta` | 只设置 real 的 tube inflation | `--real-delta 0.022` |
| `--dwm-delta` | 只设置 DWM 的 tube inflation | `--dwm-delta 0.0` |
| `--real-key` | 手动指定 real npz key | `--real-key test_traj` |
| `--dwm-key` | 手动指定 DWM npz key | `--dwm-key test_traj` |
| `--state-key` | 手动指定 states npz key | `--state-key test_states` |
| `--print-keys` | 只打印 npz keys，不画图 | `--print-keys` |

---

## 11. 输出结果怎么看

运行后，终端会显示：

```text
========== Loaded ==========
env        : pendulum
safety     : results/pendulum/safety_result.json
real       : trajectories/pendulum/real_trajectories.npz | key=test_traj | shape=(400, 31, 2)
dwm        : trajectories/pendulum/dwm_trajectories.npz | key=test_traj | shape=(400, 31, 2)
states     : trajectories/pendulum/states.npz
grid names : ['theta', 'omega']
plot dims  : (0, 1) -> theta, omega
check dims : [0, 1] -> ['theta', 'omega']
max_steps  : None
```

然后会显示 real 和 DWM 的汇总：

```text
[Real trajectory]
  checked trajectories  : 82
  fully contained       : 0/82 (0.00%)
  not fully contained   : 82/82
  mean step containment : 14.20%
  worst step containment: 3.23%
  horizon mismatch cells: 0
  max_steps             : None
  check_dims            : [0, 1]
```

重点看：

```text
fully contained
mean step containment
horizon mismatch cells
check_dims
```

其中：

```text
fully contained
```

表示有多少条 trajectory 被 reachable tube 完整包含。

```text
mean step containment
```

表示平均有多少比例的 time steps 在 tube 内。

```text
horizon mismatch cells
```

如果不是 0，说明 trajectory 长度和 reachable tube 长度不一致，可能需要使用 `--max-steps` 或重新运行 verification。

---

## 12. 常见问题

### 12.1 `checked trajectories : 0`

通常说明：

```text
没有 trajectory 的初始状态落在 verification grid 内
```

可以尝试去掉：

```bash
--only-in-grid
```

或者检查 `safety_result.json` 的 grid 范围是否太窄。

---

### 12.2 `horizon mismatch cells` 不是 0

说明：

```text
safety_result 的 reachable tube 步数小于你要比较的 trajectory 步数
```

解决方法：

```text
1. 重新运行 verification，让 safety_result 的 num_steps 和 trajectory 一致
2. 或者加 --max-steps 指定只比较前若干步
```

例如：

```bash
--max-steps 20
```

---

### 12.3 CartPole 需要 `--init-source traj`

CartPole 的 `states.npz` 有时不是完整 4 维状态，而 trajectory 是完整 4 维：

```text
[x, x_dot, theta, theta_dot]
```

因此 CartPole 推荐使用：

```bash
--init-source traj
```

---

### 12.4 图上只画两个维度，是否代表只检查两个维度？

不一定。

图上画什么由：

```bash
--plot-dims
```

决定。

containment 检查什么由：

```bash
--check-dims
```

决定。

例如：

```bash
--plot-dims 0 1 --check-dims 0 1 2 3
```

表示：

```text
图上只画 0、1 维
但 fully contained 判断检查 0、1、2、3 维
```

---

## 13. 推荐工作流

### 第一步：先看 key

```bash
python compare.py \
  --env pendulum \
  --print-keys
```

### 第二步：跑默认比较

```bash
python compare.py \
  --env pendulum \
  --split test \
  --only-in-grid
```

### 第三步：检查终端输出

重点看：

```text
checked trajectories
fully contained
horizon mismatch cells
check_dims
```

### 第四步：打开图片

```text
results/<env>/compare_tube/
```

里面会有：

```text
real_vs_reachable_tube_examples.png
dwm_vs_reachable_tube_examples.png
```

---

## 14. 最终建议

三种环境都可以用统一命令：

```bash
python compare.py \
  --env ENV_NAME \
  --split test \
  --only-in-grid
```

其中 `ENV_NAME` 是：

```text
mountain_car
pendulum
cartpole
```

只有当 safety result 的 verification horizon 和 trajectory horizon 不一致时，才需要额外加：

```bash
--max-steps
```
