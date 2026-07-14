# compare.py 使用说明

激活环境：
```bash
conda activate /home/tealab_shared/starv/env/starv_shared
```

## 作用

`compare.py` 用于比较：

- Real trajectory
- DWM trajectory
- StarV verification 生成的 reachable tube

程序会自动统计：

- 完整包含的轨迹数量
- 平均时间步包含率
- 最好和最差的 containment 轨迹

并输出两张图：

```text
real_vs_reachable_tube.png
dwm_vs_reachable_tube.png
```

---

## 支持的环境

| 环境 | 默认比较维度 | 含义 |
|---|---|---|
| `cartpole` | `(0, 2)` | 小车位置 `x` 和杆角度 `theta` |
| `mountain_car` | `(0, 1)` | 位置和速度 |
| `pendulum` | `(0, 1)` | 角度和角速度 |

---

## 直接运行

如果脚本顶部的默认路径已经设置正确：

```bash
python compare.py
```

当前默认环境由下面这一行决定：

```python
DEFAULT_ENV = "cartpole"
```

默认文件路径由以下变量决定：

```python
DEFAULT_SAFETY_PATH
DEFAULT_REAL_TRAJ_PATH
DEFAULT_DWM_TRAJ_PATH
DEFAULT_OUT_DIR
```

---

## 指定环境运行

### CartPole

```bash
python compare.py --env cartpole
```

默认使用：

```text
plot dims  = (0, 2)
check dims = (0, 2)
```

### MountainCar

```bash
python compare.py --env mountain_car
```

默认使用 `(0, 1)`。

### Pendulum

```bash
python compare.py --env pendulum
```

默认使用 `(0, 1)`。

> `--env` 只负责选择比较维度，不会自动修改文件路径。

---

## 手动指定路径

```bash
python compare.py   --env cartpole   --safety results/cartpole/safety_result.json   --real /path/to/real_trajectories.npz   --dwm /path/to/dwm_trajectories.npz   --outdir results/cartpole/compare_plot
```

---

## 常用参数

```bash
--plot-dims 0 2
```

指定图中显示的两个状态维度。

```bash
--check-dims 0 2
```

指定 containment 判断使用的状态维度。

```bash
--max-steps 10
```

只比较前10个 transition steps，即状态 `t=0` 到 `t=10`。

```bash
--delta 0.01
```

在 containment 检查时扩大 reachable tube 上下界。

```bash
--print-keys
```

查看 `.npz` 文件中的 key。

---

## 输出结果说明

图中会显示：

```text
FULLY CONTAINED
```

表示该轨迹所有检查时间步都位于 reachable tube 内。

```text
NOT fully contained
```

表示至少有一个时间步位于 reachable tube 外。

程序会自动选择：

- `Worst containment`
- `Best containment`

不需要手动指定 trajectory index。

---

## 注意

`compare.py` 只读取并绘制 `safety_result.json` 中已有的 reachable tube。

它不会重新运行 verification，也不会缩小或重新计算 reachable tube。
