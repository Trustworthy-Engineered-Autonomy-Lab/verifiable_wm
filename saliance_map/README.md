# saliance_map 说明

这个目录现在只做一件主线事情：给不同 study case 生成 saliency map / heatmap，后面作为 WM reconstruction loss 的空间权重使用。

这里不再把 CartPole 可视化当作主线。CartPole 的 renderer 问题已经修好了，之前那些对比脚本只保留为诊断工具。

## 当前结构

```text
saliance_map/
├── README.md
├── scripts/
│   ├── precompute_saliency_maps.py
│   └── diagnostics/
│       ├── compare_heatmap_methods.py
│       └── preview_cartpole_render.py
└── output/
    └── diagnostics/
```

## 放东西的规则

- 主线脚本放在 `scripts/`。
- 临时检查、画图、方法对比脚本放在 `scripts/diagnostics/`。
- 生成给训练用的 `.npz` 不放在 `saliance_map/output/`，而是放回对应的 dataset 目录。
- `saliance_map/output/` 只放临时图片、grid、preview 这类方便看的东西。

## 主线脚本

### `scripts/precompute_saliency_maps.py`

这个脚本用来给某个 dataset 预计算 saliency map。

默认跑 CartPole：

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python \
  saliance_map/scripts/precompute_saliency_maps.py
```

默认读取：

```text
config/make_decoder_dataset/cartpole.json
datasets/cartpole/data/dataset_v1/states.npz
```

默认输出：

```text
datasets/cartpole/data/dataset_v1/saliency_occlusion.npz
```

这个脚本会从 config 里读取 controller 的权重和 activation，所以不只限于 CartPole。比如 MountainCar：

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python \
  saliance_map/scripts/precompute_saliency_maps.py \
  --config config/make_decoder_dataset/mountain_car.json
```

Pendulum：

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python \
  saliance_map/scripts/precompute_saliency_maps.py \
  --config config/make_decoder_dataset/pendulum.json
```

如果之后某个 case 用不同 saliency 算法，也应该在这个脚本里加新 method，而不是另起一堆只服务某个 case 的脚本。

目前实现的方法：

```text
occlusion
```

后续可以继续加：

```text
integrated_gradients
object_occlusion
state_counterfactual
```

## 诊断脚本

这些脚本不是主线，只是用来临时检查。

### `scripts/diagnostics/preview_cartpole_render.py`

用于看 CartPole dataset 里的图片是否正常，也可以和当前 `env.py` 重新 render 的结果对比。

常用命令：

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python \
  saliance_map/scripts/diagnostics/preview_cartpole_render.py --render-current
```

输出默认在：

```text
saliance_map/output/diagnostics/cartpole_render_preview/
```

这个脚本主要是为了防止再次出现旧的黑块数据问题。现在 dataset 已经修好，所以不是每次实验都必须跑。

### `scripts/diagnostics/compare_heatmap_methods.py`

用于把不同 heatmap 方法画在一张图里比较。它是纯视觉诊断，不是正式实验主线。

它现在包含：

```text
vanilla gradient
SmoothGrad
SmoothGrad^2
IG-white^2
Grad-CAM
Occlusion-white
```

常用命令：

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python \
  saliance_map/scripts/diagnostics/compare_heatmap_methods.py
```

输出默认在：

```text
saliance_map/output/diagnostics/heatmap_methods_compare/
```

这个脚本只帮助我们看方法大概长什么样，不应该被写成主要贡献。

## 当前研究判断

目前 CartPole 上主方法暂定为 `occlusion`，也就是 occlusion-white。原因是它直接测量：

```text
遮挡某个区域后，controller action 变化有多大
```

但这不代表所有 case 都必须用同一种 saliency map。MountainCar、Pendulum、Braking 以后可以分别检查，必要时选择不同算法。

