# saliency_map 说明

这个目录做两件事：

1. 主线：给不同 study case 预计算 saliency map / heatmap，后面作为 WM reconstruction loss 的空间权重使用。
2. 诊断：围绕 saliency/decoder 的一些可视化检查（方法对比、重建对比、渲染健全性、checkpoint 指标）。

跑和看这些内容的主入口是 `notebooks/saliency_diagnostics.ipynb`；下面列的脚本都保留了 `parse_args(argv)`
/ `run(args)` 的拆分，所以既可以在 notebook 里 `run(parse_args([...]))` 调用，也可以照常独立 CLI 跑。

## 当前结构

```text
saliency_map/
├── README.md
├── methods.py                 saliency 计算方法库（controller/权重加载 + 6 种方法），被下面两处共用
├── scripts/
│   ├── precompute_saliency_maps.py        主线：产出训练用 .npz
│   └── diagnostics/
│       ├── compare_heatmap_methods.py     诊断：6 种方法并排看
│       ├── preview_cartpole_render.py     诊断：渲染健全性检查
│       └── eval_decoder.py                诊断：任意 checkpoint 的指标读数
└── output/
    └── previews/
```

`notebooks/saliency_diagnostics.ipynb` 是这五个脚本的薄封装（跟 `notebooks/compare_tube.ipynb` 封装
`compare.py` 是同一个模式）：逻辑都在 `.py` 里，notebook 只负责调用 + 展示图片，不重复实现。

## folder file 放置

- 多个脚本会用到的 controller/权重加载、saliency 计算方法放 `methods.py`，不要在各脚本里各写一份。
- 主线脚本放在 `scripts/`。
- 临时检查、画图、方法/重建对比脚本放在 `scripts/diagnostics/`；如果两个诊断脚本除了参数（env 名、
  checkpoint 路径）外逻辑一样，参数化合并成一个，不要复制粘贴出第二个文件。
- 生成给训练用的 `.npz` 不放在 `saliency_map/output/`，而是放回对应的 dataset 目录。
- `saliency_map/output/previews/` 放临时图片。一般是一张图一个清楚的文件名，不再一张图开一个文件夹。

## `methods.py`

共享库，两处调用者：

- `precompute_saliency_maps.py` 只用其中的 `occlusion`（当前唯一的主线方法）。
- `compare_heatmap_methods.py` 用全部六种方法做诊断对比。

包含：`load_json`、`load_state_dict`、`build_controller`、`normalize_per_image`，以及
`vanilla_gradient`、`smoothgrad`、`integrated_gradients`、`gradcam`、`occlusion`。

## 主线脚本

### `scripts/precompute_saliency_maps.py`

给某个 dataset 预计算 saliency map。默认读取 `config/make_decoder_dataset/<env>.json` 和
`datasets/<env>/data/dataset_v1/decoder_states.npz`，输出到
`datasets/<env>/data/dataset_v1/saliency_occlusion.npz`。

会从 config 里读取 controller 的权重和 activation，所以不限于 CartPole，CartPole/Pendulum/MountainCar
都能跑：

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python \
  saliency_map/scripts/precompute_saliency_maps.py \
  --config config/make_decoder_dataset/pendulum.json
```

如果之后某个 case 用不同 saliency 算法，应该在 `methods.py` 里加新方法，而不是另起一堆只服务某个 case
的脚本。目前实现：`occlusion`。后续可以继续加：`integrated_gradients`、`object_occlusion`、
`state_counterfactual`。

## 诊断脚本

这些脚本不是主线，只是用来临时检查；都推荐从 `notebooks/saliency_diagnostics.ipynb` 里跑。

### `scripts/diagnostics/compare_heatmap_methods.py`

把 vanilla gradient / SmoothGrad / SmoothGrad² / IG-white² / Grad-CAM / Occlusion-white 六种方法画在
一张图里比较，纯视觉诊断，不是正式实验主线，也不应该被当成主要贡献。

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python \
  saliency_map/scripts/diagnostics/compare_heatmap_methods.py --config config/make_decoder_dataset/cartpole.json
```

输出默认在 `saliency_map/output/previews/<study_case>_saliency_methods.png`。

### `scripts/diagnostics/preview_cartpole_render.py`

看 CartPole dataset 里的图片是否正常，也可以和当前 `env.py`/`dynamic.py` 重新 render 的结果对比，防止
再次出现旧的黑块渲染 bug。数据集已经修好，不是每次实验都必须跑，回归检查时用。

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python \
  saliency_map/scripts/diagnostics/preview_cartpole_render.py --render-current
```

输出默认在 `saliency_map/output/previews/cartpole_render_train_saved_vs_current.png`。

### `scripts/diagnostics/eval_decoder.py`

对不是 `train_decoder.py` 直接训出来的 checkpoint 算 test 指标，复用
`train_decoder.py` 的 `load_split`/`compute_weight`/`evaluate`，数字算法跟 `metrics.json` 里的一致，
可以放在同一张对比表里看。

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python \
  saliency_map/scripts/diagnostics/eval_decoder.py \
  --config config/train_decoder/pendulum/intensity.json \
  --weights dwm_weight/pendulum/intensity/decoder_best_total.pth \
  --label "intensity"
```

## 当前研究判断

目前 CartPole 上主方法暂定为 `occlusion`，也就是 occlusion-white。原因是它直接测量：

```text
遮挡某个区域后，controller action 变化有多大
```

但这不代表所有 case 都必须用同一种 saliency map。MountainCar、Pendulum、Braking 以后可以分别检查，必要时选择不同算法。
