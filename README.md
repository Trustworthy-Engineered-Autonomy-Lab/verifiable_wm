# Verifiable World Model 使用说明

本项目用于训练 Deterministic World Model，也就是从物理状态生成图像的 Decoder，并进一步用于闭环安全验证。当前支持三个环境：

- Pendulum
- CartPole
- MountainCar

## 项目结构

```text
verifiable_wm/
├── config/                  # 配置文件
│   ├── make_dataset/         # 生成数据集用的配置
│   └── starv_verification/   # StarV 验证用的配置
├── data_generation/          # 数据集生成代码
├── simulation/               # 普通仿真和 PyTorch 模型
├── training/                 # 模型训练代码
├── tools/                    # 可视化工具
├── datasets/                 # 生成的数据集
├── weights/                  # 训练好的模型权重
├── results/                  # 图片和验证结果
├── starv_verification/       # StarV 形式化验证代码
├── verify.py                 # 验证入口
└── verify.sh                 # 服务器验证脚本
```

## 环境准备

所有命令都需要在项目根目录下运行：

```bash
cd /home/UFAD/你的用户名/verifiable_wm
conda activate /home/tealab_shared/starv/env/starv_shared
```

## 生成 Decoder 数据集

```bash
python -m data_generation.make_decoder_dataset config/make_dataset/pendulum.json
python -m data_generation.make_decoder_dataset config/make_dataset/cartpole.json
python -m data_generation.make_decoder_dataset config/make_dataset/mountain_car.json
```

生成的数据会保存在：

```text
datasets/pendulum/data/dataset_v1/states.npz
datasets/cartpole/data/dataset_v1/states.npz
datasets/mountain_car/data/dataset_v1/states.npz
```

## 训练 Decoder 模型

### Pendulum

```bash
python -m training.train_decoder \
  --dataset datasets/pendulum/data/dataset_v1/states.npz \
  --output weights/pendulum/decoder.pth \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-3 \
  --loss weighted_mse
```

### CartPole

```bash
python -m training.train_decoder \
  --dataset datasets/cartpole/data/dataset_v1/states.npz \
  --output weights/cartpole/decoder.pth \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-3 \
  --loss weighted_mse
```

### MountainCar

```bash
python -m training.train_decoder \
  --dataset datasets/mountain_car/data/dataset_v1/states.npz \
  --output weights/mountain_car/decoder.pth \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-3 \
  --loss weighted_mse
```

## 查看 Decoder 训练效果

### Pendulum

```bash
python -m tools.visualize_decoder_result \
  --dataset datasets/pendulum/data/dataset_v1/states.npz \
  --weights weights/pendulum/decoder.pth \
  --output results/pendulum_decoder_result.png \
  --num 8
```

### CartPole

```bash
python -m tools.visualize_decoder_result \
  --dataset datasets/cartpole/data/dataset_v1/states.npz \
  --weights weights/cartpole/decoder.pth \
  --output results/cartpole_decoder_result.png \
  --num 8
```

### MountainCar

```bash
python -m tools.visualize_decoder_result \
  --dataset datasets/mountain_car/data/dataset_v1/states.npz \
  --weights weights/mountain_car/decoder.pth \
  --output results/mountain_car_decoder_result.png \
  --num 8
```

## 注意事项

1. 现在项目已经按模块整理，运行脚本时推荐使用 `python -m ...`，不要直接运行 `.py` 文件。
2. `data_generation/`、`simulation/`、`training/`、`tools/`、`starv_verification/` 文件夹下都应有 `__init__.py`。
3. 如果重新生成数据集，并且 JSON 里的 `output_dir` 还是 `dataset_v1`，会覆盖原来的 `states.npz` 和 `metadata.json`。
4. Decoder 权重建议统一保存为：

```text
weights/pendulum/decoder.pth
weights/cartpole/decoder.pth
weights/mountain_car/decoder.pth
```
