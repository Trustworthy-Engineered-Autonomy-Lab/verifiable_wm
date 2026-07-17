# StarV Verification Notebook 设计文档

## 目标

新增 `notebooks/starv_verification.ipynb`。该 notebook 沿用 `notebooks/generate_dataset.ipynb` 的组织方式和说明风格，让用户可以从 Jupyter 中分别启动 CartPole、MountainCar 和 Pendulum 的 StarV verification。每个任务都以脱离 notebook kernel 的 128-rank MPI 后台进程运行。

## 范围

notebook 覆盖以下三个现有配置：

- `config/starv_verification/cartpole.json`
- `config/starv_verification/mountain_car.json`
- `config/starv_verification/pendulum.json`

本次工作不修改 `verify.py`、verification 配置、模型权重或结果格式，也不引入任务调度系统。notebook 不提供同时启动三个任务的 cell，因为这会一次启动 384 个 MPI rank。

## Notebook 结构

notebook 仿照 `generate_dataset.ipynb`，依次包含：

1. 中文说明：介绍三个 case、所需的 `starv_shared` kernel、默认 128 rank、日志位置，以及同时运行多个 case 的资源风险。
2. 初始化 cell：无论从仓库根目录还是 `notebooks/` 启动 Jupyter，都能定位仓库根目录、切换当前目录并把根目录加入 `sys.path`。
3. 公共 helper cell：定义任务启动和状态查看函数。
4. 三个 case：分别提供 CartPole、MountainCar 和 Pendulum 的独立启动 cell。
5. 状态汇总：只读地查看已启动任务的状态和日志末尾，不启动或停止任务。

## 启动接口

notebook 提供以下函数：

```python
start_verification(env_name, nproc=128)
```

`env_name` 只支持 `cartpole`、`mountain_car` 和 `pendulum`。每个环境对应：

- 配置：`config/starv_verification/<env_name>.json`
- 日志：`log/<env_name>_verify.out`
- MPI 可执行文件：`/home/tealab_shared/starv/env/starv_shared/bin/mpirun`
- Python 可执行文件：`/home/tealab_shared/starv/env/starv_shared/bin/python`

启动前，helper 会检查环境名称是否合法，以及配置、MPI 可执行文件和 Python 可执行文件是否存在；需要时自动创建 `log/`。

子进程使用以下线程限制：

```text
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
GRB_THREADS=1
```

启动效果等价于：

```bash
nohup bash -lc 'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GRB_THREADS=1; /home/tealab_shared/starv/env/starv_shared/bin/mpirun -np 128 /home/tealab_shared/starv/env/starv_shared/bin/python verify.py config/starv_verification/<env>.json' > log/<env>_verify.out 2>&1 &
```

实际实现使用参数列表形式的 `subprocess.Popen`，指定仓库根目录为工作目录，将 stdout 和 stderr 重定向到日志，并设置 `start_new_session=True`。这样既避免 shell 引号问题，也能让任务脱离 notebook kernel 在后台继续运行。函数返回 launcher PID，同时打印环境、PID、日志路径、`tail -f` 命令和针对该 PID 的 `kill` 命令。

## 状态与日志查看

notebook 在内存中保存当前 kernel 启动的环境名和 PID 映射。只读 helper 可以报告这些 PID 是否仍然存在，并显示各环境日志的最后若干行。

PID 映射只是当前 kernel 的便捷状态，不是持久化任务注册表。重启 notebook kernel 后，已经脱离的 MPI 任务会继续运行，但内存中的 PID 映射会丢失。notebook 开头会说明可使用以下命令重新查找任务：

```bash
pgrep -af 'verify.py config/starv_verification'
```

notebook 不会自动停止任何进程。用户必须有意执行页面打印的 `kill <PID>` 命令。由于各 rank 由 MPI launcher 管理，命令针对 launcher PID；执行后仍应再次检查进程列表。

## 错误处理

出现以下情况时，启动函数会在创建子进程前直接报错：

- 环境名称不受支持；
- `nproc` 不是正整数；
- 配置或所需可执行文件不存在。

`verify.py` 已将 stdout 和 stderr 设置为行缓冲，因此 `tail -f` 可以及时看到进度和异常。verification 运行阶段的失败继续由 `verify.py` 处理，并完整保留在对应环境的日志中。

## 验证方案

自动检查不会真正启动 128-rank verification。验证内容包括：

1. 将 notebook 解析为 JSON，检查必要的 cell 结构和 metadata；
2. 编译每个 code cell，捕获 Python 语法错误；
3. 用 mock 替换 `subprocess.Popen` 后执行 helper 定义；
4. 断言默认启动使用 128 rank、`starv_shared` 的可执行文件、正确的配置和日志路径、三个线程限制、stdout/stderr 重定向、仓库根目录工作目录及新 session；
5. 检查三个环境的启动 cell 分别引用对应环境。

## 验收标准

- 新 notebook 是有效 JSON，可以在 Jupyter 中正常打开。
- 结构和中文说明与 `generate_dataset.ipynb` 保持一致。
- CartPole、MountainCar 和 Pendulum 各有独立启动 cell。
- 默认任务使用 128 个 MPI rank，且每个 rank 只使用一个计算线程。
- notebook 断开后任务仍能继续运行，日志分别写入 `log/`。
- 用户可以查看 PID 和日志末尾，不会意外启动或停止其他任务。
- 自动检查不会占用 128 核资源。
