# StarV Verification Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 新增一个中文 Jupyter notebook，让 CartPole、MountainCar 和 Pendulum 的 StarV verification 可以分别以 128-rank MPI 后台任务启动，并可查看 PID、运行状态和日志。

**架构：** notebook 自己只负责定位仓库、校验输入、构造 MPI 命令和启动脱离 kernel 的子进程，实际 verification 继续完全交给现有 `verify.py`。一个轻量 `unittest` 文件把 notebook 当作 JSON 和 Python source 检查，并 mock `subprocess.Popen` 验证启动参数，不真正运行 MPI。

**技术栈：** Jupyter Notebook v4 JSON、Python 3.9 标准库（`json`、`os`、`pathlib`、`subprocess`）、`unittest`、`unittest.mock`

## 全局约束

- design/spec/plan 的说明文字使用中文；代码、路径、命令和必要技术名词保留英文。
- 只新增 `notebooks/starv_verification.ipynb` 和 `tests/test_starv_verification_notebook.py`，不修改 `verify.py` 或任何配置。
- 支持的环境严格限定为 `cartpole`、`mountain_car`、`pendulum`。
- MPI 固定使用 `/home/tealab_shared/starv/env/starv_shared/bin/mpirun`，Python 固定使用 `/home/tealab_shared/starv/env/starv_shared/bin/python`。
- 默认 `nproc=128`，并设置 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`GRB_THREADS=1`。
- 三个环境必须分别启动；不提供一次并发启动全部环境的入口。
- 自动测试不得真正启动 MPI 任务。

---

### Task 1: 用契约测试定义 notebook 行为

**Files:**
- Create: `tests/test_starv_verification_notebook.py`
- Test: `tests/test_starv_verification_notebook.py`

**Interfaces:**
- Consumes: design 中定义的 notebook 路径、`start_verification(env_name, nproc=128)`、`show_verification_status()` 和 `show_log(env_name, lines=20)`。
- Produces: notebook 结构、三个环境启动 cell 和 `subprocess.Popen` 参数的可执行契约。

- [ ] **Step 1: 写入失败的 notebook 契约测试**

```python
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "starv_verification.ipynb"


class StarvVerificationNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.sources = ["".join(cell["source"]) for cell in cls.notebook["cells"]]

    def test_notebook_is_clean_and_uses_starv_kernel(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(
            self.notebook["metadata"]["kernelspec"]["display_name"],
            "starv_shared",
        )
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_all_code_cells_compile(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell_{index}", "exec")

    def test_three_environments_have_independent_launch_cells(self):
        for env_name in ("cartpole", "mountain_car", "pendulum"):
            expected = f'start_verification("{env_name}")'
            self.assertEqual(sum(expected in source for source in self.sources), 1)
        self.assertFalse(any("start_all" in source for source in self.sources))

    def test_default_launch_uses_128_ranks_and_detaches(self):
        helper_source = next(
            source for source in self.sources if "def start_verification" in source
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config" / "starv_verification" / "cartpole.json"
            config.parent.mkdir(parents=True)
            config.write_text("{}", encoding="utf-8")
            mpirun = root / "mpirun"
            python = root / "python"
            mpirun.touch(mode=0o755)
            python.touch(mode=0o755)

            namespace = {"NOTEBOOK_DIR": root}
            exec(helper_source, namespace)
            namespace["MPIRUN"] = mpirun
            namespace["STARV_PYTHON"] = python

            process = mock.Mock(pid=4321)
            with mock.patch.object(namespace["subprocess"], "Popen", return_value=process) as popen:
                pid = namespace["start_verification"]("cartpole")

            self.assertEqual(pid, 4321)
            command = popen.call_args.args[0]
            self.assertEqual(
                command,
                [
                    str(mpirun),
                    "-np",
                    "128",
                    str(python),
                    "verify.py",
                    "config/starv_verification/cartpole.json",
                ],
            )
            kwargs = popen.call_args.kwargs
            self.assertEqual(kwargs["cwd"], root)
            self.assertIs(kwargs["stderr"], namespace["subprocess"].STDOUT)
            self.assertIs(kwargs["stdin"], namespace["subprocess"].DEVNULL)
            self.assertTrue(kwargs["start_new_session"])
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "GRB_THREADS"):
                self.assertEqual(kwargs["env"][name], "1")
            self.assertEqual(
                Path(kwargs["stdout"].name),
                root / "log" / "cartpole_verify.out",
            )
            self.assertTrue(kwargs["stdout"].closed)
            self.assertEqual(namespace["VERIFICATION_PIDS"], {"cartpole": 4321})

    def test_launch_rejects_invalid_inputs_before_popen(self):
        helper_source = next(
            source for source in self.sources if "def start_verification" in source
        )
        namespace = {"NOTEBOOK_DIR": ROOT}
        exec(helper_source, namespace)
        with mock.patch.object(namespace["subprocess"], "Popen") as popen:
            with self.assertRaises(ValueError):
                namespace["start_verification"]("unknown")
            with self.assertRaises(ValueError):
                namespace["start_verification"]("cartpole", nproc=0)
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认因 notebook 尚不存在而失败**

Run: `python -m unittest tests.test_starv_verification_notebook -v`

Expected: `ERROR`，错误包含 `FileNotFoundError: .../notebooks/starv_verification.ipynb`。

### Task 2: 实现 StarV verification notebook

**Files:**
- Create: `notebooks/starv_verification.ipynb`
- Test: `tests/test_starv_verification_notebook.py`

**Interfaces:**
- Consumes: `verify.py config/starv_verification/<env>.json` 命令行接口和 Task 1 的测试契约。
- Produces: `start_verification(env_name: str, nproc: int = 128) -> int`、`show_verification_status() -> None`、`show_log(env_name: str, lines: int = 20) -> None`。

- [ ] **Step 1: 创建 notebook 初始化 cell**

notebook 使用 `nbformat=4`、`nbformat_minor=5` 和与 `generate_dataset.ipynb` 相同的 `starv_shared` kernelspec。所有 code cell 的 `execution_count` 为 `null`、`outputs` 为 `[]`。初始化 code cell 内容必须为：

```python
import os
import sys
from pathlib import Path

NOTEBOOK_DIR = Path.cwd()
if not (NOTEBOOK_DIR / "verify.py").exists():
    NOTEBOOK_DIR = NOTEBOOK_DIR.parent
assert (NOTEBOOK_DIR / "verify.py").exists(), (
    "没找到仓库根目录（verify.py），请从 verifiable_wm/ 或 verifiable_wm/notebooks/ 启动 Jupyter"
)
os.chdir(NOTEBOOK_DIR)
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

print("repo root:", NOTEBOOK_DIR)
```

第一个 markdown cell 必须说明：三个环境、`starv_shared` kernel、每个任务默认 128 rank、日志位于 `log/`、三个任务同时启动会占用 384 rank、kernel 重启后用 `pgrep -af 'verify.py config/starv_verification'` 恢复查看。

- [ ] **Step 2: 添加完整的后台启动与查看 helper**

```python
import os
import subprocess
from pathlib import Path

MPIRUN = Path("/home/tealab_shared/starv/env/starv_shared/bin/mpirun")
STARV_PYTHON = Path("/home/tealab_shared/starv/env/starv_shared/bin/python")
SUPPORTED_ENVS = ("cartpole", "mountain_car", "pendulum")
VERIFICATION_PIDS = {}


def start_verification(env_name, nproc=128):
    if env_name not in SUPPORTED_ENVS:
        raise ValueError(f"不支持的环境: {env_name}; 可选值: {SUPPORTED_ENVS}")
    if not isinstance(nproc, int) or isinstance(nproc, bool) or nproc <= 0:
        raise ValueError("nproc 必须是正整数")

    config_rel = Path("config") / "starv_verification" / f"{env_name}.json"
    config_path = NOTEBOOK_DIR / config_rel
    for label, path in (
        ("config", config_path),
        ("mpirun", MPIRUN),
        ("python", STARV_PYTHON),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {label}: {path}")
    if not os.access(MPIRUN, os.X_OK):
        raise PermissionError(f"mpirun 不可执行: {MPIRUN}")
    if not os.access(STARV_PYTHON, os.X_OK):
        raise PermissionError(f"python 不可执行: {STARV_PYTHON}")

    log_dir = NOTEBOOK_DIR / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{env_name}_verify.out"
    env = os.environ.copy()
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        GRB_THREADS="1",
    )
    command = [
        str(MPIRUN),
        "-np",
        str(nproc),
        str(STARV_PYTHON),
        "verify.py",
        config_rel.as_posix(),
    ]

    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command,
            cwd=NOTEBOOK_DIR,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    VERIFICATION_PIDS[env_name] = process.pid
    print(f"[Started] env={env_name}, ranks={nproc}, PID={process.pid}")
    print(f"[Log] {log_path}")
    print(f"[Watch] tail -f {log_path.relative_to(NOTEBOOK_DIR)}")
    print(f"[Stop] kill {process.pid}  # 执行后请再次检查 MPI 子进程")
    return process.pid


def _pid_is_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def show_verification_status():
    if not VERIFICATION_PIDS:
        print("当前 kernel 尚未启动 verification；kernel 重启后请使用 pgrep 查看后台任务。")
        return
    for env_name, pid in VERIFICATION_PIDS.items():
        status = "running" if _pid_is_running(pid) else "finished"
        print(f"{env_name}: PID={pid}, status={status}")


def show_log(env_name, lines=20):
    if env_name not in SUPPORTED_ENVS:
        raise ValueError(f"不支持的环境: {env_name}; 可选值: {SUPPORTED_ENVS}")
    if not isinstance(lines, int) or isinstance(lines, bool) or lines <= 0:
        raise ValueError("lines 必须是正整数")
    log_path = NOTEBOOK_DIR / "log" / f"{env_name}_verify.out"
    if not log_path.is_file():
        print(f"日志尚不存在: {log_path}")
        return
    result = subprocess.run(
        ["tail", "-n", str(lines), str(log_path)],
        check=False,
        text=True,
        capture_output=True,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
```

- [ ] **Step 3: 添加三个独立 case 和状态汇总 cell**

三个 case 的 markdown 分别说明配置和日志路径，紧随其后的 code cell 必须且只需调用：

```python
start_verification("cartpole")
```

```python
start_verification("mountain_car")
```

```python
start_verification("pendulum")
```

状态汇总 section 提供一个 code cell：

```python
show_verification_status()
```

日志查看 section 提供一个默认只读示例，并注释另外两个环境：

```python
show_log("cartpole", lines=20)
# show_log("mountain_car", lines=20)
# show_log("pendulum", lines=20)
```

- [ ] **Step 4: 运行 notebook 契约测试并确认通过**

Run: `python -m unittest tests.test_starv_verification_notebook -v`

Expected: 5 tests run，结果为 `OK`，且没有启动 `mpirun`。

- [ ] **Step 5: 运行 notebook JSON 和工作树检查**

Run: `python -m json.tool notebooks/starv_verification.ipynb >/dev/null && git diff --check -- notebooks/starv_verification.ipynb tests/test_starv_verification_notebook.py`

Expected: exit code 0，无输出。

- [ ] **Step 6: 提交实现**

```bash
git add -f notebooks/starv_verification.ipynb
git add tests/test_starv_verification_notebook.py
git commit -m "feat: add StarV verification notebook"
```
