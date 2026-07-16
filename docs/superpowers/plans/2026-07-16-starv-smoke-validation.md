# Three-Case StarV Smoke Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify CartPole, Pendulum, and MountainCar end to end with their selected decoder checkpoints on isolated central 4-by-4 StarV grids.

**Architecture:** Add three data-only smoke configurations that preserve formal cell widths and all 30 verification steps while selecting only the central sixteen cells. Keep formal configs and outputs untouched, then validate each generated result JSON with a read-only summary command.

**Tech Stack:** JSON, Python 3, unittest/pytest, mpi4py, StarV, pybdr

## Global Constraints

- Do not modify `config/starv_verification/cartpole.json`, `pendulum.json`, or `mountain_car.json`.
- Do not write to `results/<env>/safety_result.json`; use `results/smoke/<env>/safety_result.json`.
- Preserve the formal grid's cell width along every dimension.
- Run sixteen initial cells and the existing 30-step horizon for every case.
- Treat smoke results as integration evidence, not full-range formal conclusions.

---

### Task 1: Add isolated smoke configurations

**Files:**
- Create: `tests/test_starv_smoke_configs.py`
- Create: `config/starv_verification/smoke/cartpole.json`
- Create: `config/starv_verification/smoke/pendulum.json`
- Create: `config/starv_verification/smoke/mountain_car.json`

**Interfaces:**
- Consumes: the existing StarV configuration schema loaded by `verify.py::load_input(file_path: str)`.
- Produces: three JSON objects with `layers`, `verifier`, `grid`, and `output_prefix` fields.

- [ ] **Step 1: Write the failing configuration test**

```python
import json
import math
import unittest
from pathlib import Path


class StarVSmokeConfigTests(unittest.TestCase):
    CASES = {
        "cartpole": {
            "checkpoint": "dwm_weight/now_weight/cartpole/alpha_lambda_grid/alpha_8/lambda_0.1/seed_2025/decoder_best_total.pth",
            "nums": [4, 1, 4, 1],
            "ranges": [(0.18, 0.42), (0.0, 0.0), (0.078, 0.102), (0.0, 0.0)],
        },
        "pendulum": {
            "checkpoint": "dwm_weight/now_weight/pendulum/alpha_lambda_grid/alpha_16/lambda_0.5/seed_2025/decoder_best_total.pth",
            "nums": [4, 4],
            "ranges": [(1.03, 1.07), (4.53, 4.57)],
        },
        "mountain_car": {
            "checkpoint": "dwm_weight/now_weight/mountain_car/alpha_lambda_grid/background/alpha_16/lambda_0.5/seed_2025/decoder_best_total.pth",
            "nums": [4, 4],
            "ranges": [(0.03, 0.07), (0.003, 0.007)],
        },
    }

    def test_smoke_configs_use_selected_checkpoints_and_sixteen_isolated_cells(self):
        for env, expected in self.CASES.items():
            with self.subTest(env=env):
                path = Path("config/starv_verification/smoke") / f"{env}.json"
                config = json.loads(path.read_text(encoding="utf-8"))
                dims = config["grid"]["dims"]

                self.assertEqual(
                    config["layers"]["Decoder"]["kwargs"]["weights"],
                    expected["checkpoint"],
                )
                self.assertEqual([dim["num"] for dim in dims], expected["nums"])
                self.assertEqual(math.prod(expected["nums"]), 16)
                self.assertEqual(
                    [(dim["start"], dim["stop"]) for dim in dims],
                    expected["ranges"],
                )
                self.assertEqual(config["verifier"]["kwargs"]["num_steps"], 30)
                self.assertEqual(
                    config["output_prefix"],
                    f"results/smoke/{env}/safety_result",
                )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python -m pytest tests/test_starv_smoke_configs.py -q
```

Expected: FAIL with `FileNotFoundError` for `config/starv_verification/smoke/cartpole.json`.

- [ ] **Step 3: Add the minimal JSON configurations**

Create `config/starv_verification/smoke/cartpole.json`:

```json
{
  "layers": {
    "Decoder": {"args": [], "kwargs": {"weights": "dwm_weight/now_weight/cartpole/alpha_lambda_grid/alpha_8/lambda_0.1/seed_2025/decoder_best_total.pth"}},
    "Controller": {"args": [], "kwargs": {"weights": "/home/tealab_shared/starv/weights/cartpole/controller_cp.pth", "activation": "sigmoid"}}
  },
  "verifier": {"name": "CartpoleVerifier", "args": [], "kwargs": {"goal_angle_threshold": 0.209, "num_steps": 30, "early_stop": false}},
  "grid": {"dims": [
    {"name": "pos", "start": 0.18, "stop": 0.42, "num": 4},
    {"name": "vel", "start": 0.0, "stop": 0.0, "num": 1},
    {"name": "angle", "start": 0.078, "stop": 0.102, "num": 4},
    {"name": "avel", "start": 0.0, "stop": 0.0, "num": 1}
  ]},
  "output_prefix": "results/smoke/cartpole/safety_result"
}
```

Create `config/starv_verification/smoke/pendulum.json`:

```json
{
  "layers": {
    "Decoder": {"args": [], "kwargs": {"weights": "dwm_weight/now_weight/pendulum/alpha_lambda_grid/alpha_16/lambda_0.5/seed_2025/decoder_best_total.pth"}},
    "Controller": {"args": [], "kwargs": {"weights": "/home/tealab_shared/starv/weights/pendulum/controller_pen.pth"}}
  },
  "verifier": {"name": "PendulumVerifier", "args": [], "kwargs": {"goal_angle_threshold": 0.15, "num_steps": 30, "early_stop": true}},
  "grid": {"dims": [
    {"name": "theta", "start": 1.03, "stop": 1.07, "num": 4},
    {"name": "omega", "start": 4.53, "stop": 4.57, "num": 4}
  ]},
  "output_prefix": "results/smoke/pendulum/safety_result"
}
```

Create `config/starv_verification/smoke/mountain_car.json`:

```json
{
  "layers": {
    "Decoder": {"args": [], "kwargs": {"weights": "dwm_weight/now_weight/mountain_car/alpha_lambda_grid/background/alpha_16/lambda_0.5/seed_2025/decoder_best_total.pth"}},
    "Controller": {"args": [], "kwargs": {"weights": "/home/tealab_shared/starv/weights/mountain_car/controller_mc.pth"}}
  },
  "verifier": {"name": "MountainCarVerifier", "args": [], "kwargs": {"goal_position_threshold": 0.6, "num_steps": 30, "early_stop": false, "save_history": true}},
  "grid": {"dims": [
    {"name": "pos", "start": 0.03, "stop": 0.07, "num": 4},
    {"name": "vel", "start": 0.003, "stop": 0.007, "num": 4}
  ]},
  "output_prefix": "results/smoke/mountain_car/safety_result"
}
```

- [ ] **Step 4: Run focused and full tests and verify GREEN**

Run:

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python -m pytest tests/test_starv_smoke_configs.py -q
/home/tealab_shared/starv/env/starv_shared/bin/python -m pytest -q
```

Expected: focused test PASS; full suite exits zero with no failures.

### Task 2: Run and audit StarV smoke validation

**Files:**
- Create: `results/smoke/cartpole/safety_result.json`
- Create: `results/smoke/pendulum/safety_result.json`
- Create: `results/smoke/mountain_car/safety_result.json`

**Interfaces:**
- Consumes: the three Task 1 JSON configs through `python verify.py <config>`.
- Produces: result JSON containing the effective config and exactly sixteen entries under `cells`.

- [ ] **Step 1: Run all three smoke configurations**

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python verify.py config/starv_verification/smoke/cartpole.json
/home/tealab_shared/starv/env/starv_shared/bin/python verify.py config/starv_verification/smoke/pendulum.json
/home/tealab_shared/starv/env/starv_shared/bin/python verify.py config/starv_verification/smoke/mountain_car.json
```

Expected for each command: `Generated 16 cells for verification`, exit zero, and `Verification results saved` under `results/smoke/<env>/`.

- [ ] **Step 2: Audit every result**

```python
import json
from pathlib import Path

for env in ("cartpole", "pendulum", "mountain_car"):
    path = Path("results/smoke") / env / "safety_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    cells = result["cells"]
    assert len(cells) == 16
    assert all(isinstance(cell.get("result"), bool) or "error_msg" in cell for cell in cells)
    print(
        env,
        "safe=", sum(cell.get("result") is True for cell in cells),
        "unsafe=", sum(cell.get("result") is False for cell in cells),
        "errors=", sum("error_msg" in cell for cell in cells),
        "total_bounds=", sum(len(cell.get("bounds", [])) for cell in cells),
    )
```

Expected: all assertions pass and each case reports sixteen classified or errored cells.

- [ ] **Step 3: Report the smoke evidence**

Report the checkpoint, grid cardinality, safe/unsafe/error counts, elapsed time printed by `verify.py`, and total stored bounds for each case. Explicitly state that the central sixteen cells do not establish a result over the complete formal grid.
