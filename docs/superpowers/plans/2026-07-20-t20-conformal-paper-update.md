# T20 Conformal Results and Paper Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate paper-ready CartPole, MountainCar, and Pendulum T20 rollouts and L2 conformal bounds from consistent decoder data and checkpoints, then align the paper with those results.

**Architecture:** Configuration tests establish one T20/range contract across decoder data, sampling, and StarV inputs. Artifact writers and validators carry horizon/checkpoint metadata so stale T30 rollouts are rejected. New checkpoints live in isolated T20 directories; validation trajectories select MountainCar hyperparameters and held-out test trajectories produce six `Gamma_0.95` values.

**Tech Stack:** Python 3.9, NumPy, PyTorch, pandas, `unittest`, JSON/NPZ artifacts, LaTeX.

## Global Constraints

- Work only on CartPole, MountainCar, and Pendulum; do not run or modify cGAN, Braking System, or StarV containment.
- Use `T = 20`, producing 21 states and 20 actions per trajectory.
- Use L2 for every conformal score; wrap Pendulum theta before the norm.
- Keep the current Experimental Setup lambda sentence unchanged.
- Keep MountainCar decoder position at `[-1.20, 0.60]` and set decoder velocity to `[-0.08, 0.08]`.
- Use the MountainCar goal `p_20 >= 0.6`.
- Preserve the user's `.gitignore` edits and extend, rather than discard, the user's uncommitted L2 change in `conformal.py`.
- Preserve old T30 checkpoint directories by writing new experiments below `t20_matched` and `background_v008_t20`.
- Unknown paper-table values are `--`; never copy stale cGAN, Braking, containment, or T30 results.

---

### Task 1: Lock the T20 and Range Configuration Contract

**Files:**
- Create: `tests/test_experiment_configs.py`
- Modify: `config/make_decoder_dataset/{cartpole,mountain_car,pendulum}.json`
- Modify: `config/sampling/{cartpole,mountain_car,pendulum}.json`
- Modify: `config/starv_verification/{cartpole,mountain_car,pendulum}.json`
- Modify: `config/starv_verification/smoke/{cartpole,mountain_car,pendulum}.json`
- Modify: `starv_verification/verifiers.py`
- Modify: `tests/test_starv_smoke_configs.py`

**Interfaces:**
- Consumes: the approved decoder ranges and initial-state grids.
- Produces: active configurations with one mechanically tested horizon/range contract.

- [ ] **Step 1: Write the failing configuration test**

Create a test with these exact expected grids:

```python
GRIDS = {
    "cartpole": [
        ("pos", 0.0, 0.6, 60),
        ("vel", 0.0, 0.0, 1),
        ("angle", 0.06, 0.12, 60),
        ("avel", 0.0, 0.0, 1),
    ],
    "mountain_car": [
        ("pos", -0.2, 0.6, 80),
        ("vel", 0.0, 0.08, 80),
    ],
    "pendulum": [
        ("theta", 1.0, 2.0, 100),
        ("omega", 4.5, 5.0, 50),
    ],
}
```

For each environment, assert `make_decoder_dataset.rollout_steps == 20`, `sampling.rollout_steps == 20`, full/smoke `num_steps == 20`, and the full grid equals `GRIDS[env]`. Separately assert MountainCar decoder position `(-1.2, 0.6)`, velocity `(-0.08, 0.08)`, and goal threshold `0.6`.

- [ ] **Step 2: Run the test and verify it fails on current T30/old grids**

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python -m unittest tests.test_experiment_configs tests.test_starv_smoke_configs -v
```

Expected: failures identify the current `30` horizons and the old MountainCar/Pendulum grids.

- [ ] **Step 3: Apply the exact configuration changes**

Set the active rollout/verifier horizons and `BaseVerifier` default to 20. Replace only the full MountainCar and Pendulum grids; retain the already-correct full CartPole grid. Change only MountainCar decoder velocity, retaining its position. Change the smoke assertion to 20 without expanding smoke grids.

- [ ] **Step 4: Re-run focused tests and commit**

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python -m unittest tests.test_experiment_configs tests.test_starv_smoke_configs -v
git add -f tests/test_experiment_configs.py
git add config/make_decoder_dataset config/sampling config/starv_verification starv_verification/verifiers.py tests/test_starv_smoke_configs.py
git commit -m "config: align three benchmarks to T20 ranges"
```

Expected: all focused tests pass.

---

### Task 2: Record and Validate T20 Trajectory Provenance

**Files:**
- Modify: `make_decoder_dataset.py`
- Modify: `sampling.py`
- Modify: `ablation.py`
- Modify: `tests/test_sampling.py`
- Modify: `tests/test_ablation.py`

**Interfaces:**
- Consumes: `rollout_steps`, StarV initial states, decoder variant, and checkpoint path.
- Produces: trajectory NPZ metadata and validators that reject T30/stale/non-finite artifacts.

- [ ] **Step 1: Add failing metadata and validation tests**

Give the sampling fixture `rollout_steps: 1` and `starv_config: "config/starv.json"`, then assert:

```python
self.assertEqual(data["rollout_steps"].item(), 1)
self.assertEqual(data["starv_config"].item(), "config/starv.json")
```

Extend ablation fixtures with `rollout_steps=np.array(20)` and 21-state/20-action arrays. Add tests that reject a 31-state file with a reason containing `horizon`, reject non-finite trajectory/actions, and reject changed initial states.

- [ ] **Step 2: Run tests and verify the new assertions fail**

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python -m unittest tests.test_sampling tests.test_ablation -v
```

- [ ] **Step 3: Add minimal metadata to both artifact writers**

Keep only one `save_real_trajectories` definition in `make_decoder_dataset.py`. Add these keys to real and DWM trajectory files:

```python
arrays["rollout_steps"] = np.array(int(config["rollout_steps"]))
arrays["starv_config"] = np.array(str(config["starv_config"]))
arrays["controller_weights"] = np.array(str(config["controller"]["weights"]))
```

Retain DWM `variant` and `decoder_weights` metadata.

- [ ] **Step 4: Strengthen `_validate_trajectory_file`**

Pass an explicit `expected_steps` from the sampling config. For every split require:

```python
traj.shape[1] == expected_steps + 1
actions.shape[1] == expected_steps
np.isfinite(traj).all()
np.isfinite(actions).all()
int(data["rollout_steps"].item()) == expected_steps
np.array_equal(states[f"{split}_states"], traj[:, 0, :])
```

Do not weaken existing checkpoint/variant checks.

- [ ] **Step 5: Isolate condition-specific summary CSVs**

Update `write_summary_tables` so all selected experiments must share one condition. For a non-default condition, write below `alpha_lambda_grid/<condition>/` rather than overwriting the root summary. Add a test for `background_v008_t20`.

- [ ] **Step 6: Re-run tests and commit**

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python -m unittest tests.test_sampling tests.test_ablation -v
git add make_decoder_dataset.py sampling.py ablation.py tests/test_sampling.py tests/test_ablation.py
git commit -m "feat: validate T20 trajectory provenance"
```

---

### Task 3: Complete the L2 Conformal Calibration Contract

**Files:**
- Modify: `conformal.py`
- Modify: `tests/test_conformal_delta.py`

**Interfaces:**
- Consumes: matched real/DWM NPZ paths and held-out split `test`.
- Produces: traceable JSON with T20 L2 `gamma` and finite-sample rank.

- [ ] **Step 1: Correct the stale L1 test and add failing contract tests**

Rename the score test to `test_computes_max_over_time_of_l2_distance_on_selected_dims` and change its expected result from 7.0 to 5.0. Add assertions for:

```python
self.assertEqual(result["norm"], "L2")
self.assertEqual(result["horizon"], 20)
self.assertEqual(result["rank"], 381)
self.assertEqual(result["gamma"], expected_order_statistic)
```

Add cases for non-finite arrays, mismatched initial states, wrong horizon, and `alpha` outside `(0, 1)`.

- [ ] **Step 2: Run conformal tests and observe failure**

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python -m unittest tests.test_conformal_delta -v
```

- [ ] **Step 3: Implement validation and result metadata**

Keep the user's L2 norm. Compute `rank = ceil((n + 1) * (1 - alpha))` once and use the one-indexed order statistic without interpolation. Validate real/DWM shapes, T20 metadata, actions, finite values, and equal initial states. Return at least:

```python
{
    "split": split,
    "n": int(scores.shape[0]),
    "alpha": float(alpha),
    "rank": rank,
    "horizon": 20,
    "norm": "L2",
    "dims": list(dims),
    "circular_dims": list(circular_dims),
    "gamma": float(gamma),
    "real_path": str(real_path),
    "dwm_path": str(dwm_path),
    "decoder_weights": decoder_weights,
}
```

Change the CLI default split to `test`: validation selects MountainCar hyperparameters, while the 400 test trajectories remain untouched for calibration.

- [ ] **Step 4: Run tests and commit**

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python -m unittest tests.test_conformal_delta -v
git add conformal.py tests/test_conformal_delta.py
git commit -m "feat: add traceable T20 L2 conformal calibration"
```

This commit intentionally incorporates the user's existing L2 diff rather than reverting it.

---

### Task 4: Update Active Pipeline Documentation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1–3 behavior.
- Produces: commands and shapes consistent with T20/L2.

- [ ] **Step 1: Update active README statements**

Change trajectory examples to `(N, 21, d)` and actions to `(N, 20, 1)`. Change `t=0..30`/`t=30` to `t=0..20`/`t=20`. Replace active L1 conformal wording with L2 and state that `val` selects MC hyperparameters while `test` calibrates Gamma. Document the MountainCar heatmap command:

```bash
python saliency_map/scripts/precompute_saliency_maps.py \
  --config config/make_decoder_dataset/mountain_car.json \
  --occlusion-baseline background_median
```

- [ ] **Step 2: Verify and commit**

```bash
rg -n 'N, 31|N, 30|t=0\.\.30|t=30|L1 non-conformity' README.md
git add README.md
git commit -m "docs: describe the T20 L2 experiment pipeline"
```

Expected: the search has no matches before committing.

---

### Task 5: Regenerate Decoder Data, Real Rollouts, and Heatmaps

**Files:**
- Regenerate: `datasets/<env>/data/dataset_v1/decoder_states.npz`
- Regenerate: `datasets/<env>/data/dataset_v1/starv_states.npz`
- Regenerate: `datasets/<env>/data/dataset_v1/real_trajectories.npz`
- Regenerate: `datasets/cartpole/data/dataset_v1/saliency_occlusion.npz`
- Regenerate: `datasets/pendulum/data/dataset_v1/saliency_occlusion.npz`
- Regenerate: `datasets/mountain_car/data/dataset_v1/saliency_occlusion_background_median.npz`

**Interfaces:**
- Consumes: approved configs and fixed controllers.
- Produces: current decoder inputs/heatmaps and 2,400 real T20 trajectories per environment.

- [ ] **Step 1: Verify GPU and controller files**

```bash
nvidia-smi
test -f /home/tealab_shared/starv/weights/cartpole/controller_cp.pth
test -f /home/tealab_shared/starv/weights/mountain_car/controller_mc.pth
test -f /home/tealab_shared/starv/weights/pendulum/controller_pen.pth
```

- [ ] **Step 2: Regenerate all three datasets and real rollouts**

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python make_decoder_dataset.py config/make_decoder_dataset/cartpole.json
/home/tealab_shared/starv/env/starv_shared/bin/python make_decoder_dataset.py config/make_decoder_dataset/mountain_car.json
/home/tealab_shared/starv/env/starv_shared/bin/python make_decoder_dataset.py config/make_decoder_dataset/pendulum.json
```

Expected split counts: 1,600 train, 400 val, 400 test; real trajectories have 21 states and 20 actions.

- [ ] **Step 3: Regenerate the three heatmaps**

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python saliency_map/scripts/precompute_saliency_maps.py --config config/make_decoder_dataset/cartpole.json --occlusion-baseline white
/home/tealab_shared/starv/env/starv_shared/bin/python saliency_map/scripts/precompute_saliency_maps.py --config config/make_decoder_dataset/mountain_car.json --occlusion-baseline background_median
/home/tealab_shared/starv/env/starv_shared/bin/python saliency_map/scripts/precompute_saliency_maps.py --config config/make_decoder_dataset/pendulum.json --occlusion-baseline white
```

- [ ] **Step 4: Validate generated data before training**

Run a read-only NumPy check over all splits: correct counts, 21/20 horizon, finite arrays, matching heatmap/image counts, and trajectory `t=0` equal to `starv_states.npz`. Check MC heatmap metadata is `background_median` and CP/Pendulum are `white`.

Expected: zero failures. Generated artifacts are ignored and are not committed.

---

### Task 6: Train and Roll Out the Matched Models

**Files:**
- Generate: `dwm_weight/now_weight/cartpole/alpha_lambda_grid/t20_matched/...`
- Generate: `dwm_weight/now_weight/pendulum/alpha_lambda_grid/t20_matched/...`
- Generate: `dwm_weight/now_weight/mountain_car/alpha_lambda_grid/background_v008_t20/...`

**Interfaces:**
- Consumes: Task 5 decoder data/heatmaps and StarV initial states.
- Produces: 53 matched checkpoints and their T20 validation/test rollouts.

- [ ] **Step 1: Train CartPole's matched pair**

Use `build_experiments("cartpole", alphas=(8,), lambdas=(0, 0.1), condition="t20_matched")` and `run_training_grid(..., skip_existing=False, continue_on_error=False)`.

Expected: two `trained` rows using identical dataset, heatmap, seed, and architecture.

- [ ] **Step 2: Train Pendulum's matched pair**

Use `build_experiments("pendulum", alphas=(16,), lambdas=(0, 0.5), condition="t20_matched")` with forced training.

Expected: two `trained` rows.

- [ ] **Step 3: Train the MC background-only 49-point grid**

```python
experiments = ablation.build_experiments(
    "mountain_car",
    alphas=(0.5, 1, 2, 4, 8, 16, 32),
    lambdas=(0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
    seed=2025,
    condition="background_v008_t20",
    base_config_path=Path("config/train_decoder/mountain_car/saliency_background.json"),
)
frame = ablation.run_training_grid(
    "mountain_car",
    experiments=experiments,
    skip_existing=False,
    continue_on_error=False,
)
assert len(frame) == 49
assert set(frame["status"]) == {"trained"}
```

Do not run the white-occlusion branch.

- [ ] **Step 4: Generate grid rollouts and summaries**

Run `run_rollout_grid(..., skip_existing=False, continue_on_error=False)` for all 53 new experiments. Run `write_summary_tables` for the 49 MC experiments.

Expected: every rollout passes T20/checkpoint/initial-state validation; MC CSVs are written below `background_v008_t20`.

- [ ] **Step 5: Select MC only from validation rollouts**

Filter the MC combined table to `split == "val"` and `lambda_ctrl > 0`. Sort ascending by `max_l2_p95`, `max_l2_mean`, `ctrl_mse`, `alpha`, and `lambda_ctrl`. Select the first row. Select the no-control comparison at the same alpha and `lambda_ctrl == 0`. Do not inspect test metrics while selecting.

Expected: one nonzero-lambda winner and one same-alpha lambda-zero checkpoint.

---

### Task 7: Set Final Paths and Compute Six Gamma Values

**Files:**
- Modify: `config/sampling/{cartpole,mountain_car,pendulum}.json`
- Modify: full and smoke `config/starv_verification` JSON files
- Modify: `tests/test_starv_smoke_configs.py`
- Regenerate: canonical `dwm_trajectories_{saliency,saliency_lambda0}.npz`
- Generate: `results/conformal/t20/<env>_{saliency,saliency_lambda0}.json`
- Create: `report/2026-07-20-t20-conformal-results.md`

**Interfaces:**
- Consumes: selected matched checkpoints and held-out test trajectories.
- Produces: active path mappings, six Gamma values, and a textual provenance record.

- [ ] **Step 1: Update active checkpoint mappings**

CartPole uses `t20_matched/alpha_8/lambda_0.1` with control and same-alpha `lambda_0` without control. Pendulum uses `t20_matched/alpha_16/lambda_0.5` and same-alpha `lambda_0`. MountainCar uses the selected `background_v008_t20` pair. Use keys `saliency` and `saliency_lambda0` consistently. Set full/smoke StarV decoder paths to the with-control checkpoint, but do not execute verification.

- [ ] **Step 2: Test and commit checkpoint paths**

```bash
/home/tealab_shared/starv/env/starv_shared/bin/python -m unittest tests.test_experiment_configs tests.test_starv_smoke_configs tests.test_ablation -v
git add config/sampling config/starv_verification tests/test_starv_smoke_configs.py
git commit -m "config: select matched T20 decoder checkpoints"
```

- [ ] **Step 3: Generate six canonical T20 DWM rollouts**

For each environment, run `sampling.py config/sampling/<env>.json --decoder-variant saliency` and repeat with `--decoder-variant saliency_lambda0`.

Expected: six canonical NPZ files with correct T20/checkpoint metadata.

- [ ] **Step 4: Calibrate six held-out bounds**

For each environment/variant, run `conformal.py --alpha 0.05 --split test` and save under `results/conformal/t20/`.

Expected in every JSON: `n=400`, `rank=381`, `horizon=20`, `norm="L2"`, finite `gamma`; Pendulum records circular dim 0.

- [ ] **Step 5: Record measured provenance and commit it**

Create `report/2026-07-20-t20-conformal-results.md` containing the selected MC alpha/lambda, all six unrounded/rounded Gamma values, NPZ/checkpoint paths, split/rank, and commands. Explicitly state that coverage, robustness, tube area, certification time, cGAN, Braking, and containment were not computed.

```bash
git add -f report/2026-07-20-t20-conformal-results.md
git commit -m "docs: record T20 L2 conformal results"
```

---

### Task 8: Align the Paper with the Implemented Pipeline and Results

**Files:**
- Modify: `paper/main.tex`

**Interfaces:**
- Consumes: six measured Gamma values from Task 7.
- Produces: a compiling manuscript with correct heatmap loss, L2 theory, MC goal, and honest tables.

- [ ] **Step 1: Replace the intensity-threshold loss definition**

Define the heatmap from absolute controller-output change under patch occlusion, per-image min-max normalization, raw weight `1 + alpha_H H`, per-image mean normalization, and weighted pixel MSE. State patch size 8/stride 4, white occlusion for CP/Pendulum, and training-image background median for MC. Leave the Experimental Setup lambda sentence exactly unchanged.

- [ ] **Step 2: Unify the conformal section and proof**

Define `delta_i` as max over `t=0,...,20` of the wrapped state L2 error and `Gamma_{1-alpha}` as order statistic `ceil((n+1)(1-alpha))`. Replace active CP `L1`/`Delta` notation in the main section and proof appendix. State that every time-indexed reachable set is inflated. Do not change the unrelated binary safety-map matrix also named Gamma.

- [ ] **Step 3: Correct the MC goal**

Change `x >= 0.5` to `p_20 >= 0.6`; retain T20 and the already-correct full grid text.

- [ ] **Step 4: Replace `tab:cp_bounds`**

Use only CP, MC, and Pendulum columns and only DWM without/with `L_ctrl` rows. Populate the six cells from Task 7 JSON outputs with consistent rounding. Rewrite adjacent prose to make only measured three-environment claims. Do not add an MSE table.

- [ ] **Step 5: Add `tab:cert-summary` with explicit unknowns**

For CP/MC/Pendulum, place the selected with-control Gamma in both A1 and B1 because both use the same calibrated DWM. Put `--` in cGAN Gamma cells, every Braking cell, and every unavailable coverage/robustness/tube-area/time cell. Explain that repeated A1/B1 values are one shared calibration, not two estimates.

- [ ] **Step 6: Compile and commit**

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cd ..
git add -f paper/main.tex
git commit -m "docs: align paper with T20 L2 conformal results"
```

Expected: no new undefined references/citations and both tables render.

---

### Task 9: Final Verification and Scope Audit

**Files:**
- Verify only; edit earlier task files only for defects exposed by these checks.

**Interfaces:**
- Consumes: all implementation/results.
- Produces: completion evidence and confirmation that unrelated user work remains intact.

- [ ] **Step 1: Run the full unit suite**

```bash
MPLCONFIGDIR=/tmp/matplotlib-codex /home/tealab_shared/starv/env/starv_shared/bin/python -m unittest discover -s tests -v
```

Expected: `OK`; tests do not launch training or StarV.

- [ ] **Step 2: Audit active horizon/norm statements**

```bash
rg -n 'rollout_steps.*30|num_steps.*30|N, 31|t=0\.\.30|t=30' config README.md starv_verification tests
rg -n 'maximum.*L_1|Delta_\{1-' paper/main.tex README.md
```

Expected: no active T30 or CP L1/Delta statement. Inspect comments separately; do not rewrite historical reports or unrelated constants.

- [ ] **Step 3: Validate six final artifacts again**

Assert T20 shapes, finite arrays, equal initial states, exact checkpoint metadata, test `n=400`, rank 381, L2, and finite Gamma. Assert MC with/without checkpoints use identical alpha and differ only in lambda.

- [ ] **Step 4: Recompile from clean auxiliary output**

Remove only LaTeX auxiliary files produced during Task 8, then repeat the compile commands. Do not remove `paper/main.pdf` before a successful rebuild.

- [ ] **Step 5: Audit scope and report evidence**

```bash
git status --short
git log --oneline -8
```

Expected: the user's `.gitignore` edit remains uncommitted unless separately authorized; no cGAN, Braking, or containment result was generated. Report selected MC alpha/lambda, six Gamma values, tests, LaTeX result, commits, and all fields remaining `--`.
