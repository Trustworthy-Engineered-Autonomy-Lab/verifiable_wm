# Alpha-Lambda Ablation and Rollout Notebook Design

## Goal

Turn `notebooks/train_decoder.ipynb` into the single interactive entry point for:

1. training one decoder;
2. training a complete saliency `alpha x lambda_ctrl` ablation grid;
3. running DWM-only rollouts for the main intensity/saliency baselines;
4. running DWM-only rollouts for every ablation checkpoint;
5. producing single-frame and closed-loop L2 result tables; and
6. deliberately promoting a selected experiment to the canonical saliency mainline.

StarV verification and conformal inflation are explicitly out of scope for this change.

## Experiment Scope

Run the same fixed-seed grid for CartPole and Pendulum:

```text
alpha       = [0.5, 1, 2, 4, 8, 16, 32]
lambda_ctrl = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
seed        = 2025
```

This produces 49 training runs and 49 DWM-only rollouts per environment, or 98 of each across both environments. Every trained grid point enters the rollout table; there is no seed screening or single-frame-metric shortlist.

The existing CartPole `saliency_alpha_sweep/` and `lambda_ablation/` directories remain historical records. The new two-dimensional grid uses a separate canonical layout and may supersede the historical conclusions without requiring the old directories to be deleted.

## Architecture

Use a thin notebook backed by a new reusable Python module, `ablation.py`.

- `train_decoder.py` remains the only implementation of decoder training.
- `sampling.py` remains the only implementation of DWM rollout generation.
- `ablation.py` owns experiment enumeration, configuration overrides, artifact paths, resume checks, metric collection, wrapped L2 calculation, and mainline promotion orchestration.
- `notebooks/train_decoder.ipynb` exposes readable cells that call these functions and display the resulting tables.

This preserves the repository convention that notebooks organize experiments while Python files contain reusable and testable logic.

## Artifact Layout

Each grid point is self-contained:

```text
dwm_weight/now_weight/<env>/alpha_lambda_grid/
  alpha_<alpha>/
    lambda_<lambda>/
      seed_2025/
        decoder_best_total.pth
        decoder_last.pth
        metrics.json
        dwm_trajectories_saliency.npz
```

Numbers use compact decimal formatting, for example `alpha_0.5`, `alpha_8`, `lambda_0.001`, and `lambda_0.1`.

The grid root also receives derived summaries:

```text
training_metrics.csv
rollout_l2.csv
combined_metrics.csv
```

These files are derived artifacts and can be regenerated from the per-experiment JSON/NPZ files.

Main baseline trajectories remain in the existing dataset directory:

```text
datasets/<env>/data/dataset_v1/dwm_trajectories_intensity.npz
datasets/<env>/data/dataset_v1/dwm_trajectories_saliency.npz
```

## Training Flow

For each environment and grid point, `ablation.py` will:

1. load `config/train_decoder/<env>/saliency.json`;
2. override `weight.alpha`, `lambda_ctrl`, `training.seed`, and `output_dir` on a copied configuration;
3. call the existing `train_decoder.train(...)` entry point; and
4. verify that `metrics.json`, `decoder_best_total.pth`, and `decoder_last.pth` exist.

The source JSON is never mutated by a grid run.

Training is resume-safe. With `skip_existing=True`, a grid point is skipped only when its metrics and both checkpoints exist and the saved metrics configuration matches the requested environment, alpha, lambda, seed, and output directory. A partial or mismatched experiment is rerun.

Long grids use `continue_on_error=True` by default: failures are collected, the remaining independent grid points continue, and the notebook prints a failure table at the end. A non-empty failure table means the grid is incomplete and blocks full rollout aggregation.

## DWM-Only Rollout Flow

For every completed grid point, `ablation.py` will:

1. load `config/sampling/<env>.json`;
2. set the decoder weights to that grid point's `decoder_best_total.pth`;
3. keep the decoder variant as `saliency`;
4. redirect `output_dir` to the grid point directory; and
5. call `sampling.generate_dataset(...)`.

The output is therefore `dwm_trajectories_saliency.npz` beside its checkpoint and metrics. Sampling continues to read the shared `starv_states.npz` and does not render real images or regenerate `transition_dataset.npz`.

Rollout resume checks require:

- the trajectory NPZ to exist;
- `variant` to equal `saliency`;
- `decoder_weights` to resolve to the requested checkpoint; and
- every split's initial state to match the corresponding `starv_states.npz` split.

A stale trajectory generated from another checkpoint is rerun instead of silently reused.

## L2 Metrics

For each experiment, compare its validation and test DWM trajectories with the corresponding shared real trajectories from the same initial states.

CartPole uses ordinary full-state L2 distance. Pendulum replaces the raw theta difference with the shortest signed circular difference before applying the full-state L2 norm:

```text
d_t = ||s_t_dwm - s_t_real||_2
```

The table reports:

- `mean_step_l2`: mean over all trajectories and states `t=0..30`;
- `final_l2`: mean at `t=30`;
- `max_l2_mean`: mean of each trajectory's maximum L2 deviation;
- `max_l2_p95`: 95th percentile of each trajectory's maximum L2 deviation.

The combined table joins these values with:

- alpha;
- lambda_ctrl;
- seed;
- best epoch;
- controller MSE; and
- pixel MSE.

The notebook displays validation and test flat tables plus alpha-by-lambda pivot tables for controller MSE, pixel MSE, `max_l2_mean`, and `max_l2_p95`. Validation results are used to choose hyperparameters; test results are reserved for reporting the selected configuration and baseline comparison.

These are descriptive model-comparison metrics. The notebook does not change them to the paper's L1 conformal score.

## Notebook Layout

`notebooks/train_decoder.ipynb` will be reorganized into these sections:

1. **Setup and experiment constants** — locate the repository, import helpers, and define the shared alpha/lambda grid.
2. **Single decoder training** — retain a simple `run_train(env, mode)` entry point for intensity/saliency one-off training.
3. **Full alpha-lambda training** — cells for Pendulum, CartPole, or both, with resume and failure reporting.
4. **Single-frame summaries** — main baseline summary plus full-grid training tables.
5. **Main baseline DWM-only rollout** — regenerate intensity and saliency trajectories with current canonical checkpoints.
6. **Full-grid DWM-only rollout** — run all completed grid checkpoints and report missing/failing cases.
7. **Wrapped L2 summaries** — create CSVs, flat tables, and pivot tables for each environment.
8. **Mainline promotion** — explicitly retrain and regenerate the selected saliency mainline only after the user chooses an experiment.

Expensive cells are never executed automatically when the notebook is opened. Each section shows the exact function call to run.

## Mainline Promotion and JSON Behavior

Grid training never overwrites the canonical mainline.

After reviewing the complete L2 table, the user may explicitly promote a chosen `(alpha, lambda_ctrl)` pair. Promotion will:

1. compare and display the selected row beside the current mainline row;
2. require an explicit `force=True` call from the notebook;
3. update `config/train_decoder/<env>/saliency.json` only if alpha or lambda changed;
4. retrain the selected deterministic `seed=2025` configuration into `dwm_weight/now_weight/<env>/saliency/`;
5. regenerate the canonical `dwm_trajectories_saliency.npz`; and
6. re-display the canonical metrics and L2 row.

`config/sampling/<env>.json` and `config/starv_verification/<env>.json` already point to the stable canonical saliency checkpoint path. They do not change during promotion. If the winning pair remains `(8, 0.1)`, no JSON file changes.

Promotion is not automatic. The notebook sorts and presents evidence, but it does not encode an arbitrary multi-metric definition of "better" or overwrite the mainline without an explicit user decision.

## Validation and Tests

Add unit tests that do not require GPU training:

- the Cartesian product contains exactly 49 unique experiments per environment;
- numeric directory names are stable;
- config overrides do not mutate source dictionaries;
- training resume checks accept complete matching artifacts and reject partial/mismatched ones;
- rollout resume checks reject a mismatched checkpoint provenance field;
- initial states must match `starv_states.npz`;
- ordinary CartPole L2 values are correct;
- Pendulum theta wrap-around uses the short circular difference;
- metric collection joins all expected columns and reports missing grid points.

Validation of the notebook includes:

- `nbformat` parsing;
- importing every new helper;
- the existing unit test suite;
- a lightweight synthetic end-to-end test from experiment enumeration through combined-table generation; and
- confirmation that no StarV command or real renderer is invoked by grid rollout orchestration.

The implementation must preserve the user's unrelated uncommitted modification to `notebooks/generate_dataset.ipynb`.
