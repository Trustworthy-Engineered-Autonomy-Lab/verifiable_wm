# T20 Conformal Results and Paper Alignment Design

## Objective

Make the CartPole, MountainCar, and Pendulum experiment pipeline internally consistent before updating the paper. All newly reported rollout and conformal results use a 20-step horizon, L2 trajectory error, the final full initial-state grids, and decoder checkpoints trained from traceable data. Braking System, cGAN, and StarV containment execution remain outside this work.

## Fixed Decisions

- The rollout horizon is `T = 20`: each trajectory stores 21 states (`t = 0, ..., 20`) and 20 actions.
- All conformal nonconformity scores use L2 distance.
- Existing 30-step result artifacts are historical only and must not populate the new paper tables.
- Unknown certification-table entries are written as `--`; they are not estimated or copied from stale runs.
- The Experimental Setup value of `lambda` is not edited in this work. Its reconciliation is deferred.
- MountainCar uses the goal `p_20 >= 0.6`.
- MountainCar decoder position training remains `[-1.20, 0.60]`; it is not expanded from rollout observations.
- MountainCar decoder velocity training becomes `[-0.08, 0.08]`.
- cGAN, Braking System, and containment computation are not modified or rerun.

## Initial-State Grids

The StarV configuration is the single source for the states subsequently consumed by sampling.

### CartPole

- `pos`: `[0.00, 0.60]`, 60 points
- `vel`: `[0.00, 0.00]`, 1 point
- `angle`: `[0.06, 0.12]`, 60 points
- `avel`: `[0.00, 0.00]`, 1 point
- Total: 3,600 grid cells

### MountainCar

- `pos`: `[-0.20, 0.60]`, 80 points
- `vel`: `[0.00, 0.08]`, 80 points
- Total: 6,400 grid cells

### Pendulum

- `theta`: `[1.00, 2.00]`, 100 points
- `omega`: `[4.50, 5.00]`, 50 points
- Total: 5,000 grid cells

## Configuration and Artifact Flow

The implementation changes active CartPole, MountainCar, and Pendulum configurations as follows:

1. `config/make_decoder_dataset/<env>.json` uses `rollout_steps = 20`.
2. `config/sampling/<env>.json` uses `rollout_steps = 20` and points only to checkpoints created for the current dataset version.
3. `config/starv_verification/<env>.json` uses `num_steps = 20` and the grids above.
4. Active smoke configurations and defaults that represent the experimental horizon are changed to 20. Unrelated numeric constants and historical reports are untouched.
5. Generated NPZ files record or preserve enough metadata to identify horizon, environment, split, decoder variant, and checkpoint. A configuration edit alone does not make an old artifact current.

Changing the StarV grid changes the initial-state distribution used by real and DWM rollouts after the states and trajectories are regenerated. It therefore changes the conformal result. Decoder training-state ranges affect decoder data, heatmaps, checkpoints, and downstream DWM rollouts, but not already generated artifacts retroactively.

## Decoder Regeneration

For each environment, no-control and control-loss checkpoints used in the paper must be trained from the same decoder dataset, heatmap, split, seed, architecture, and optimization settings. The only comparison variable is `lambda_ctrl`.

MountainCar requires this order:

1. Regenerate the decoder dataset with position `[-1.20, 0.60]` and velocity `[-0.08, 0.08]`.
2. Recompute the background-median occlusion heatmap from the regenerated training images.
3. Rerun the background-only 7-by-7 grid with seed 2025:
   - `alpha in {0.5, 1, 2, 4, 8, 16, 32}`
   - `lambda_ctrl in {0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5}`
4. Select the best nonzero-lambda setting using T20 L2 rollout behavior on the selection split, not weighted training loss or pixel MSE.
5. Use the selected alpha for both final models. The no-control model uses that alpha with `lambda_ctrl = 0`; the control-loss model uses the same alpha with the selected nonzero lambda.

The white-occlusion MountainCar branch is not rerun. The previous `(alpha = 16, lambda_ctrl = 0.5)` result is only a reference candidate, not an assumed winner on the regenerated data.

CartPole and Pendulum final no-control/control-loss pairs are also regenerated as matched pairs because their existing checkpoints do not share the same current dataset and heatmap provenance. This avoids attributing dataset-version differences to the control loss.

## Rollout and Conformal Computation

Real and DWM trajectories are paired by identical initial state and split. For trajectory `i`, define

```text
delta_i = max over t=0,...,20 of ||s_real[i,t] - s_dwm[i,t]||_2.
```

Pendulum applies the circular angular difference before computing its L2 norm. CartPole uses position and angle; MountainCar uses position and velocity; Pendulum uses theta and omega.

For calibration size `n` and `alpha = 0.05`, the finite-sample rank is

```text
k = ceil((n + 1) * (1 - alpha)).
Gamma_0.95 = the k-th ordered delta_i.
```

For `n = 400`, `k = 381`. No interpolation is used. Hyperparameter selection and conformal calibration must not reuse the same trajectories; the selected fixed model is calibrated on a held-out split. Output metadata identifies the input real trajectory, DWM trajectory, split, horizon, norm, sample count, rank, and checkpoint.

`compare.py --delta Gamma_0.95` may visualize or check an inflated tube, but it is not the source of Gamma. When containment work resumes, every time-indexed DWM reachable set is inflated by the conformal radius; this project does not run that containment stage.

## Paper Changes

The paper is aligned with the executable pipeline:

- Set rollout notation and descriptions to `T = 20`.
- Define individual trajectory errors as `delta_i` and the calibrated radius as `Gamma_{1-alpha}` using L2 throughout.
- Replace the old intensity-threshold loss description with the implemented occlusion heatmap loss: controller-output change produces a per-image normalized heatmap, raw weights are `1 + alpha_H H`, weights are normalized by their per-image mean, and reconstruction uses weighted pixel MSE. MountainCar uses the background-median occlusion baseline; CartPole and Pendulum use white occlusion.
- State the MountainCar final-step goal as `p_20 >= 0.6`.
- Do not add an MSE results table. Pixel MSE remains a training/reconstruction diagnostic and is not described as a conformal bound.
- Update the trajectory conformal table to CartPole, MountainCar, and Pendulum with two rows only: DWM without `L_ctrl` and DWM with `L_ctrl`. This table contains the six independently computed bounds.
- Keep the certification summary visible. A1 and B1 use the same selected with-control DWM within an environment, so they receive the same environment-specific `Gamma_0.95`; this is one calibration result displayed for both construction methods, not two separate estimates. Write `--` for unavailable coverage, robustness, normalized tube area, and certification time values. cGAN and every Braking value remain `--` in this work.
- Describe tube inflation at every time step, consistent with the theorem, rather than only at the final step.
- Leave the Experimental Setup lambda sentence unchanged for later discussion.

## Validation and Failure Handling

Before accepting regenerated results:

- Configuration tests assert T20 and the exact grid/range values.
- Trajectory validation rejects arrays that do not have 21 states and 20 actions, mismatched real/DWM shapes, non-finite values, mismatched initial states, or incompatible metadata.
- Conformal tests cover L2 scoring, Pendulum angle wrapping, max-over-time aggregation, the finite-sample rank, and the no-interpolation order statistic.
- The six final conformal outputs are traced to matched current checkpoints and T20 artifacts.
- Paper searches find no active T30 or L1 statement in the modified experimental/conformal sections, while unrelated physical constants and historical reports remain unchanged.
- The LaTeX document compiles with `--` entries and without undefined table references.

## Deliverables and Success Criteria

The work is complete when:

1. Active three-environment configurations encode the approved T20 horizon and exact ranges.
2. MountainCar has a regenerated dataset, background heatmap, 49-point ablation, and a selected matched checkpoint pair.
3. CartPole and Pendulum have matched no-control/control-loss checkpoints with current provenance.
4. New paired T20 real/DWM trajectories exist for all six model/environment combinations.
5. Six L2 `Gamma_0.95` values are reproducibly generated from held-out trajectories.
6. The paper reflects the loss, norm, horizon, goal, table scope, and known/unknown results above without changing the deferred lambda statement.
7. Targeted tests and LaTeX compilation pass.
