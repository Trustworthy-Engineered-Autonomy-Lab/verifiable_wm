# Trajectory-Conformal Inflated Table Design

## Goal

Generate the `B1 (inflated)` row of Table II for CartPole, MountainCar,
Pendulum, and Braking System. Each benchmark receives a readable CSV and a
full-precision raw CSV in a new result directory. Existing result artifacts
remain unchanged.

## Statistical Protocol

For each benchmark, pair real and DWM trajectories from the validation split.
For trajectory \(i\), calculate

\[
\delta_i = \max_t \left\|s^{DWM}_{i,t,\mathcal D}
                       - s^{real}_{i,t,\mathcal D}\right\|_2 .
\]

CartPole uses dimensions `(0, 2)`. MountainCar, Pendulum, and Braking System
use dimensions `(0, 1)`. Pendulum angle differences use the shortest circular
difference with period \(2\pi\).

With `n=400` and `alpha=0.05`, calculate
`k = ceil((n + 1) * (1 - alpha)) = 381` and use the 381st ascending score as
the scalar \(\Gamma_{0.95}\).

Inflate both checked dimensions of every symbolic tube bound by the same
scalar \(\Gamma_{0.95}\). Evaluate the inflated tube against the independent
real test split.

## Inputs

| Benchmark | Symbolic tube | Real/DWM trajectory pair |
|---|---|---|
| CartPole | `results/cartpole/safety_result_big_cell_a8_lamda01.json` | `datasets/cartpole/big_cell/{real_trajectories,dwm_trajectories_saliency}.npz` |
| MountainCar | `results/mountain_car/safety_result_big_cell_best.json` | `datasets/mountain_car/big_cell_best/{real_trajectories,dwm_trajectories_saliency}.npz` |
| Pendulum | `results/pendulum/safety_result_big_cell_a16_lambda05.json` | `datasets/pendulum/big_cell/{real_trajectories,dwm_trajectories_saliency}.npz` |
| Braking System | `safety_results/brake_system/safety_result.json` | `safety_results/brake_system/{real_trajectories,dwm_trajectories_old}.npz` |

The Braking System uses the symbolic `old` DWM construction, not the cGAN
construction.

## Metrics

Report only one method row: `B1 (inflated)`.

- Coverage rate: fraction of valid real test trajectories whose worst signed
  tube margin is nonnegative.
- \(\Gamma_{1-\alpha}\): validation trajectory conformal radius.
- Robustness mean, minimum, and maximum: summary of real test trajectory
  signed margins against the inflated symbolic tube.
- Average normalized tube area: mean and population standard deviation across
  valid symbolic cells, averaged over future time steps.
- Valid real trajectories and valid tube cells: explicit denominators used by
  the metrics.
- Certification time: `--`, because the saved symbolic results contain no
  reliable runtime field.

## Outputs

Create the following directory for each benchmark:

`results/<env>/tube_comparison_trajectory_inflation/`

Each directory contains:

- `tube_table_metrics.csv`: paper-readable values with percentages and six
  decimal places.
- `tube_table_metrics_raw.csv`: full-precision numeric values and provenance
  fields.

Neither CSV contains A1 or cGAN rows. Existing
`results/<env>/tube_comparison_inflation/` directories are not modified.

## Validation

Generation must fail if paired trajectory shapes, horizons, actions, or
initial states do not match. After writing, reload every CSV and verify:

- exactly one data row named `B1 (inflated)`;
- calibration split is `val`, evaluation split is `test`, `n=400`,
  `alpha=0.05`, and rank is 381;
- all reported numeric metrics are finite;
- readable values agree with raw values after formatting;
- output files exist for all four benchmarks.
