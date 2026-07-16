# Three-Case StarV Smoke Validation Design

## Goal

Quickly exercise the complete StarV verification path for CartPole, Pendulum,
and MountainCar using each case's selected decoder checkpoint, without
overwriting formal verification results or changing the formal grid.

## Selected checkpoints

- CartPole: `dwm_weight/now_weight/cartpole/alpha_lambda_grid/alpha_8/lambda_0.1/seed_2025/decoder_best_total.pth`
- Pendulum: `dwm_weight/now_weight/pendulum/alpha_lambda_grid/alpha_16/lambda_0.5/seed_2025/decoder_best_total.pth`
- MountainCar: `dwm_weight/now_weight/mountain_car/alpha_lambda_grid/background/alpha_16/lambda_0.5/seed_2025/decoder_best_total.pth`

## Smoke grid

Each smoke configuration selects the four central formal cells along every
non-fixed grid dimension. It preserves the formal cell width instead of
covering the whole range with larger cells.

| Case | Formal varying ranges | Formal cell width | Smoke ranges | Smoke cells |
|---|---|---|---|---:|
| CartPole | position `[0.00, 0.60]`, angle `[0.06, 0.12]` | `0.06`, `0.006` | position `[0.18, 0.42]`, angle `[0.078, 0.102]` | 16 |
| Pendulum | theta `[1.00, 1.10]`, omega `[4.50, 4.60]` | `0.01`, `0.01` | theta `[1.03, 1.07]`, omega `[4.53, 4.57]` | 16 |
| MountainCar | position `[0.00, 0.10]`, velocity `[0.000, 0.010]` | `0.01`, `0.001` | position `[0.03, 0.07]`, velocity `[0.003, 0.007]` | 16 |

CartPole velocity and angular velocity stay fixed at zero. Every case keeps
its existing verifier class, safety threshold, and full 30-step horizon.
All three smoke configurations set `save_history=true` and
`early_stop=false` so every result contains the aligned sequence
`t=0, ..., 30` required by `compare.py`.

## Files and isolation

Add one configuration per case under `config/starv_verification/smoke/`.
Formal configurations under `config/starv_verification/*.json` remain
unchanged. Smoke results use these independent paths:

- `results/smoke/cartpole/safety_result.json`
- `results/smoke/pendulum/safety_result.json`
- `results/smoke/mountain_car/safety_result.json`

The smoke configurations do not include `starv_states`, because this run
only verifies cells and does not regenerate trajectory datasets.

## Execution

Run `verify.py` once per smoke configuration in the shared StarV Python
environment. A single MPI rank is sufficient for sixteen cells; if a working
MPI launcher is available, up to four ranks may be used. The sandboxed
runtime currently prevents MPI initialization, so execution may require an
approved unsandboxed command.

## Success criteria

For every case:

1. The selected decoder and controller checkpoints load successfully.
2. Exactly sixteen cells are generated and processed for all 30 steps.
3. The result JSON is written only under `results/smoke/<case>/`.
4. Every cell contains either a Boolean `result` or an `error_msg`.
5. Every error-free cell contains exactly 31 ordered bounds entries, one for
   each state time from `t=0` through `t=30`.
6. The final report gives cell counts, safe/unsafe/error counts, elapsed
   time, and observed bound/history sizes. A smoke result is an integration
   signal only; it is not a formal conclusion over the full configured
   initial-state range.

## Testing

Add a focused regression test that loads all three smoke configurations and
checks checkpoint selection, sixteen-cell grid cardinality, 30-step horizon,
full-history settings, and isolated output prefixes. Run it failing before
updating the configs, then passing afterward. Run the full existing test suite
before launching StarV.
