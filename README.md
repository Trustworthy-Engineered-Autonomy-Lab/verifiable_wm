# Deterministic World Models for Closed-loop Reachability Analysis of End-to-End Vision-based Control

Reference implementation for verifying end-to-end vision-based controllers by replacing the
camera with a **Deterministic World Model (DWM)** — a latent-free neural decoder that maps
physical states directly to synthetic images.

Because the DWM has no stochastic latent input, the closed loop

```text
state --> decoder --> controller --> dynamics --> next state
```

becomes a plain feed-forward network composition that symbolic reachability tools can propagate
without the overapproximation introduced by sampling a latent variable. The DWM is trained with a
dual objective combining saliency-weighted reconstruction and a control-consistency term, and the
resulting reachable tubes are inflated by a distribution-free conformal bound so that the
surrogate guarantee transfers to the real system with high probability.

## Method

1. **Train the decoder.** A controller-occlusion saliency map `H` weights the reconstruction loss
   pixel-wise, `w = 1 + αH`; a control-consistency term keeps the controller's action on the
   reconstructed image close to its action on the real frame.
2. **Verify the closed loop.** StarV / ImageStar propagates each initial grid cell through
   decoder → controller → dynamics for a fixed horizon, producing a reachable tube per cell.
3. **Transfer the guarantee.** The signed margin of held-out real trajectories against the tube
   gives a conformal quantile γ, which inflates every cell so the tube covers real trajectories
   with probability ≥ 1 − α.

Two baselines are compared against: a **cGAN** surrogate camera (a generator with a bounded
latent interval) and an **image-free trajectory predictor** (a transformer trained directly on
state trajectories, inflated by its calibration residuals). Only the trajectory predictor is
trained here; the cGAN generators are pretrained artifacts, and this repository verifies and
evaluates them rather than reproducing their training.

## Benchmarks

| Benchmark | State | Image source | Initial grid | Horizon |
|---|---|---|---|---|
| CartPole | position, velocity, angle, angular velocity | Gym `CartPole` | 60 × 60 = 3600 cells | 20 |
| MountainCar | position, velocity | Gym `MountainCar` | 80 × 80 = 6400 cells | 20 |
| Pendulum | angle, angular velocity | Gym `Pendulum-v1` | 100 × 50 = 5000 cells | 20 |
| Braking (AEBS) | distance, velocity | CARLA camera | 40 × 40 = 1600 cells | 10 |

Across all four benchmarks the DWM yields substantially tighter reachable tubes than both
baselines while meeting the 95% target coverage after conformal inflation.

## Repository layout

```text
make_decoder_dataset.py       training images, StarV initial states, real trajectories
train_decoder.py              DWM decoder training (saliency + control-consistency loss)
sampling.py                   closed-loop rollout with the decoder in place of the renderer
verify.py                     StarV / MPI symbolic reachability
sampled_tube.py               per-cell sampling-based tube construction
signed_tube_margin.py         signed margin, conformal quantile, tube inflation
conformal.py                  finite-sample conformal quantile
compare.py                    containment scoring and tube plots, driven by signed_tube_margin
repeated_tube_evaluation.py   seeded repeated evaluation, produces the comparison table

model.py                      controller, DWM decoder, cGAN generator
dynamic.py                    analytic dynamics for the four benchmarks
env.py                        renderers and environment wrappers
utils.py                      sampling helpers and dynamics provenance

saliency_map/                 occlusion saliency and its precomputation script
starv_verification/           StarV-side models, dynamics and verifiers
trajectory_predictor/         image-free transformer baseline
tools/                        ground-truth rollouts, safety maps, dynamics provenance
tests/                        internal-consistency checks for the artifact formats
```

`train_decoder.py` takes the loss weights from the command line: one `--alpha` and one
`--lambda-ctrl` value train a single decoder, several values run the ablation grid over their
cartesian product and write `alpha_lambda_grid.csv` next to the runs.

```bash
python train_decoder.py config/train_decoder/cartpole/saliency.json --alpha 8 --lambda-ctrl 0.1
python train_decoder.py config/train_decoder/cartpole/saliency.json --alpha 4 8 16 --lambda-ctrl 0 0.1 0.5
```

Configurations live under `config/` grouped by stage (`make_decoder_dataset`, `train_decoder`,
`sampling`, `sampled_tube`, `starv_verification`). Configurations, datasets, trained
weights and verification results are not distributed with the repository; `config/`,
`dwm_weight/` and `safety_results/` are expected to point at wherever those artifacts live, and
every path inside the configuration files is relative to the repository root.

## Notes

- The braking benchmark is the one whose frames come from a simulator rather than a gym renderer,
  so `make_decoder_dataset.py` does not apply to it. Its dataset is captured over the verification
  grid with `tools/collect_brake_grid_dataset.py`, flattened by
  `tools/convert_brake_grid_dataset.py`, and split into the repo's format by
  `tools/make_brake_decoder_dataset.py`; held-out real trajectories come from
  `tools/collect_brake_real_trajectories.py`. Its decoder and controller are pretrained artifacts
  of the original AEBS study, like every other benchmark's controller and cGAN generator. Every
  step needs a CARLA server.
- CartPole feeds only `[position, angle]` to the decoder; the full four-dimensional state is still
  used for the dynamics and for locating the initial grid cell.
- Angular differences for Pendulum are mapped to the shortest circular difference before any
  distance is computed, so trajectories crossing ±π do not produce spurious errors near 2π.
- All conformal scores use the L2 norm.
- Trajectory `.npz` files record the dynamics parameters actually in effect, so datasets produced
  under different integration steps can be told apart (`tools/check_dynamics_vintage.py`).

## Citation

Paper under review. Citation information will be added here once it is available.
