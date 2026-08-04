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

Two baselines are compared against. A **cGAN** surrogate camera swaps the decoder for a generator
whose latent is verified as a bounded interval, which is what makes its tubes wider. An
**image-free trajectory predictor** skips images entirely: a transformer trained on state
trajectories, inflated by its calibration residuals. The decoder is the only model this
repository trains — the controllers, the cGAN generators and the pretrained braking decoder are
given artifacts that it verifies and evaluates.

## Benchmarks

| Benchmark | State | Image source | Initial grid | Horizon |
|---|---|---|---|---|
| CartPole | position, velocity, angle, angular velocity | `ContinuousCartPoleEnv` | 60 × 60 = 3600 cells | 20 |
| MountainCar | position, velocity | Gym `MountainCarContinuous-v0` | 80 × 80 = 6400 cells | 20 |
| Pendulum | angle, angular velocity | Gym `Pendulum-v1` | 100 × 50 = 5000 cells | 20 |
| Braking (AEBS) | distance, velocity | CARLA camera | 40 × 40 = 1600 cells | 10 |

CartPole and MountainCar pin their velocity dimensions to a single point, so all four grids are
two-dimensional in practice. Across the four benchmarks the DWM yields substantially tighter
reachable tubes than both baselines while meeting the 95% target coverage after conformal
inflation.

## Pipeline

Each stage reads one JSON config. Steps 1–3 build the decoder, so `<env>` there is one of the
three gym benchmarks — the braking decoder is a given artifact and its dataset is captured
differently (see Notes). Steps 4–7 accept all four.

```bash
# 1. Render the decoder training set, the StarV initial states, and the real
#    closed-loop trajectories the conformal step later calibrates against.
python make_decoder_dataset.py config/make_decoder_dataset/<env>.json

# 2. Precompute the controller-occlusion saliency maps H.
python saliency_map/scripts/precompute_saliency_maps.py --config config/train_decoder/<env>/saliency.json

# 3. Train the decoder. One --alpha/--lambda-ctrl value each trains a single
#    decoder; several values run the ablation grid over their cartesian product
#    and write alpha_lambda_grid.csv next to the runs.
python train_decoder.py config/train_decoder/<env>/saliency.json --alpha 8 --lambda-ctrl 0.1
python train_decoder.py config/train_decoder/<env>/saliency.json --alpha 4 8 16 --lambda-ctrl 0 0.1 0.5

# 4. Symbolic reachability over the grid, one MPI rank per shard of cells.
mpirun -n 16 python verify.py config/starv_verification/<env>.json

# 5. Empirical tube from three rollouts per cell, for the sampling-based comparison.
python sampling.py config/sampling/<env>.json

# 6. Signed margin, conformal quantile γ, inflated tube, containment plots.
python signed_tube_margin.py --env <env> --decoder dwm --construction symbolic

# 7. Seeded repeated evaluation across both models and both constructions.
python repeated_tube_evaluation.py run-all
```

Swap `<env>.json` for `<env>_g_mlp.json` in steps 4–5 and `--decoder cgan` in step 6 to run the
cGAN baseline; the trajectory-predictor baseline lives entirely under `trajectory_predictor/`.

**Reachability ground truth.** `tools/gt_grid_eval.py` (gym) and `tools/gt_brake_grid_eval.py`
(braking) run the true closed loop from real camera images and label every grid cell, and
`tools/compare_ground_truth.py` scores a verified safety map against those labels. A false
positive there would break soundness; conservatism shows up as recall below one.

## Repository layout

```text
make_decoder_dataset.py       training images, StarV initial states, real trajectories
train_decoder.py              DWM decoder training and the alpha x lambda ablation grid
sampling.py                   three-rollout-per-cell empirical tube construction
verify.py                     StarV / MPI symbolic reachability
signed_tube_margin.py         signed margin, conformal quantile, tube inflation
conformal.py                  finite-sample conformal quantile
compare.py                    containment scoring and tube plots, driven by signed_tube_margin
repeated_tube_evaluation.py   seeded repeated evaluation, produces the comparison table

model.py                      controller, DWM decoder, cGAN generator
dynamic.py                    analytic dynamics for the four benchmarks
env.py                        the CartPole renderer and the CARLA AEBS environment
utils.py                      sampling, image and dynamics-provenance helpers

saliency_map/                 occlusion saliency and its precomputation script
starv_verification/           StarV-side models, dynamics and verifiers
trajectory_predictor/         image-free transformer baseline
tools/                        braking dataset capture, ground truth, safety maps
tests/                        internal-consistency checks for the artifact formats
```

Run the checks with `python -m pytest tests/`. They validate artifact formats and config
invariants and need no GPU, dataset or CARLA server.

## Artifacts and configuration

Configs live under `config/`, grouped by stage: `make_decoder_dataset`, `train_decoder`,
`sampling`, `starv_verification`. Every path inside them is relative to the repository root.

Configs, datasets, trained weights and verification results are not distributed with the
repository. `config/`, `dwm_weight/` and `safety_results/` are expected to point at wherever
those artifacts live — a symlink into shared storage works — and `datasets/` and `results/` are
produced locally.

## Notes

- The braking benchmark is the one whose frames come from a simulator rather than a gym renderer,
  so `make_decoder_dataset.py` does not apply to it. Its dataset is captured over the verification
  grid with `tools/collect_brake_grid_dataset.py`, flattened by
  `tools/convert_brake_grid_dataset.py`, and split into the repo's format by
  `tools/make_brake_decoder_dataset.py`; the held-out real trajectories come from
  `tools/collect_brake_real_trajectories.py`. Every one of those steps needs a CARLA server.
- Braking frames go through PIL's `"L"` conversion before being resized, reproducing the capture
  pipeline the braking controller and decoder were trained on; the gym benchmarks convert with
  luma weights after rendering and resize with `F.interpolate`. The two are equivalent in intent
  but not interchangeable within one benchmark (`utils.carla_frame_to_gray` versus
  `utils.rgb_to_gray_01`).
- CartPole feeds only `[position, angle]` to the decoder; the full four-dimensional state is still
  used for the dynamics and for locating the initial grid cell.
- Angular differences for Pendulum are mapped to the shortest circular difference before any
  distance is computed, so trajectories crossing ±π do not produce spurious errors near 2π.
- All conformal scores use the L2 norm.
- Trajectory `.npz` files record the dynamics parameters actually in effect, so datasets produced
  under different integration steps can be told apart (`tools/check_dynamics_vintage.py`).

## Citation

Paper under review. Citation information will be added here once it is available.
