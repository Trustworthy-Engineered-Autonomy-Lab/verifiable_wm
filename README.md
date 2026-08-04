# Deterministic World Models for Closed-loop Reachability Analysis of End-to-End Vision-based Control

Reference implementation for the closed-loop verification of end-to-end vision-based controllers.
The camera is replaced by a Deterministic World Model (DWM): a latent-free neural decoder mapping
physical states directly to synthetic images.

Since the DWM admits no stochastic latent input, the closed loop

```text
state -> decoder -> controller -> dynamics -> next state
```

reduces to a feed-forward network composition, which symbolic reachability tools propagate without
the overapproximation induced by sampling a latent variable. The decoder is trained under a dual
objective combining saliency-weighted reconstruction with a control-consistency term. The
resulting reachable tubes are inflated by a distribution-free conformal bound, transferring the
surrogate guarantee to the real system with high probability.

## Method

1. **Decoder training.** A controller-occlusion saliency map `H` weights the reconstruction loss
   pixel-wise as `w = 1 + αH`. A control-consistency term penalises the discrepancy between the
   controller's action on the reconstructed image and on the real frame.
2. **Closed-loop verification.** StarV/ImageStar propagates each initial grid cell through
   decoder, controller and dynamics over a fixed horizon, yielding one reachable tube per cell.
3. **Guarantee transfer.** The signed margin of held-out real trajectories against the tube yields
   a conformal quantile γ. Inflating every cell by γ gives coverage of real trajectories with
   probability at least 1 − α.

Two baselines are evaluated for comparison. The cGAN baseline substitutes a conditional generator
for the decoder; its latent is verified as a bounded interval, which accounts for its wider tubes.
The image-free trajectory predictor bypasses image generation altogether, using a transformer
trained on state trajectories and inflated by its calibration residuals. The decoder is the only
model trained by this repository; controllers, cGAN generators and the braking decoder are
pretrained artifacts that the pipeline verifies and evaluates.

## Benchmarks

| Benchmark | State | Image source | Initial grid | Horizon |
|---|---|---|---|---|
| CartPole | position, velocity, angle, angular velocity | `ContinuousCartPoleEnv` | 60 × 60 = 3600 cells | 20 |
| MountainCar | position, velocity | Gym `MountainCarContinuous-v0` | 80 × 80 = 6400 cells | 20 |
| Pendulum | angle, angular velocity | Gym `Pendulum-v1` | 100 × 50 = 5000 cells | 20 |
| Braking (AEBS) | distance, velocity | CARLA camera | 40 × 40 = 1600 cells | 10 |

The CartPole and MountainCar grids fix their velocity dimensions to a single point, so all four
grids are effectively two-dimensional. Across the four benchmarks the DWM yields substantially
tighter reachable tubes than both baselines while meeting the 95% target coverage after conformal
inflation.

## Usage

Each stage is driven by a single JSON configuration. Stages 1 to 3 construct the decoder and
apply to the three gym benchmarks; the braking decoder is a pretrained artifact whose dataset is
captured separately (see Implementation notes). Stages 4 to 7 apply to all four benchmarks.

```bash
# 1. Decoder training set, StarV initial states, real closed-loop trajectories
python make_decoder_dataset.py config/make_decoder_dataset/<env>.json

# 2. Controller-occlusion saliency maps
python saliency_map/scripts/precompute_saliency_maps.py \
    --config config/train_decoder/<env>/saliency.json

# 3. Decoder training
python train_decoder.py config/train_decoder/<env>/saliency.json --alpha 8 --lambda-ctrl 0.1

# 4. Symbolic reachability, distributed over MPI ranks
mpirun -n 16 python verify.py config/starv_verification/<env>.json

# 5. Empirical tube from three rollouts per cell
python sampling.py config/sampling/<env>.json

# 6. Signed margin, conformal quantile, inflated tube, containment plots
python signed_tube_margin.py --env <env> --decoder dwm --construction symbolic

# 7. Seeded repeated evaluation over both models and both constructions
python repeated_tube_evaluation.py run-all
```

Supplying several values to `--alpha` or `--lambda-ctrl` runs the ablation grid over their
cartesian product, writing one run per combination together with a summary `alpha_lambda_grid.csv`:

```bash
python train_decoder.py config/train_decoder/<env>/saliency.json \
    --alpha 4 8 16 --lambda-ctrl 0 0.1 0.5
```

The cGAN baseline is obtained by substituting `<env>_g_mlp.json` in stages 4 and 5 and
`--decoder cgan` in stage 6. The trajectory-predictor baseline is self-contained under
`trajectory_predictor/`.

Reachability ground truth is computed by executing the true closed loop from real camera images:
`tools/gt_grid_eval.py` for the gym benchmarks and `tools/gt_brake_grid_eval.py` for the braking
system label every grid cell, and `tools/compare_ground_truth.py` scores a verified safety map
against those labels. False positives indicate a violation of soundness; conservatism is reflected
in recall below one.

Consistency checks are run with `python -m pytest tests/`. They validate artifact formats and
configuration invariants, requiring no GPU, dataset or CARLA server.

## Repository structure

```text
make_decoder_dataset.py       training images, StarV initial states, real trajectories
train_decoder.py              decoder training and the alpha-lambda ablation grid
sampling.py                   three-rollout-per-cell empirical tube construction
verify.py                     StarV/MPI symbolic reachability
signed_tube_margin.py         signed margin, conformal quantile, tube inflation
conformal.py                  finite-sample conformal quantile
compare.py                    containment scoring and tube plots
repeated_tube_evaluation.py   seeded repeated evaluation and comparison table

model.py                      controller, DWM decoder, cGAN generator
dynamic.py                    analytic dynamics for the four benchmarks
env.py                        CartPole renderer and CARLA AEBS environment
utils.py                      sampling, image and dynamics-provenance helpers

saliency_map/                 occlusion saliency and its precomputation script
starv_verification/           StarV-side models, dynamics and verifiers
trajectory_predictor/         image-free transformer baseline
tools/                        braking dataset capture, ground truth, safety maps
tests/                        internal-consistency checks
```

## Data and configuration

Configurations reside under `config/`, grouped by stage: `make_decoder_dataset`, `train_decoder`,
`sampling` and `starv_verification`. All paths within them are relative to the repository root.

Configurations, datasets, trained weights and verification results are not distributed with the
repository. The entries `config/`, `dwm_weight/` and `safety_results/` are expected to resolve to
the locations holding those artifacts, for which a symbolic link into shared storage is
sufficient. The directories `datasets/` and `results/` are produced locally.

## Implementation notes

- The braking benchmark obtains its frames from a simulator rather than a gym renderer, so
  `make_decoder_dataset.py` does not apply. Its dataset is captured over the verification grid by
  `tools/collect_brake_grid_dataset.py`, flattened by `tools/convert_brake_grid_dataset.py` and
  split into the repository format by `tools/make_brake_decoder_dataset.py`. Held-out real
  trajectories are collected by `tools/collect_brake_real_trajectories.py`. All of these stages
  require a CARLA server.
- Braking frames are converted through PIL's `"L"` mode before resizing, reproducing the capture
  pipeline under which the braking controller and decoder were trained. The gym benchmarks apply
  luma weights after rendering and resize with `F.interpolate`. The two conventions are equivalent
  in intent but not interchangeable within a single benchmark; see `utils.carla_frame_to_gray` and
  `utils.rgb_to_gray_01`.
- CartPole supplies only `[position, angle]` to the decoder. The full four-dimensional state
  remains in use for the dynamics and for locating the initial grid cell.
- Pendulum angular differences are mapped to the shortest circular difference prior to any
  distance computation, so trajectories crossing ±π incur no spurious error near 2π.
- All conformal scores are computed under the L2 norm.
- Trajectory `.npz` files record the dynamics parameters in effect at generation time, allowing
  datasets produced under different integration steps to be distinguished
  (`tools/check_dynamics_vintage.py`).

## Citation

Paper under review. Citation information will be added once available.
