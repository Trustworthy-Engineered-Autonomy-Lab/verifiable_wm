# MountainCar Sampling Variant Design

## Scope

Add an explicit sampling variant to the MountainCar decoder configuration.
Do not change decoder weights, rollout behavior, output directory, or any
Python implementation.

## Design

Set `decoder.variant` to `a16_lambda05` in
`config/sampling/mountain_car.json`. `sampling.py` will then use this value for
both the trajectory filename and the NPZ provenance field while continuing to
load `weight_decoder/mountain_car/mc_a16_lambda05.pth`.

The next sampling run will write:

```text
datasets/mountain_car/data_cell_100/dwm_trajectories_a16_lambda05.npz
```

The existing `dwm_trajectories_old.npz` is not renamed or deleted.

## Verification

Parse the JSON, call `sampling.decoder_variant()`, and confirm that it returns
`a16_lambda05`. Confirm that `sampling.resolve_decoder_weights()` still returns
the existing MountainCar checkpoint path. No rollout is required to verify the
configuration change.
