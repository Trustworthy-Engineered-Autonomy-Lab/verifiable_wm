# Dataset Output Paths Design

## Goal

Make dataset generation use repository-relative output directories so the
current user can run the notebook without writing to `/datasets` or another
user's home directory.

## Scope

Change only the `output_dir` field in the six environment configs under:

- `config/make_decoder_dataset/`
- `config/sampling/`

The target mapping is:

| Environment | Output directory |
|---|---|
| CartPole | `datasets/cartpole/data/dataset_v1` |
| MountainCar | `datasets/mountain_car/data/dataset_v1` |
| Pendulum | `datasets/pendulum/data/dataset_v1` |

## Non-goals

- Do not change decoder training `state_space` ranges.
- Do not change StarV grid ranges or sampling counts.
- Do not change Python scripts or notebook cells.
- Do not clean stored notebook outputs.
- Do not address the duplicate functions currently present in `sampling.py`
  or `make_decoder_dataset.py` as part of this path-only change.

## Verification

Parse all six JSON files and assert that every `output_dir` equals the target
repository-relative path for its environment. Confirm that no fields other
than `output_dir` changed and that the user's existing notebook changes remain
untouched.
